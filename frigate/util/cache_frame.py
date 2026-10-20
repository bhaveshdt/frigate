"""Grab a still frame from a camera's recording cache."""

import json
import logging
import os
import subprocess as sp
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from glob import escape, glob
from typing import Any

import cv2
import numpy as np
import psutil

from frigate.const import CACHE_DIR, CACHE_SEGMENT_FORMAT, SUB_CACHE_TAG
from frigate.util.image import run_ffmpeg_snapshot
from frigate.util.segment_time import measure_segment_start

logger = logging.getLogger(__name__)

# a finished segment older than this is no longer a usable latest frame
CACHE_FRAME_MAX_AGE_S = 60
CACHE_FRAME_TIMEOUT_S = 10
# seconds before the end of the segment to take the frame from
CACHE_FRAME_SEEK_FROM_END_S = 0.5
SEGMENT_PROBE_TIMEOUT_S = 10
# A failed probe may be temporary, so it is tried again after a pause that
# doubles each time. After this many failures the file is given up on until it
# is rewritten
SEGMENT_PROBE_MAX_ATTEMPTS = 5
SEGMENT_PROBE_BACKOFF_S = 5.0
SEGMENT_PROBE_MAX_BACKOFF_S = 60.0
PROBE_CACHE_SIZE = 256

# Both tables are keyed by (ffprobe path, file, mtime), so a rewritten file
# starts fresh. Only a successful probe is kept as a result: caching a failure
# would hide a segment that a later probe can read. A failure is kept as
# (attempts so far, earliest time of the next attempt).
_probe_lock = threading.Lock()
_probe_results: dict[tuple[str, str, float], tuple[float, float]] = {}
_probe_failures: dict[tuple[str, str, float], tuple[int, float]] = {}

# the encoded frame of the newest finished segment per camera. A finished
# segment never changes, so its frame is made once however often it is asked
# for, instead of once per request. The encoded image is kept, not the decoded
# one: a full resolution BGR frame is tens of megabytes per camera
_latest_frame_cache: dict[str, tuple[str, bytes]] = {}
_latest_frame_lock = threading.Lock()
# one per camera, so requests that arrive together as a segment finishes share
# a single ffmpeg run instead of each starting their own
_latest_frame_camera_locks: dict[str, threading.Lock] = {}


@dataclass(frozen=True)
class CacheSegment:
    """A finished main stream cache segment and when its footage was captured."""

    path: str
    # epoch seconds of the first frame in the segment
    start: float
    # epoch seconds when the segment's footage ended
    end: float
    # presentation time of the first frame inside the file, normally zero
    # because the cache is cut with reset timestamps
    first_pts: float = 0.0


class SegmentListing(list[CacheSegment]):
    """Finished segments, oldest first, and where unreadable footage begins.

    blocked_from is the start of the earliest segment that is finished but
    could not be probed yet and may still be, or None. Footage from there on
    must not be treated as consumed, or a segment that only needed another
    probe would be skipped for good.
    """

    blocked_from: float | None = None


def get_cache_files_in_use(cache_dir: str = CACHE_DIR) -> set[str]:
    """Paths of cache files an ffmpeg process currently has open.

    This is how a segment that is still being written is told from a finished
    one when the caller does not know which processes write the cache. It
    walks every process, so a caller that knows its own ffmpeg processes should
    pass its own lookup to list_finished_cache_segments instead. The newest
    file name is not a substitute: it stays the newest after a camera is
    disabled or reconnects, long after its file was closed.
    """
    in_use: set[str] = set()

    for process in psutil.process_iter():
        try:
            if process.name() != "ffmpeg":
                continue

            for opened in process.open_files() or []:
                if opened.path.startswith(cache_dir):
                    in_use.add(opened.path)
        except psutil.Error:
            continue

    return in_use


def clear_probe_cache() -> None:
    """Forget every probe result and failure."""
    with _probe_lock:
        _probe_results.clear()
        _probe_failures.clear()


def _remember(table: dict, key: tuple[str, str, float], value: Any) -> None:
    table[key] = value

    while len(table) > PROBE_CACHE_SIZE:
        table.pop(next(iter(table)))


def probe_exhausted(ffprobe_path: str, path: str, mtime: float) -> bool:
    """Whether probing this file has been given up on."""
    with _probe_lock:
        attempts, _ = _probe_failures.get((ffprobe_path, path, mtime), (0, 0.0))

    return attempts >= SEGMENT_PROBE_MAX_ATTEMPTS


def _run_probe(ffprobe_path: str, path: str) -> tuple[float, float] | None:
    cmd = [
        ffprobe_path,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "format=start_time,duration",
        "-of",
        "json",
        path,
    ]

    try:
        result = sp.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=SEGMENT_PROBE_TIMEOUT_S,
            check=False,
        )
    except (OSError, sp.TimeoutExpired):
        return None

    if result.returncode != 0:
        return None

    try:
        info = json.loads(result.stdout).get("format", {})
        duration = float(info["duration"])
        first_pts = float(info.get("start_time", 0.0))
    except (ValueError, KeyError, TypeError):
        return None

    if duration <= 0:
        return None

    return first_pts, duration


def probe_segment(
    ffprobe_path: str, path: str, mtime: float
) -> tuple[float, float] | None:
    """Return (first pts, duration) of a segment, or None if it is unreadable.

    A successful result is remembered. A failure is not: it is tried again
    after a growing pause, up to SEGMENT_PROBE_MAX_ATTEMPTS times, and calls
    made during the pause return None without running ffprobe. The mtime only
    keys the tables so a rewritten file is probed again.
    """
    key = (ffprobe_path, path, mtime)
    now = time.time()

    with _probe_lock:
        known = _probe_results.get(key)

        if known is not None:
            return known

        attempts, retry_at = _probe_failures.get(key, (0, 0.0))

    if attempts >= SEGMENT_PROBE_MAX_ATTEMPTS or now < retry_at:
        return None

    probed = _run_probe(ffprobe_path, path)

    with _probe_lock:
        if probed is not None:
            _remember(_probe_results, key, probed)
            _probe_failures.pop(key, None)
            return probed

        attempts += 1
        pause = min(
            SEGMENT_PROBE_BACKOFF_S * 2 ** (attempts - 1), SEGMENT_PROBE_MAX_BACKOFF_S
        )
        _remember(_probe_failures, key, (attempts, now + pause))

    if attempts >= SEGMENT_PROBE_MAX_ATTEMPTS:
        logger.error(
            "Giving up on cache segment %s after %s failed probes", path, attempts
        )

    return None


def _main_stream_files(camera_name: str, cache_dir: str) -> list[tuple[float, str]]:
    """(start from the name, path) of a camera's main stream cache files.

    Main segments are {camera}@{date}, sub stream segments {camera}@sub@{date}.
    A file whose name is not a valid segment name is not a segment. Replay and
    the latest frame both pick their footage here, so they always agree on
    which files count.
    """
    files: list[tuple[float, str]] = []

    for path in glob(os.path.join(cache_dir, f"{escape(camera_name)}@*.mp4")):
        prefix, date = os.path.splitext(os.path.basename(path))[0].rsplit(
            "@", maxsplit=1
        )

        if prefix != camera_name or prefix.endswith(SUB_CACHE_TAG):
            continue

        try:
            name_start = datetime.strptime(date, CACHE_SEGMENT_FORMAT).timestamp()
        except ValueError:
            continue

        files.append((name_start, path))

    return files


def list_finished_cache_segments(
    camera_name: str,
    cache_dir: str = CACHE_DIR,
    ffprobe_path: str = "ffprobe",
    in_use: Callable[[str], set[str]] = get_cache_files_in_use,
) -> list[CacheSegment]:
    """Main stream cache segments ffmpeg has finished writing, oldest first.

    A file an ffmpeg process still has open is being written. An mp4 only gets
    its moov atom when it is closed, so it cannot be decoded yet and is left
    out. Every segment is timed from its own file, so a gap between segments
    does not stretch the footage on either side of it.

    A finished segment that cannot be probed is left out too. While probing it
    may still succeed, the listing says where that footage begins in
    blocked_from, so the caller does not treat it as consumed.
    """
    found: list[tuple[float, str, float]] = []
    open_files = in_use(cache_dir)

    for name_start, path in _main_stream_files(camera_name, cache_dir):
        if path in open_files:
            continue

        try:
            mtime = os.path.getmtime(path)
        except OSError:
            # the recording maintainer moved or removed it
            continue

        found.append((name_start, path, mtime))

    found.sort()
    segments = SegmentListing()

    for name_start, path, mtime in found:
        probed = probe_segment(ffprobe_path, path, mtime)

        if probed is None:
            logger.debug("Unable to probe cache segment %s", path)

            if segments.blocked_from is None and not probe_exhausted(
                ffprobe_path, path, mtime
            ):
                segments.blocked_from = name_start

            continue

        first_pts, duration = probed
        measured = measure_segment_start(name_start, mtime, duration)
        start = name_start if measured is None else measured
        segments.append(CacheSegment(path, start, start + duration, first_pts))

    return segments


def get_latest_finished_cache_segment(
    camera_name: str,
    cache_dir: str = CACHE_DIR,
    max_age: float = CACHE_FRAME_MAX_AGE_S,
    now: float | None = None,
) -> str | None:
    """Return the newest main stream cache segment ffmpeg has finished writing.

    The newest segment is still being written and an mp4 is only readable once
    it is closed, so the one before it is used.
    """
    segments: list[tuple[float, str]] = []

    for _, path in _main_stream_files(camera_name, cache_dir):
        try:
            segments.append((os.path.getmtime(path), path))
        except OSError:
            # the recording maintainer moved or removed it
            continue

    if len(segments) < 2:
        return None

    segments.sort()
    mtime, path = segments[-2]

    if (time.time() if now is None else now) - mtime > max_age:
        return None

    return path


def get_latest_cache_frame(ffmpeg: Any, camera_name: str) -> np.ndarray | None:
    """Decode the last frame of the latest finished cache segment as a BGR image.

    Lets stills be served for a camera whose detect stream is idle, without
    decoding anything until a frame is asked for. The segment ended when the
    next one began, so the frame is at most one segment length old. The encoded
    frame is kept until a newer segment finishes, so polling it does not run
    ffmpeg again.
    """
    path = get_latest_finished_cache_segment(camera_name)

    if path is None:
        return None

    with _latest_frame_lock:
        camera_lock = _latest_frame_camera_locks.setdefault(
            camera_name, threading.Lock()
        )

    with camera_lock:
        with _latest_frame_lock:
            cached = _latest_frame_cache.get(camera_name)

        if cached is not None and cached[0] == path:
            image_data: bytes | None = cached[1]
        else:
            image_data, error = run_ffmpeg_snapshot(
                ffmpeg,
                path,
                "mjpeg",
                timeout=CACHE_FRAME_TIMEOUT_S,
                seek_from_end=CACHE_FRAME_SEEK_FROM_END_S,
            )

            if not image_data:
                logger.debug("Unable to read a frame from %s: %s", path, error)
                return None

            with _latest_frame_lock:
                _latest_frame_cache[camera_name] = (path, image_data)

    # callers draw on the frame, so each gets a decode of its own
    frame = cv2.imdecode(np.frombuffer(image_data, dtype=np.uint8), cv2.IMREAD_COLOR)

    if frame is None:
        with _latest_frame_lock:
            _latest_frame_cache.pop(camera_name, None)

    return frame

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
from functools import lru_cache
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

# the decoded frame of the newest finished segment per camera. A finished
# segment never changes, so its frame is decoded once however often it is asked
# for, instead of once per request
_latest_frame_cache: dict[str, tuple[str, np.ndarray]] = {}
_latest_frame_lock = threading.Lock()


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


def get_cache_files_in_use(cache_dir: str = CACHE_DIR) -> set[str]:
    """Paths of cache files an ffmpeg process currently has open.

    This is how the recording maintainer tells a segment that is still being
    written from a finished one. The newest file name is not a substitute: it
    stays the newest after a camera is disabled or reconnects, long after its
    file was closed.
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


@lru_cache(maxsize=256)
def probe_segment(
    ffprobe_path: str, path: str, mtime: float
) -> tuple[float, float] | None:
    """Return (first pts, duration) of a segment, or None if it is unreadable.

    The mtime only keys the cache so a rewritten file is probed again.
    """
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
    """
    found: list[tuple[float, str, float]] = []
    open_files = in_use(cache_dir)

    for path in glob(os.path.join(cache_dir, f"{escape(camera_name)}@*.mp4")):
        prefix, date = os.path.splitext(os.path.basename(path))[0].rsplit(
            "@", maxsplit=1
        )

        # main segments are {camera}@{date}, sub segments {camera}@sub@{date}
        if prefix != camera_name or prefix.endswith(SUB_CACHE_TAG):
            continue

        try:
            name_start = datetime.strptime(date, CACHE_SEGMENT_FORMAT).timestamp()
        except ValueError:
            continue

        if path in open_files:
            continue

        try:
            mtime = os.path.getmtime(path)
        except OSError:
            # the recording maintainer moved or removed it
            continue

        found.append((name_start, path, mtime))

    found.sort()
    segments: list[CacheSegment] = []

    for name_start, path, mtime in found:
        probed = probe_segment(ffprobe_path, path, mtime)

        if probed is None:
            logger.debug("Unable to probe cache segment %s", path)
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

    for path in glob(os.path.join(cache_dir, f"{escape(camera_name)}@*.mp4")):
        # main segments are {camera}@{date}, sub segments {camera}@sub@{date}
        prefix = os.path.basename(path).rsplit("@", maxsplit=1)[0]

        if prefix != camera_name or prefix.endswith(SUB_CACHE_TAG):
            continue

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
    next one began, so the frame is at most one segment length old. The decoded
    frame is kept until a newer segment finishes, so polling it does not run
    ffmpeg again.
    """
    path = get_latest_finished_cache_segment(camera_name)

    if path is None:
        return None

    with _latest_frame_lock:
        cached = _latest_frame_cache.get(camera_name)

    if cached is not None and cached[0] == path:
        return cached[1].copy()

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

    frame = cv2.imdecode(np.frombuffer(image_data, dtype=np.uint8), cv2.IMREAD_COLOR)

    if frame is not None:
        with _latest_frame_lock:
            _latest_frame_cache[camera_name] = (path, frame)

        # callers draw on the frame, the cached one stays untouched
        return frame.copy()

    return None

"""Replays recorded main stream footage through the detection pipeline.

In detect mode replay nothing decodes a stream continuously. A trigger (the
detect toggle driven by a hardware event) makes the runner decode recent
finished cache segments of the main stream, starting a little before the
trigger and continuing until after it is released. Frames keep the time they
were captured at.

The cache is cut with reset timestamps, so a frame's capture time is the start
of its segment plus its presentation time in the file. ffmpeg reports that time
for every frame it keeps (see CameraConfig.get_replay_ffmpeg_cmd), and it is
read back from stderr in step with the frames on stdout.
"""

import logging
import math
import queue
import re
import subprocess as sp
import threading
import time
from collections import deque
from collections.abc import Callable
from functools import partial
from multiprocessing.synchronize import Event as MpEvent
from typing import Any

from frigate.config import CameraConfig
from frigate.const import CACHE_DIR, STREAM_TYPE_MAIN
from frigate.util.builtin import get_record_segment_time
from frigate.util.cache_frame import CacheSegment, list_finished_cache_segments
from frigate.util.image import FrameManager, SharedMemoryFrameManager

logger = logging.getLogger(__name__)

# how long to keep waiting for a segment that never finishes, in segment lengths
GIVE_UP_SEGMENTS = 3
GIVE_UP_MARGIN_S = 30
# windows older than this can no longer match a frame still waiting to be replayed
WINDOW_RETENTION_S = 3600
# ffmpeg is killed if a single frame takes this long to arrive
READ_HANG_TIMEOUT_S = 60
# how long to wait for the capture time that belongs to a frame already read
TIME_WAIT_S = 5
# how long to give ffmpeg to exit by itself after it closes its output
EXIT_WAIT_S = 10
STDERR_TAIL_LINES = 20

SHOWINFO_TIME = re.compile(r"\[info\]\s+n:\s*\d+\s+pts:\s*\S+\s+pts_time:(\S+)")


def frame_timestamp(segment: CacheSegment, pts: float) -> float:
    """Capture time of a frame from its presentation time in the segment."""
    return segment.start + (pts - segment.first_pts)


class FfmpegStderr(threading.Thread):
    """Reads ffmpeg stderr, splitting frame times from everything else.

    showinfo logs one line per kept frame, in the order the frames are written
    to stdout. Warnings and errors are kept so a failed segment can say why.
    """

    def __init__(self, stream: Any) -> None:
        threading.Thread.__init__(self, name="replay_stderr", daemon=True)
        self.stream = stream
        # one entry per frame, None when the time could not be read
        self.times: queue.Queue[float | None] = queue.Queue()
        self.errors: deque[str] = deque(maxlen=STDERR_TAIL_LINES)

    def run(self) -> None:
        for raw in iter(self.stream.readline, b""):
            line = raw.decode(errors="replace").strip()

            if not line:
                continue

            if "showinfo" in line:
                match = SHOWINFO_TIME.search(line)

                if match:
                    try:
                        self.times.put(float(match.group(1)))
                    except ValueError:
                        self.times.put(None)

                continue

            # progress and stream info, plus the banner libva prints on stderr
            if "[info]" in line or line.startswith("libva info"):
                continue

            self.errors.append(line)


class DetectWindows:
    """Spans of time in which detection was wanted, from toggle events.

    Replay runs behind real time, so by the time a frame is replayed the live
    detect toggle has usually flipped back. Each frame is judged against these
    spans instead, using the time it was captured at.
    """

    def __init__(self) -> None:
        self._windows: list[list[float]] = []

    def open(self, start: float) -> None:
        if self._windows:
            last = self._windows[-1]

            if last[1] == float("inf"):
                return

            # a trigger that lands inside or right after the previous span extends it
            if start <= last[1]:
                last[1] = float("inf")
                return

        self._windows.append([start, float("inf")])

    def close(self, end: float) -> None:
        if self._windows and self._windows[-1][1] == float("inf"):
            self._windows[-1][1] = end

    def contains(self, timestamp: float) -> bool:
        return any(start <= timestamp <= end for start, end in self._windows)

    def prune(self, before: float) -> None:
        self._windows = [w for w in self._windows if w[1] >= before]


class DetectReplayRunner(threading.Thread):
    """Decodes finished main stream cache segments for one camera on demand."""

    def __init__(
        self,
        config: CameraConfig,
        shm_frame_count: int,
        frame_queue: Any,
        camera_fps: Any,
        stop_event: MpEvent,
        frame_manager: FrameManager | None = None,
        cache_dir: str = CACHE_DIR,
        list_segments: Callable[..., list[CacheSegment]] | None = None,
    ) -> None:
        threading.Thread.__init__(self, name=f"replay:{config.name}", daemon=True)
        self.config = config
        self.shm_frame_count = shm_frame_count
        self.frame_queue = frame_queue
        self.camera_fps = camera_fps
        self.stop_event = stop_event
        self.frame_manager = frame_manager or SharedMemoryFrameManager()
        self.cache_dir = cache_dir
        self.list_segments = list_segments or partial(
            list_finished_cache_segments, ffprobe_path=config.ffmpeg.ffprobe_path
        )

        self.frame_size = config.frame_shape_yuv[0] * config.frame_shape_yuv[1]
        self.fps = max(config.detect.fps, 1)
        self.segment_time = float(get_record_segment_time(config, STREAM_TYPE_MAIN))
        # Footage already cut into a finished segment is replayed from the
        # start of the segment that holds this moment, so detection can see up
        # to one segment length before the trigger. The first frames of a
        # replay can be older than that, never newer.
        self.pre_roll = self.segment_time
        self.tail = self._tail_seconds()

        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._local_stop = threading.Event()
        self._windows = DetectWindows()
        self._demand = False
        self._range_start = 0.0
        self._replay_until = 0.0
        self._covered = 0.0
        self._replayed: set[tuple[str, float]] = set()
        self._process: Any = None
        self._read_started: float | None = None
        self._frame_index = 0
        self._warned_no_source = False
        self.active = False

    def _tail_seconds(self) -> float:
        """Footage to replay after a release.

        The tracker only ends objects by processing frames with no detections,
        and Frigate only reports motion OFF from a frame mqtt_off_delay after
        the last motion, so the replay has to outlive both.
        """
        max_disappeared = self.config.detect.max_disappeared or self.fps * 5
        mqtt_off_delay = getattr(self.config.motion, "mqtt_off_delay", 0) or 0
        return max(max_disappeared / self.fps, float(mqtt_off_delay)) + 1

    def trigger(self, now: float) -> None:
        """A hardware event wants detection from just before now onwards."""
        with self._lock:
            if not self._demand:
                if self._covered >= self._replay_until:
                    # nothing is pending, so this starts a new session
                    self._range_start = now - self.pre_roll
                self._windows.open(now - self.pre_roll)

            self._demand = True
            self._replay_until = float("inf")

        self._wake.set()

    def release(self, now: float) -> None:
        """The hardware event is over, replay a tail so objects can end."""
        with self._lock:
            if not self._demand:
                return

            self._demand = False
            self._windows.close(now)
            self._replay_until = now + self.tail

        self._wake.set()

    def stop(self) -> None:
        self._local_stop.set()
        self._wake.set()
        process = self._process

        if process is not None:
            # the reader sees end of file and the owning thread reaps the process
            process.terminate()

        if self.is_alive() and threading.current_thread() is not self:
            self.join(timeout=5)

            if self.is_alive():
                logger.warning("Replay thread for %s did not stop in time", self.name)

    def _stopped(self) -> bool:
        return self.stop_event.is_set() or self._local_stop.is_set()

    def _pending(self) -> bool:
        with self._lock:
            return self._demand or self._covered < self._replay_until

    def _detect_wanted(self, timestamp: float) -> bool:
        with self._lock:
            return self._windows.contains(timestamp)

    def run(self) -> None:
        while not self._stopped():
            self._wake.wait(1.0)
            self._wake.clear()

            if self._stopped() or not self._pending():
                continue

            try:
                self._replay_pending()
            except Exception:
                logger.exception("Replay failed for %s", self.config.name)

    def _replay_pending(self) -> None:
        """Replay every finished segment that overlaps the wanted range."""
        while not self._stopped():
            segments = self.list_segments(self.config.name, self.cache_dir)
            now = time.time()

            with self._lock:
                range_start = self._range_start
                until = self._replay_until
                self._replayed &= {(s.path, s.start) for s in segments}
                self._windows.prune(now - WINDOW_RETENTION_S)

            # every segment carries its own span, so gaps between segments
            # neither stretch footage nor hide it
            next_segment = next(
                (
                    segment
                    for segment in segments
                    if (segment.path, segment.start) not in self._replayed
                    and segment.end > range_start
                    and segment.start < until
                ),
                None,
            )

            if next_segment is None:
                self._finish_if_caught_up(now, until)
                return

            self._replay_segment(next_segment)

            with self._lock:
                self._replayed.add((next_segment.path, next_segment.start))
                self._covered = max(self._covered, next_segment.end)

    def _finish_if_caught_up(self, now: float, until: float) -> None:
        """Stop waiting for footage that is never going to be finished."""
        with self._lock:
            if self._demand or until == float("inf"):
                return

            give_up_at = until + GIVE_UP_SEGMENTS * self.segment_time + GIVE_UP_MARGIN_S

            if now > give_up_at:
                self._covered = max(self._covered, until)

    def _start_process(self, cmd: list[str]) -> sp.Popen:
        return sp.Popen(
            cmd,
            stdin=sp.DEVNULL,
            stdout=sp.PIPE,
            stderr=sp.PIPE,
            bufsize=self.frame_size,
            start_new_session=True,
        )

    def _guard(self, process: sp.Popen, done: threading.Event) -> None:
        """Kill ffmpeg if it stops producing frames without exiting."""
        while not done.wait(1.0):
            started = self._read_started

            if started is not None and time.time() - started > READ_HANG_TIMEOUT_S:
                logger.warning(
                    "Replay ffmpeg for %s produced no frame in %s seconds, killing it",
                    self.config.name,
                    READ_HANG_TIMEOUT_S,
                )
                process.kill()
                return

    @staticmethod
    def _next_time(reader: FfmpegStderr) -> float | None:
        """Capture time of the frame that was just read from stdout."""
        try:
            pts = reader.times.get(timeout=TIME_WAIT_S)
        except queue.Empty:
            return None

        if pts is None or not math.isfinite(pts):
            return None

        return pts

    @staticmethod
    def _reap(process: sp.Popen, ended: bool) -> int | None:
        """Wait for ffmpeg to exit and return its exit code.

        When stdout reached end of file ffmpeg is already finishing, and its
        own exit code says whether the segment decoded cleanly. Otherwise it is
        still running and is stopped.
        """
        return_code = None

        if ended:
            try:
                return_code = process.wait(timeout=EXIT_WAIT_S)
            except sp.TimeoutExpired:
                pass

        if process.poll() is None:
            process.terminate()

            try:
                process.wait(timeout=5)
            except sp.TimeoutExpired:
                process.kill()
                process.wait()

        if process.stdout is not None:
            process.stdout.close()

        return return_code if return_code is not None else process.returncode

    def _replay_segment(self, segment: CacheSegment) -> None:
        cmd = self.config.get_replay_ffmpeg_cmd(segment.path)

        if cmd is None:
            # the config validator and the runtime updater both refuse replay
            # without a record input, so this is a state that should not occur
            if not self._warned_no_source:
                logger.warning(
                    "Detect mode replay for %s has no input with the record role to decode",
                    self.config.name,
                )
                self._warned_no_source = True

            return

        logger.debug("Replaying %s from %s", self.config.name, segment.path)
        process = self._start_process(cmd)
        reader: FfmpegStderr | None = None
        done = threading.Event()
        self._process = process
        self.active = True
        started = time.time()
        frames = 0
        untimed = 0
        ended = False
        truncated = False

        # everything after the process exists is inside the try, so a failure
        # while setting up the helper threads cannot leave ffmpeg running
        try:
            reader = FfmpegStderr(process.stderr)
            reader.start()
            threading.Thread(
                target=self._guard,
                args=(process, done),
                name="replay_guard",
                daemon=True,
            ).start()

            while not self._stopped():
                self._read_started = time.time()
                data = process.stdout.read(self.frame_size)
                self._read_started = None

                # a short read is the end of the segment, or a broken one
                if len(data) < self.frame_size:
                    ended = True
                    truncated = len(data) > 0
                    break

                pts = self._next_time(reader)

                if pts is None:
                    untimed += 1
                    continue

                if self._feed(data, frame_timestamp(segment, pts)):
                    frames += 1

                elapsed = time.time() - started
                if elapsed > 0:
                    self.camera_fps.value = frames / elapsed
        finally:
            self._read_started = None
            done.set()
            return_code = self._reap(process, ended)

            if reader is not None and reader.is_alive():
                reader.join(timeout=2)

            self._process = None
            self.active = False
            self.camera_fps.value = 0

        if self._stopped():
            return

        if untimed:
            logger.warning(
                "Skipped %s frames without a capture time from %s",
                untimed,
                segment.path,
            )

        if return_code != 0 or truncated:
            logger.warning(
                "Replay of %s ended early (exit code %s, %s frames): %s",
                segment.path,
                return_code,
                frames,
                "; ".join(reader.errors) or "no error output",
            )
        elif frames == 0:
            logger.debug("No frames replayed from %s", segment.path)

    def _feed(self, data: bytes, timestamp: float) -> bool:
        """Hand one frame to the tracker, returns whether it was sent."""
        frame_name = f"{self.config.name}_frame{self._frame_index}"
        frame_buffer = self.frame_manager.write(frame_name)

        if frame_buffer is None:
            logger.warning("No shared memory for frame %s", frame_name)
            return False

        # only the writer's mapping is closed, the tracker still owns the buffer
        try:
            frame_buffer[:] = data

            # unlike live capture, replay waits for the tracker so no frame is dropped
            item = (frame_name, timestamp, self._detect_wanted(timestamp))
            while not self._stopped():
                try:
                    self.frame_queue.put(item, True, 0.5)
                    break
                except queue.Full:
                    continue
        finally:
            self.frame_manager.close(frame_name)

        self._frame_index = (self._frame_index + 1) % self.shm_frame_count
        return True

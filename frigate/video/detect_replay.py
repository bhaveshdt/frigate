"""Replays recorded main stream footage through the detection pipeline.

In detect mode replay nothing decodes a stream continuously. A trigger (the
detect toggle driven by a hardware event) makes the runner decode recent
finished cache segments of the main stream, starting a little before the
trigger and continuing until after it is released. Frames keep the time they
were captured at.

The replay is bounded by capture time, not by segment. It stops decoding at the
end of the wanted range, and a cursor remembers how far the footage has been
consumed, so a later trigger resumes inside a partly replayed segment instead
of losing the footage that was not needed yet.

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
from dataclasses import dataclass
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
# how long stop() waits for the thread after asking ffmpeg to end, and again
# after killing it
STOP_GRACE_S = 5.0
# after a hardware decode failure the CPU recovered from, later segments decode
# on the CPU for this long before hardware is tried again
HARDWARE_RETRY_AFTER_S = 600
STDERR_TAIL_LINES = 20
# A segment that fails to replay is tried this many times, with a pause that
# doubles after each failure, before its footage is given up on and the cursor
# moves past it. Giving up is the only way unprocessed footage is skipped
SEGMENT_MAX_ATTEMPTS = 4
SEGMENT_RETRY_BACKOFF_S = 10.0
SEGMENT_RETRY_MAX_BACKOFF_S = 120.0
# A frame stamped further than this outside its own segment has a bad capture
# time. It is skipped: one such time would move the newest fed time out of reach
# of every real frame after it, and the tracker would silently lose them
SEGMENT_TIME_SLACK_S = 1.0
# The tracker's rates are zeroed once no frame was fed for this long. The tracker
# can still be working on the last frames when replay ends, and it writes the
# rates itself while it does
IDLE_STATS_DELAY_S = 2.0

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
        """Read stderr until it closes, queueing frame times and keeping errors."""
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
        """Start a span at start, or extend the last one if it is still open."""
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
        """End the span that is open, if any."""
        if self._windows and self._windows[-1][1] == float("inf"):
            self._windows[-1][1] = end

    def contains(self, timestamp: float) -> bool:
        """Whether detection was wanted at a capture time."""
        return any(start <= timestamp <= end for start, end in self._windows)

    def prune(self, before: float) -> None:
        """Forget spans that ended before a time."""
        self._windows = [w for w in self._windows if w[1] >= before]

    def snapshot(self) -> tuple[tuple[float, float], ...]:
        """The spans as they are now, for a replacement runner."""
        return tuple((start, end) for start, end in self._windows)

    def restore(self, windows: tuple[tuple[float, float], ...]) -> None:
        """Replace the spans with ones from a snapshot."""
        self._windows = [[start, end] for start, end in windows]


@dataclass(frozen=True)
class ReplaySession:
    """A replay session that was not over when its runner ended.

    A session lasts while detection is wanted and, after the release, until the
    tail has been replayed. The tail is what lets the tracker end its objects,
    so a replacement runner is given the session and carries on with it.
    """

    range_start: float
    # infinite while detection is still wanted
    replay_until: float
    demand: bool
    windows: tuple[tuple[float, float], ...]


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
        in_use: Callable[[str], set[str]] | None = None,
        cursor: float = 0.0,
        last_fed: float = 0.0,
        process_fps: Any = None,
        detection_fps: Any = None,
    ) -> None:
        threading.Thread.__init__(self, name=f"replay:{config.name}", daemon=True)
        self.config = config
        self.shm_frame_count = shm_frame_count
        self.frame_queue = frame_queue
        self.camera_fps = camera_fps
        # the tracker only writes these while it is given frames, so they are
        # zeroed here once replay has gone quiet
        self._tracker_rates = [
            rate for rate in (process_fps, detection_fps) if rate is not None
        ]
        self.stop_event = stop_event
        self.frame_manager = frame_manager or SharedMemoryFrameManager()
        self.cache_dir = cache_dir
        # in_use says which cache files are still being written. It is given
        # by the watchdog, which knows this camera's ffmpeg processes, so the
        # lookup does not have to scan every process in the container
        segment_args: dict[str, Any] = {"ffprobe_path": config.ffmpeg.ffprobe_path}

        if in_use is not None:
            segment_args["in_use"] = in_use

        self.list_segments = list_segments or partial(
            list_finished_cache_segments, **segment_args
        )

        self.frame_size = config.frame_shape_yuv[0] * config.frame_shape_yuv[1]
        # Every frame is read into this one buffer. Reading a frame as a new
        # bytes object allocates frame_size bytes (about 11 MB at 4096x1860)
        # per frame, and with a new set of helper threads for every segment
        # glibc keeps spreading those allocations over more malloc arenas that
        # never shrink, so the process memory only ever went up. Only the
        # replay thread uses it, and only one segment is decoded at a time.
        self._frame_buffer = bytearray(self.frame_size)
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
        # footage up to this time has been consumed, replay never goes back.
        # A runner that replaces a dead one starts from where it stopped
        self._cursor = cursor
        # capture time of the newest frame handed to the tracker, the tracker
        # needs frames in order and must never see one twice
        self._last_fed = last_fed
        self._process: Any = None
        self._read_started: float | None = None
        self._frame_index = 0
        self._warned_no_source = False
        self._warned_no_ledger = False
        # segments decode on the CPU until this time
        self._software_until = 0.0
        # failed attempts per segment, and the earliest time of the next try
        self._failures: dict[str, int] = {}
        self._retry_not_before = 0.0
        # whether the last decode stopped on a frame that could not be written
        self._frame_unsent = False
        # when the last frame was fed, and whether the tracker rates may still
        # hold values from it
        self._last_fed_at = 0.0
        self._rates_stale = False
        self.active = False

    def _tail_seconds(self) -> float:
        """Footage to replay after a release.

        The tracker only ends objects by processing frames with no detections,
        and Frigate only reports motion OFF from a frame mqtt_off_delay after
        the last motion, so the replay has to outlive both.
        """
        max_disappeared = self.config.detect.max_disappeared or self.fps * 5
        mqtt_off_delay = self.config.motion.mqtt_off_delay
        return max(max_disappeared / self.fps, float(mqtt_off_delay)) + 1

    def trigger(self, now: float) -> None:
        """A hardware event wants detection from just before now onwards."""
        with self._lock:
            if not self._demand:
                if self._cursor >= self._replay_until:
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

    def progress(self) -> tuple[float, float]:
        """How far replay got, as (cursor, last fed capture time).

        A runner that replaces this one is started from these so frames the
        tracker has already seen are not fed to it again.
        """
        with self._lock:
            return self._cursor, self._last_fed

    def session(self) -> ReplaySession | None:
        """The session in progress, or None when nothing is pending."""
        with self._lock:
            if not (self._demand or self._cursor < self._replay_until):
                return None

            return ReplaySession(
                self._range_start,
                self._replay_until,
                self._demand,
                self._windows.snapshot(),
            )

    def resume(self, session: ReplaySession) -> None:
        """Carry on a session that a previous runner did not finish."""
        with self._lock:
            self._range_start = session.range_start
            self._replay_until = session.replay_until
            self._demand = session.demand
            self._windows.restore(session.windows)

        self._wake.set()

    def stop(self) -> bool:
        """Stop the thread, killing ffmpeg if it ignores the request.

        Returns whether the thread ended. False is not the same as stopped: the
        thread may still write frames and change its progress, so nothing may
        take over its work until it has ended.
        """
        self._local_stop.set()
        self._wake.set()
        process = self._process

        if process is not None:
            # the reader sees end of file and the owning thread reaps the process
            process.terminate()

        if not self.is_alive():
            return True

        if threading.current_thread() is self:
            return False

        self.join(timeout=STOP_GRACE_S)

        if self.is_alive():
            # ffmpeg that ignores terminate keeps the reader blocked, and the
            # hang guard has already ended with the local stop
            process = self._process

            if process is not None:
                process.kill()

            self.join(timeout=STOP_GRACE_S)

        if self.is_alive():
            logger.warning("Replay thread for %s did not stop in time", self.name)
            return False

        return True

    def _stopped(self) -> bool:
        return self.stop_event.is_set() or self._local_stop.is_set()

    def _pending(self) -> bool:
        with self._lock:
            return self._demand or self._cursor < self._replay_until

    def _detect_wanted(self, timestamp: float) -> bool:
        with self._lock:
            return self._windows.contains(timestamp)

    def run(self) -> None:
        """Replay pending footage whenever detection is wanted, until stopped.

        One hang guard thread lives as long as the runner. A failed replay
        pass is logged and the loop carries on.
        """
        # one guard for the life of the runner, not one per segment
        threading.Thread(
            target=self._guard,
            args=(self._local_stop,),
            name=f"replay_guard:{self.config.name}",
            daemon=True,
        ).start()

        try:
            while not self._stopped():
                self._wake.wait(1.0)
                self._wake.clear()

                self._zero_idle_rates()

                if self._stopped() or not self._pending():
                    continue

                try:
                    self._replay_pending()
                except Exception:
                    logger.exception("Replay failed for %s", self.config.name)
        finally:
            # also ends the guard when the global stop event was what stopped us
            self._local_stop.set()

    def _zero_idle_rates(self) -> None:
        """Zero the tracker rates once replay has been quiet for a while.

        Live capture keeps the tracker fed or marks the camera offline, replay
        goes quiet between events, and the rates would keep the values of the
        last event for good. Runs on the replay thread only.
        """
        if not self._rates_stale or self.active:
            return

        if time.time() - self._last_fed_at < IDLE_STATS_DELAY_S:
            return

        for rate in self._tracker_rates:
            rate.value = 0

        self._rates_stale = False

    def _replay_pending(self) -> None:
        """Replay every finished segment that overlaps the wanted range.

        The cursor only moves over footage that was fed to the tracker, or that
        was given up on after SEGMENT_MAX_ATTEMPTS failed attempts. A segment
        that fails holds back the ones after it, because the tracker needs
        frames in order, and is tried again after a growing pause.
        """
        while not self._stopped():
            now = time.time()

            if now < self._retry_not_before:
                return

            segments = self.list_segments(self.config.name, self.cache_dir)
            # footage in a segment that could not be probed yet is not skipped
            blocked_from = getattr(segments, "blocked_from", None)
            listed = {segment.path for segment in segments}
            self._failures = {
                path: count for path, count in self._failures.items() if path in listed
            }

            with self._lock:
                until = self._replay_until
                # footage before the cursor was already consumed
                start_from = max(self._range_start, self._cursor)
                self._windows.prune(now - WINDOW_RETENTION_S)

            # every segment carries its own span, so gaps between segments
            # neither stretch footage nor hide it. A segment is wanted while
            # it holds footage between where replay stands and where it ends
            next_segment = next(
                (
                    segment
                    for segment in segments
                    if segment.start < until
                    and min(segment.end, until) > start_from
                    and (blocked_from is None or segment.start < blocked_from)
                ),
                None,
            )

            if next_segment is None:
                # footage that may still become readable is waited for, not
                # given up on
                if blocked_from is None or blocked_from >= until:
                    self._finish_if_caught_up(now, until)

                return

            try:
                covered, failed = self._replay_segment(next_segment)
            except Exception:
                logger.exception("Replay of %s failed", next_segment.path)
                covered, failed = 0.0, True

            with self._lock:
                self._cursor = max(self._cursor, covered)

            if not failed or self._stopped():
                self._failures.pop(next_segment.path, None)
                continue

            if not self._note_failure(next_segment, now):
                # later segments wait, and this one is tried again after a pause
                return

    def _note_failure(self, segment: CacheSegment, now: float) -> bool:
        """Count a failed attempt, returns whether the segment was given up on.

        Giving up is explicit: it is logged as an error with the span that
        detection will not see, and is the only way the cursor moves past
        footage that was not fed to the tracker.
        """
        attempts = self._failures.get(segment.path, 0) + 1

        if attempts >= SEGMENT_MAX_ATTEMPTS:
            logger.error(
                "Giving up on replay of %s after %s failed attempts, detection "
                "will not see its footage from %.1f to %.1f",
                segment.path,
                attempts,
                segment.start,
                segment.end,
            )

            with self._lock:
                self._cursor = max(self._cursor, segment.end)

            self._failures.pop(segment.path, None)
            return True

        self._failures[segment.path] = attempts
        self._retry_not_before = now + min(
            SEGMENT_RETRY_BACKOFF_S * 2 ** (attempts - 1), SEGMENT_RETRY_MAX_BACKOFF_S
        )
        return False

    def _finish_if_caught_up(self, now: float, until: float) -> None:
        """Stop waiting for footage that is never going to be finished."""
        with self._lock:
            if self._demand or until == float("inf"):
                return

            give_up_at = until + GIVE_UP_SEGMENTS * self.segment_time + GIVE_UP_MARGIN_S

            if now > give_up_at:
                self._cursor = max(self._cursor, until)

    def _start_process(self, cmd: list[str]) -> sp.Popen:
        return sp.Popen(
            cmd,
            stdin=sp.DEVNULL,
            stdout=sp.PIPE,
            stderr=sp.PIPE,
            # frames are read straight into self._frame_buffer, so there is no
            # need for a frame sized buffer in the pipe reader as well
            start_new_session=True,
        )

    def _guard(self, done: threading.Event) -> None:
        """Kill ffmpeg if it stops producing frames without exiting."""
        while not done.wait(1.0):
            started = self._read_started
            process = self._process

            if (
                process is not None
                and started is not None
                and time.time() - started > READ_HANG_TIMEOUT_S
            ):
                logger.warning(
                    "Replay ffmpeg for %s produced no frame in %s seconds, killing it",
                    self.config.name,
                    READ_HANG_TIMEOUT_S,
                )
                process.kill()
                # the guard outlives this segment, so it must not fire again
                # for the same stalled read
                self._read_started = None

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

    def _replay_segment(self, segment: CacheSegment) -> tuple[float, bool]:
        """Replay one segment.

        Returns the capture time the segment was consumed up to, which is zero
        when nothing of it was, and whether the replay failed.
        """
        use_software = time.time() < self._software_until
        cmd = self.config.get_replay_ffmpeg_cmd(segment.path, software=use_software)

        if cmd is None:
            # the config validator and the runtime updater both refuse replay
            # without a record input, so this is a state that should not occur
            if not self._warned_no_source:
                logger.warning(
                    "Detect mode replay for %s has no input with the record role to decode",
                    self.config.name,
                )
                self._warned_no_source = True

            return segment.end, False

        frames, failed, covered = self._decode_segment(segment, cmd)

        if (
            failed
            and not use_software
            and not self._frame_unsent
            and not self._stopped()
        ):
            # a hardware decoder that cannot read a segment, or dies part way
            # through, should not cost the footage, so it is tried once more on
            # the CPU. Frames already fed are skipped by their capture time, so
            # the retry only adds what is missing. A frame that could not be
            # written is not the decoder's fault and is not retried here
            software_cmd = self.config.get_replay_ffmpeg_cmd(
                segment.path, software=True
            )

            if software_cmd is not None and software_cmd != cmd:
                logger.warning(
                    "Hardware decode of %s failed, retrying it in software",
                    segment.path,
                )
                software_frames, software_failed, software_covered = (
                    self._decode_segment(segment, software_cmd)
                )
                failed = software_failed
                # a retry that adds nothing new must not lose what was consumed
                covered = max(covered, software_covered)

                if software_frames > 0 and not software_failed:
                    # the decoder is the problem, not the footage, so the next
                    # segments skip the hardware attempt for a while
                    self._software_until = time.time() + HARDWARE_RETRY_AFTER_S
                    logger.warning(
                        "Hardware decode keeps failing for %s, decoding in "
                        "software for the next %s seconds",
                        self.config.name,
                        HARDWARE_RETRY_AFTER_S,
                    )

        return covered, failed

    def _decode_segment(
        self, segment: CacheSegment, cmd: list[str]
    ) -> tuple[int, bool, float]:
        """Decode a segment with one ffmpeg command.

        Returns how many frames were handed to the tracker, whether ffmpeg
        failed, and the capture time the segment was consumed up to. That is
        the end of the segment only when all of it was read, otherwise the
        capture time of the last frame handed over, or zero if there was none.
        """
        logger.debug("Replaying %s from %s", self.config.name, segment.path)

        try:
            process = self._start_process(cmd)
        except OSError:
            logger.exception("Unable to start ffmpeg for %s", segment.path)
            return 0, True, 0.0

        reader: FfmpegStderr | None = None
        self._process = process
        self.active = True
        started = time.time()
        frames = 0
        desynced = False
        ended = False
        truncated = False
        trimmed = False
        out_of_span = 0
        unsent = False
        # nothing is consumed until a frame is handed over or the segment is
        # read to its end
        covered = 0.0

        # everything after the process exists is inside the try, so a failure
        # while setting up the helper threads cannot leave ffmpeg running
        try:
            reader = FfmpegStderr(process.stderr)
            reader.start()

            frame_view = memoryview(self._frame_buffer)

            while not self._stopped():
                self._read_started = time.time()
                read = process.stdout.readinto(frame_view) or 0
                self._read_started = None

                # a short read is the end of the segment, or a broken one
                if read < self.frame_size:
                    ended = True
                    truncated = read > 0
                    break

                pts = self._next_time(reader)

                if pts is None:
                    # Times are paired with frames by order, so once one frame
                    # has no time, a time that arrives late would be given to
                    # the next frame and every later frame would be stamped
                    # wrongly. Stop here, the rest is not timed reliably
                    desynced = True
                    break

                timestamp = frame_timestamp(segment, pts)

                if not (
                    segment.start - SEGMENT_TIME_SLACK_S
                    <= timestamp
                    <= segment.end + SEGMENT_TIME_SLACK_S
                ):
                    # a time that cannot belong to this footage. Checked before
                    # the end of the range, which it would otherwise look like
                    out_of_span += 1
                    continue

                with self._lock:
                    until = self._replay_until

                if timestamp > until:
                    # the range ends inside this segment, the rest of it stays
                    # unconsumed in case a later trigger wants it
                    trimmed = True
                    covered = until
                    break

                if timestamp <= self._last_fed:
                    # already handed to the tracker by an earlier replay
                    continue

                if self._feed(frame_view, timestamp):
                    frames += 1
                    self._last_fed = timestamp
                    covered = timestamp
                elif not self._stopped():
                    # the tracker never got this frame, so nothing after it
                    # counts as consumed
                    unsent = True
                    break

                now = time.time()
                self._last_fed_at = now
                self._rates_stale = True
                elapsed = now - started
                if elapsed > 0:
                    self.camera_fps.value = frames / elapsed
        finally:
            self._read_started = None
            return_code = self._reap(process, ended)

            if reader is not None and reader.is_alive():
                reader.join(timeout=2)

            # the pipe is only closed once nothing is still reading it
            if (reader is None or not reader.is_alive()) and process.stderr is not None:
                process.stderr.close()

            self._process = None
            self.active = False
            self.camera_fps.value = 0

        self._frame_unsent = unsent

        if self._stopped():
            return frames, False, covered

        if desynced:
            logger.warning(
                "Replay of %s stopped after %s frames, a frame had no capture "
                "time and the rest of the segment cannot be timed reliably",
                segment.path,
                frames,
            )

        if out_of_span:
            # excluded by rule, not a failure: ffmpeg would give the same times
            # again, so retrying cannot recover these frames
            logger.warning(
                "Replay of %s skipped %s frames with capture times outside the "
                "segment, they cannot be placed in the footage",
                segment.path,
                out_of_span,
            )

        # ffmpeg that was stopped on purpose at the end of the range exits
        # with a signal, that is not a failure
        failed = desynced or unsent or (not trimmed and (return_code != 0 or truncated))

        if not failed and not trimmed:
            # ffmpeg read the whole segment, including frames that an earlier
            # replay already handed over
            covered = segment.end

        # every kept frame logs one time, so a time left over after a clean
        # finish means frames and times were paired wrongly
        if (
            not failed
            and not trimmed
            and reader is not None
            and not reader.is_alive()
            and not reader.times.empty()
        ):
            logger.warning(
                "Replay of %s left %s capture times unmatched, the frame "
                "timestamps of that segment may be off",
                segment.path,
                reader.times.qsize(),
            )

        if failed:
            logger.warning(
                "Replay of %s ended early (exit code %s, %s frames): %s",
                segment.path,
                return_code,
                frames,
                "; ".join(reader.errors) or "no error output",
            )
        elif frames == 0:
            logger.debug("No frames replayed from %s", segment.path)

        return frames, failed, covered

    def _frame_times(self) -> Any:
        """The camera's frame time ledger, None when its segment is missing."""
        frame_times = self.frame_manager.frame_times(self.config.name)

        if frame_times is None and not self._warned_no_ledger:
            self._warned_no_ledger = True
            logger.warning(
                "No frame time ledger for %s, readers will skip replayed frames",
                self.config.name,
            )

        return frame_times

    def _feed(self, data: bytes | bytearray | memoryview, timestamp: float) -> bool:
        """Hand one frame to the tracker, returns whether it was sent.

        The frame is copied into shared memory, so the caller can reuse data.
        A stopped runner touches nothing: the ring may be owned by another by now.
        """
        if self._stopped():
            return False

        frame_name = f"{self.config.name}_frame{self._frame_index}"
        frame_buffer = self.frame_manager.write(frame_name)

        if frame_buffer is None:
            logger.warning("No shared memory for frame %s", frame_name)
            return False

        # Readers only use a slot whose entry in the camera's frame time ledger
        # matches the frame time they were told about, as in live capture. The
        # entry is cleared while the slot is rewritten and set once it is whole.
        frame_times = self._frame_times()

        if frame_times is not None:
            frame_times[self._frame_index] = 0

        # only the writer's mapping is closed, the tracker still owns the buffer
        sent = False

        try:
            frame_buffer[:] = data

            if frame_times is not None:
                frame_times[self._frame_index] = timestamp

            # unlike live capture, replay waits for the tracker so no frame is dropped
            item = (frame_name, timestamp, self._detect_wanted(timestamp))
            while not self._stopped():
                try:
                    self.frame_queue.put(item, True, 0.5)
                    sent = True
                    break
                except queue.Full:
                    continue
        finally:
            self.frame_manager.close(frame_name)

        if not sent:
            # stopped before the tracker was told about the frame. The slot is
            # a ring buffer owned elsewhere, so it is not freed, but its
            # ledger entry must not vouch for a frame nobody will read
            if frame_times is not None:
                frame_times[self._frame_index] = 0

            return False

        self._frame_index = (self._frame_index + 1) % self.shm_frame_count
        return True

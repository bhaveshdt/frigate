"""Tests for replaying recorded segments through detection."""

import io
import os
import queue
import subprocess as sp
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import psutil

from frigate.config import CameraConfig, FrigateConfig
from frigate.util import cache_frame
from frigate.util.cache_frame import (
    CacheSegment,
    get_cache_files_in_use,
    list_finished_cache_segments,
    probe_segment,
)
from frigate.util.segment_time import measure_segment_start
from frigate.video.detect_replay import (
    DetectReplayRunner,
    DetectWindows,
    FfmpegStderr,
    frame_timestamp,
)

# capture times inside a segment are uneven because the cache stamps frames
# with their arrival time
UNEVEN_TIMES = [0.0, 0.3, 0.5]
LOGGER = "frigate.video.detect_replay"
JPEG = cv2.imencode(".jpg", np.zeros((8, 8, 3), np.uint8))[1].tobytes()


def segment(path: str, start: float, duration: float = 10.0, **kwargs):
    return CacheSegment(path, start, start + duration, **kwargs)


def showinfo(pts_time: str, n: int = 0) -> bytes:
    return (
        f"[Parsed_showinfo_1 @ 0x55d0] [info] n:{n:4d} pts:{n * 1000:7d} "
        f"pts_time:{pts_time} duration: 1024\n"
    ).encode()


class TestFrameTimestamp(unittest.TestCase):
    def test_adds_the_presentation_time_to_the_segment_start(self):
        assert frame_timestamp(segment("/c/a.mp4", 1000.0), 0.0) == 1000.0
        assert frame_timestamp(segment("/c/a.mp4", 1000.0), 7.25) == 1007.25

    def test_removes_the_first_presentation_time_of_the_file(self):
        seg = segment("/c/a.mp4", 1000.0, first_pts=0.5)

        assert frame_timestamp(seg, 0.5) == 1000.0
        assert frame_timestamp(seg, 1.5) == 1001.0

    def test_is_not_the_time_of_replay(self):
        with patch("time.time", return_value=5000.0):
            assert frame_timestamp(segment("/c/a.mp4", 1000.0), 0.2) == 1000.2


class TestDetectWindows(unittest.TestCase):
    def test_contains_only_the_open_span(self):
        windows = DetectWindows()
        windows.open(100)
        windows.close(120)

        assert not windows.contains(99.9)
        assert windows.contains(100)
        assert windows.contains(120)
        assert not windows.contains(120.1)

    def test_open_span_has_no_end(self):
        windows = DetectWindows()
        windows.open(100)

        assert windows.contains(10**9)

    def test_retrigger_before_the_span_ended_extends_it(self):
        windows = DetectWindows()
        windows.open(100)
        windows.close(120)
        windows.open(115)

        assert windows.contains(500)

    def test_separate_spans_stay_separate(self):
        windows = DetectWindows()
        windows.open(100)
        windows.close(120)
        windows.open(200)
        windows.close(210)

        assert windows.contains(105)
        assert not windows.contains(150)
        assert windows.contains(205)

    def test_prune_drops_old_spans(self):
        windows = DetectWindows()
        windows.open(100)
        windows.close(120)
        windows.prune(1000)

        assert not windows.contains(110)


class TestMeasureSegmentStart(unittest.TestCase):
    def test_uses_the_mtime_minus_the_duration_inside_the_truncated_second(self):
        self.assertAlmostEqual(measure_segment_start(1000.0, 1010.4, 10.0), 1000.4)

    def test_cannot_measure_outside_the_truncation_window(self):
        # media shorter than the wall span, such as a stalled stream
        assert measure_segment_start(1000.0, 1010.0, 4.0) is None
        # an mtime that is earlier than the name allows
        assert measure_segment_start(1000.0, 1009.0, 10.0) is None

    def test_cannot_measure_without_an_mtime(self):
        assert measure_segment_start(1000.0, None, 10.0) is None


class TestProbeSegment(unittest.TestCase):
    def setUp(self):
        probe_segment.cache_clear()

    @staticmethod
    def _run(stdout: str = "", returncode: int = 0):
        return MagicMock(stdout=stdout, returncode=returncode)

    def test_reads_the_first_pts_and_duration(self):
        out = '{"format": {"start_time": "0.000000", "duration": "9.983000"}}'

        with patch("subprocess.run", return_value=self._run(out)):
            assert probe_segment("ffprobe", "/c/a.mp4", 1.0) == (0.0, 9.983)

    def test_unreadable_files_return_none(self):
        with patch("subprocess.run", return_value=self._run("", returncode=1)):
            assert probe_segment("ffprobe", "/c/a.mp4", 1.0) is None

        with patch("subprocess.run", return_value=self._run("{}")):
            assert probe_segment("ffprobe", "/c/b.mp4", 1.0) is None

        zero = '{"format": {"start_time": "0", "duration": "0"}}'
        with patch("subprocess.run", return_value=self._run(zero)):
            assert probe_segment("ffprobe", "/c/c.mp4", 1.0) is None

    def test_a_timeout_or_missing_ffprobe_returns_none(self):
        with patch("subprocess.run", side_effect=sp.TimeoutExpired("ffprobe", 10)):
            assert probe_segment("ffprobe", "/c/a.mp4", 1.0) is None

        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert probe_segment("ffprobe", "/c/b.mp4", 1.0) is None


class TestListFinishedCacheSegments(unittest.TestCase):
    # 2026-01-01T00:00:00Z
    T0 = 1767225600.0

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _touch(self, name: str, mtime: float | None = None) -> str:
        path = os.path.join(self.tmp.name, name)
        open(path, "w").close()

        if mtime is not None:
            os.utime(path, (mtime, mtime))

        return path

    def _list(self, probed=(0.0, 9.98), open_paths=()) -> list[CacheSegment]:
        with patch(
            "frigate.util.cache_frame.probe_segment", return_value=probed
        ) as probe:
            result = list_finished_cache_segments(
                "front_door",
                self.tmp.name,
                in_use=lambda _: set(open_paths),
            )

        self.probe = probe
        return result

    def test_leaves_out_the_segment_ffmpeg_has_open(self):
        first = self._touch("front_door@20260101000000+0000.mp4")
        second = self._touch("front_door@20260101000010+0000.mp4")
        writing = self._touch("front_door@20260101000020+0000.mp4")

        assert [s.path for s in self._list(open_paths={writing})] == [first, second]

    def test_the_newest_file_is_listed_once_ffmpeg_has_closed_it(self):
        # a disabled or reconnecting camera leaves its last file closed
        first = self._touch("front_door@20260101000000+0000.mp4")
        last = self._touch("front_door@20260101000010+0000.mp4")

        assert [s.path for s in self._list()] == [first, last]

    def test_start_and_end_come_from_the_segment_itself(self):
        self._touch("front_door@20260101000000+0000.mp4", self.T0 + 10.38)

        (seg,) = self._list()

        self.assertAlmostEqual(seg.start, self.T0 + 10.38 - 9.98)
        self.assertAlmostEqual(seg.end, seg.start + 9.98)
        assert seg.first_pts == 0.0

    def test_a_missing_neighbor_does_not_change_the_span(self):
        # the segment at :10 was never written, so :20 follows :00 with a gap
        self._touch("front_door@20260101000000+0000.mp4", self.T0 + 9.98)
        self._touch("front_door@20260101000020+0000.mp4", self.T0 + 29.98)
        writing = self._touch("front_door@20260101000030+0000.mp4")

        first, second = self._list(open_paths={writing})

        assert (first.start, first.end) == (self.T0, self.T0 + 9.98)
        assert (second.start, second.end) == (self.T0 + 20.0, self.T0 + 29.98)

    def test_falls_back_to_the_name_when_the_mtime_does_not_fit(self):
        self._touch("front_door@20260101000000+0000.mp4", self.T0 + 50)

        (seg,) = self._list()

        assert seg.start == self.T0

    def test_skips_segments_that_cannot_be_probed(self):
        self._touch("front_door@20260101000000+0000.mp4")
        self._touch("front_door@20260101000010+0000.mp4")

        assert self._list(probed=None) == []

    def test_ignores_sub_streams_other_cameras_and_junk(self):
        self._touch("front_door@sub@20260101000000+0000.mp4")
        self._touch("front_door@sub@20260101000010+0000.mp4")
        self._touch("front_door_2@20260101000000+0000.mp4")
        self._touch("front_door_2@20260101000010+0000.mp4")
        self._touch("front_door@notadate.mp4")

        assert self._list() == []
        self.probe.assert_not_called()

    def test_an_empty_cache_has_no_segments(self):
        assert self._list() == []

    def test_a_file_that_disappears_during_the_listing_is_skipped(self):
        # the recording maintainer moved it after it was globbed
        first = self._touch("front_door@20260101000000+0000.mp4")
        gone = self._touch("front_door@20260101000010+0000.mp4")
        writing = self._touch("front_door@20260101000020+0000.mp4")
        real_getmtime = os.path.getmtime

        def getmtime(path):
            if path == gone:
                raise FileNotFoundError(path)

            return real_getmtime(path)

        with patch("os.path.getmtime", side_effect=getmtime):
            segments = self._list(open_paths={writing})

        assert [s.path for s in segments] == [first]


class TestCacheFilesInUse(unittest.TestCase):
    @staticmethod
    def _process(name: str, paths=(), error: Exception | None = None):
        process = MagicMock()
        process.name.return_value = name

        if error is not None:
            process.open_files.side_effect = error
        else:
            process.open_files.return_value = [MagicMock(path=p) for p in paths]

        return process

    def _in_use(self, processes) -> set[str]:
        with patch(
            "frigate.util.cache_frame.psutil.process_iter", return_value=processes
        ):
            return get_cache_files_in_use("/tmp/cache")

    def test_collects_the_cache_files_ffmpeg_has_open(self):
        processes = [
            self._process("ffmpeg", ["/tmp/cache/a.mp4", "/dev/null"]),
            self._process("ffmpeg", ["/tmp/cache/b.mp4"]),
        ]

        assert self._in_use(processes) == {"/tmp/cache/a.mp4", "/tmp/cache/b.mp4"}

    def test_only_ffmpeg_processes_count(self):
        processes = [self._process("python3", ["/tmp/cache/a.mp4"])]

        assert self._in_use(processes) == set()

    def test_a_process_that_exits_mid_scan_is_skipped(self):
        processes = [
            self._process("ffmpeg", error=psutil.NoSuchProcess(1)),
            self._process("ffmpeg", error=psutil.AccessDenied(2)),
            self._process("ffmpeg", ["/tmp/cache/b.mp4"]),
        ]

        assert self._in_use(processes) == {"/tmp/cache/b.mp4"}


class TestLatestCacheFrame(unittest.TestCase):
    def setUp(self):
        cache_frame._latest_frame_cache.clear()

    def _get(self, path: str = "/c/a.mp4"):
        with (
            patch.object(
                cache_frame, "get_latest_finished_cache_segment", return_value=path
            ),
            patch.object(
                cache_frame, "run_ffmpeg_snapshot", return_value=(JPEG, "")
            ) as snapshot,
        ):
            frame = cache_frame.get_latest_cache_frame("ffmpeg", "front_door")

        return frame, snapshot

    def test_a_finished_segment_is_decoded_once(self):
        first, snapshot = self._get()
        second, again = self._get()

        assert first is not None
        snapshot.assert_called_once()
        again.assert_not_called()
        assert (first == second).all()

    def test_a_newer_segment_is_decoded_again(self):
        self._get("/c/a.mp4")
        _, snapshot = self._get("/c/b.mp4")

        snapshot.assert_called_once()

    def test_callers_cannot_change_the_cached_frame(self):
        first, _ = self._get()
        first[:] = 255
        second, _ = self._get()

        assert second.max() == 0


class TestFfmpegStderr(unittest.TestCase):
    def test_splits_frame_times_from_everything_else(self):
        lines = b"".join(
            [
                b"libva info: VA-API version 1.22.0\n",
                b"[Parsed_showinfo_1 @ 0x55d0] [info] config in time_base: 1/15360\n",
                b"[info] Stream mapping:\n",
                showinfo("0", 0),
                showinfo("0.401562", 1),
                b"\n",
                b"[h264_qsv @ 0x1] [warning] Dropped a frame\n",
                b"[mov,mp4 @ 0x2] [error] moov atom not found\n",
                showinfo("nan", 2),
                b"unprefixed message\n",
            ]
        )
        reader = FfmpegStderr(io.BytesIO(lines))
        reader.start()
        reader.join(timeout=2)

        times = []
        while not reader.times.empty():
            times.append(reader.times.get())

        assert times[:2] == [0.0, 0.401562]
        assert len(times) == 3
        assert list(reader.errors) == [
            "[h264_qsv @ 0x1] [warning] Dropped a frame",
            "[mov,mp4 @ 0x2] [error] moov atom not found",
            "unprefixed message",
        ]

    def test_keeps_only_the_latest_error_lines(self):
        lines = b"".join(f"[error] problem {i}\n".encode() for i in range(100))
        reader = FfmpegStderr(io.BytesIO(lines))
        reader.start()
        reader.join(timeout=2)

        assert len(reader.errors) == 20
        assert reader.errors[-1] == "[error] problem 99"

    def test_frame_times_never_block_the_reader(self):
        # a blocked reader would stop draining stderr and hang ffmpeg, so the
        # time queue must accept any number of entries
        lines = b"".join(showinfo(str(i), i) for i in range(5000))
        reader = FfmpegStderr(io.BytesIO(lines))
        reader.start()
        reader.join(timeout=5)

        assert not reader.is_alive()
        assert reader.times.maxsize == 0
        assert reader.times.qsize() == 5000


def build_runner(segments: list[CacheSegment], detect: dict | None = None):
    config = FrigateConfig(
        **{
            "mqtt": {"host": "mqtt"},
            "cameras": {
                "front_door": {
                    "ffmpeg": {
                        "inputs": [
                            {
                                "path": "rtsp://10.0.0.1:554/video",
                                "roles": ["record", "detect"],
                            }
                        ]
                    },
                    "detect": {
                        "mode": "replay",
                        "width": 64,
                        "height": 48,
                        "fps": 5,
                        **(detect or {}),
                    },
                    "record": {"enabled": True},
                }
            },
        }
    )
    camera = config.cameras["front_door"]
    frame_size = 48 * 3 // 2 * 64

    frame_queue: queue.Queue = queue.Queue()
    frame_manager = MagicMock()
    frame_manager.write.side_effect = lambda name: bytearray(frame_size)

    runner = DetectReplayRunner(
        camera,
        10,
        frame_queue,
        MagicMock(),
        threading.Event(),
        frame_manager=frame_manager,
        list_segments=lambda *_: list(segments),
    )
    return runner, frame_queue, frame_size


def fake_process(
    frame_size: int,
    times: list[str] | None = None,
    return_code: int = 0,
    trailing: bytes = b"",
    errors: tuple[bytes, ...] = (),
) -> MagicMock:
    times = [str(t) for t in UNEVEN_TIMES] if times is None else times

    process = MagicMock()
    process.stdout = io.BytesIO(b"\x00" * frame_size * len(times) + trailing)
    process.stderr = io.BytesIO(
        b"".join([showinfo(t, i) for i, t in enumerate(times)] + list(errors))
    )
    process.wait.return_value = return_code
    process.poll.return_value = return_code
    process.returncode = return_code
    return process


class TestDetectReplayRunner(unittest.TestCase):
    def _replay(self, runner, frame_size, **process_kwargs):
        # the fake footage is dated near 1000, so pin the clock near it as well
        with (
            patch.object(
                runner,
                "_start_process",
                side_effect=lambda *_: fake_process(frame_size, **process_kwargs),
            ) as start,
            patch("frigate.video.detect_replay.time") as fake_time,
        ):
            fake_time.time.return_value = 1100.0
            runner._replay_pending()

        return start

    @staticmethod
    def _drain(frame_queue) -> list[tuple]:
        items = []
        while not frame_queue.empty():
            items.append(frame_queue.get())
        return items

    def test_replays_the_segment_before_the_trigger_with_original_timestamps(self):
        segments = [segment("/c/a.mp4", 1000.0), segment("/c/b.mp4", 1010.0)]
        runner, frame_queue, frame_size = build_runner(segments)

        runner.trigger(1015.0)
        self._replay(runner, frame_size)
        items = self._drain(frame_queue)

        assert [i[1] for i in items] == [
            1000.0,
            1000.3,
            1000.5,
            1010.0,
            1010.3,
            1010.5,
        ]
        # none of these were captured at the time of replay
        assert all(i[1] < 1100 for i in items)

    def test_timestamps_follow_the_capture_times_not_an_even_spacing(self):
        runner, frame_queue, frame_size = build_runner([segment("/c/a.mp4", 1000.0)])

        runner.trigger(1008.0)
        self._replay(runner, frame_size, times=["0", "0.401562", "0.577409"])
        times = [i[1] for i in self._drain(frame_queue)]

        assert times == [1000.0, 1000.401562, 1000.577409]
        # the old index over fps model would have said 1000.0, 1000.2, 1000.4
        assert times[1] != 1000.2

    def test_first_pts_of_the_file_is_removed(self):
        runner, frame_queue, frame_size = build_runner(
            [segment("/c/a.mp4", 1000.0, first_pts=2.0)]
        )

        runner.trigger(1008.0)
        self._replay(runner, frame_size, times=["2.0", "2.5"])

        assert [i[1] for i in self._drain(frame_queue)] == [1000.0, 1000.5]

    def test_frames_before_the_pre_roll_are_not_detection_frames(self):
        segments = [segment("/c/a.mp4", 1000.0), segment("/c/b.mp4", 1010.0)]
        runner, frame_queue, frame_size = build_runner(segments)

        runner.trigger(1015.0)
        self._replay(runner, frame_size)
        items = self._drain(frame_queue)

        # pre roll is a segment, so detection is wanted from 1005
        assert [i[2] for i in items] == [False] * 3 + [True] * 3

    def test_detection_flag_follows_capture_time_not_replay_time(self):
        segments = [segment("/c/a.mp4", 1000.0), segment("/c/b.mp4", 1010.0)]
        runner, frame_queue, frame_size = build_runner(segments)

        # the event was over before any of it was replayed
        runner.trigger(1004.0)
        runner.release(1011.0)
        self._replay(runner, frame_size)
        items = self._drain(frame_queue)

        flags = {round(i[1], 1): i[2] for i in items}
        assert flags[1000.5] is True
        assert flags[1010.0] is True
        assert flags[1010.5] is True

    def test_tail_replays_footage_after_the_release_with_detection_off(self):
        segments = [
            segment("/c/a.mp4", 1000.0),
            segment("/c/b.mp4", 1010.0),
            segment("/c/c.mp4", 1020.0),
        ]
        runner, frame_queue, frame_size = build_runner(segments)

        runner.trigger(1005.0)
        runner.release(1012.0)
        self._replay(runner, frame_size)
        items = self._drain(frame_queue)

        by_time = {round(i[1], 1): i[2] for i in items}
        assert by_time[1010.5] is True
        assert by_time[1020.0] is False
        # motion off needs frames mqtt_off_delay after the last motion
        assert runner.tail > 30

    def test_each_segment_is_replayed_once(self):
        segments = [segment("/c/a.mp4", 1000.0), segment("/c/b.mp4", 1010.0)]
        runner, frame_queue, frame_size = build_runner(segments)

        runner.trigger(1015.0)
        first = self._replay(runner, frame_size)
        assert first.call_count == 2

        # a second trigger while the footage is still in cache
        runner.release(1016.0)
        runner.trigger(1017.0)
        second = self._replay(runner, frame_size)

        assert second.call_count == 0

    def test_a_trigger_while_replaying_extends_the_same_session(self):
        segments = [segment("/c/a.mp4", 1000.0), segment("/c/b.mp4", 1010.0)]
        runner, frame_queue, frame_size = build_runner(segments)
        runner.trigger(1015.0)
        starts: list[int] = []

        def start(*_):
            starts.append(1)

            if len(starts) == 1:
                # a second hardware event lands while the first segment decodes
                runner.trigger(1016.0)

            return fake_process(frame_size)

        with (
            patch.object(runner, "_start_process", side_effect=start),
            patch("frigate.video.detect_replay.time") as fake_time,
        ):
            fake_time.time.return_value = 1100.0
            runner._replay_pending()

        times = [i[1] for i in self._drain(frame_queue)]

        # one ffmpeg process per segment, and no frame is replayed twice
        assert len(starts) == 2
        assert times == [1000.0, 1000.3, 1000.5, 1010.0, 1010.3, 1010.5]
        # still one open span that began a pre roll before the first event
        assert runner._replay_until == float("inf")
        assert runner._range_start == 1005.0
        assert runner._windows.contains(1005.0)
        assert runner._windows.contains(1500.0)

    def test_a_trigger_during_the_tail_of_a_release_continues_the_session(self):
        segments = [segment("/c/a.mp4", 1000.0), segment("/c/b.mp4", 1010.0)]
        runner, frame_queue, frame_size = build_runner(segments)

        runner.trigger(1015.0)
        runner.release(1016.0)
        # nothing was replayed yet, so this is the same session, not a new one
        runner.trigger(1020.0)

        assert runner._range_start == 1005.0
        assert runner._replay_until == float("inf")
        assert runner._windows.contains(1030.0)

        start = self._replay(runner, frame_size)
        items = self._drain(frame_queue)

        assert start.call_count == 2
        assert [i[2] for i in items] == [False] * 3 + [True] * 3

    def test_repeated_triggers_do_not_start_a_second_ffmpeg_process(self):
        runner, frame_queue, frame_size = build_runner([segment("/c/a.mp4", 1000.0)])

        for event_time in (1005.0, 1006.0, 1007.0):
            runner.trigger(event_time)

        start = self._replay(runner, frame_size)

        assert start.call_count == 1
        assert len(self._drain(frame_queue)) == 3

    def test_replay_stops_at_the_end_of_the_range_and_that_is_not_a_failure(self):
        segments = [segment("/c/a.mp4", 1000.0), segment("/c/b.mp4", 1010.0)]
        runner, frame_queue, frame_size = build_runner(segments)
        runner.tail = 4.4

        runner.trigger(1005.0)
        runner.release(1006.0)
        calls: list[list[str]] = []

        def start(cmd):
            calls.append(cmd)
            # the first segment ends by itself, the second is stopped on
            # purpose at the end of the range, which ends ffmpeg with a signal
            return fake_process(frame_size, return_code=0 if len(calls) == 1 else -15)

        with (
            patch.object(runner, "_start_process", side_effect=start),
            patch("frigate.video.detect_replay.time") as fake_time,
            self.assertNoLogs(LOGGER, level="WARNING"),
        ):
            fake_time.time.return_value = 1100.0
            runner._replay_pending()

        times = [i[1] for i in self._drain(frame_queue)]

        # the range ends at 1010.4, so the frame at 1010.5 is never fed
        assert times == [1000.0, 1000.3, 1000.5, 1010.0, 1010.3]
        self.assertAlmostEqual(runner._cursor, 1010.4)
        assert runner._pending() is False

    def test_a_trigger_right_after_a_trimmed_replay_gets_its_pre_roll(self):
        segments = [segment("/c/a.mp4", 1000.0)]
        runner, frame_queue, frame_size = build_runner(segments)
        runner.tail = 4.0
        times = ["0", "5", "9"]

        runner.trigger(1001.0)
        runner.release(1002.0)
        self._replay(runner, frame_size, times=times, return_code=-15)
        first = [i[1] for i in self._drain(frame_queue)]

        # a new event arrives after the first range ended
        runner.trigger(1015.0)
        self._replay(runner, frame_size, times=times)
        second = [(i[1], i[2]) for i in self._drain(frame_queue)]

        assert first == [1000.0, 1005.0]
        # the frame the first range never reached is replayed as detection
        # footage, and nothing already replayed is fed twice
        assert second == [(1009.0, True)]

    def test_a_hardware_decode_failure_is_retried_on_the_cpu(self):
        runner, frame_queue, frame_size = build_runner([segment("/c/a.mp4", 1000.0)])
        runner.trigger(1008.0)
        commands: list[list[str]] = []

        def start(cmd):
            commands.append(cmd)

            if len(commands) == 1:
                return fake_process(frame_size, times=[], return_code=1)

            return fake_process(frame_size)

        def build(path, software=False):
            return ["ffmpeg-cpu" if software else "ffmpeg-gpu", path]

        with (
            patch.object(CameraConfig, "get_replay_ffmpeg_cmd", side_effect=build),
            patch.object(runner, "_start_process", side_effect=start),
            self.assertLogs(LOGGER, level="WARNING") as logs,
        ):
            runner._replay_pending()

        assert [c[0] for c in commands] == ["ffmpeg-gpu", "ffmpeg-cpu"]
        assert len(self._drain(frame_queue)) == 3
        assert "retrying it in software" in "\n".join(logs.output)

    def test_nothing_is_retried_once_frames_were_handed_over(self):
        runner, frame_queue, frame_size = build_runner([segment("/c/a.mp4", 1000.0)])
        runner.trigger(1008.0)

        def build(path, software=False):
            return ["ffmpeg-cpu" if software else "ffmpeg-gpu", path]

        with (
            patch.object(CameraConfig, "get_replay_ffmpeg_cmd", side_effect=build),
            patch.object(
                runner,
                "_start_process",
                side_effect=lambda cmd: fake_process(frame_size, return_code=69),
            ) as start,
            self.assertLogs(LOGGER, level="WARNING"),
        ):
            runner._replay_pending()

        assert start.call_count == 1
        assert len(self._drain(frame_queue)) == 3

    def test_there_is_no_retry_when_the_command_has_no_hardware_to_drop(self):
        runner, _, frame_size = build_runner([segment("/c/a.mp4", 1000.0)])
        runner.trigger(1008.0)

        with (
            patch.object(
                CameraConfig,
                "get_replay_ffmpeg_cmd",
                side_effect=lambda path, software=False: ["ffmpeg", path],
            ),
            patch.object(
                runner,
                "_start_process",
                side_effect=lambda cmd: fake_process(
                    frame_size, times=[], return_code=1
                ),
            ) as start,
            self.assertLogs(LOGGER, level="WARNING"),
        ):
            runner._replay_pending()

        assert start.call_count == 1

    def test_a_segment_that_raises_is_skipped_and_not_retried(self):
        runner, _, _ = build_runner([segment("/c/a.mp4", 1000.0)])
        runner.trigger(1008.0)

        with (
            patch.object(runner, "_start_process", side_effect=RuntimeError("boom")),
            self.assertLogs(LOGGER, level="ERROR"),
        ):
            runner._replay_pending()

        assert runner._cursor == 1010.0

    def test_the_software_command_has_no_hardware_decode_or_scale(self):
        runner, _, _ = build_runner([])
        camera = runner.config
        camera.ffmpeg.hwaccel_args = "preset-intel-qsv-h264"

        hardware = camera.get_replay_ffmpeg_cmd("/c/a.mp4")
        software = camera.get_replay_ffmpeg_cmd("/c/a.mp4", software=True)

        assert "-hwaccel" in hardware
        assert "-hwaccel" not in software
        assert not any("vpp_qsv" in part for part in software)
        assert any("scale=64:48" in part for part in software)
        assert software[software.index("-i") + 1] == "/c/a.mp4"
        assert software[-1] == "pipe:"

    def test_new_segments_are_picked_up_while_triggered(self):
        segments = [segment("/c/a.mp4", 1000.0)]
        runner, frame_queue, frame_size = build_runner(segments)

        runner.trigger(1008.0)
        assert self._replay(runner, frame_size).call_count == 1

        # the open segment finished and another one started
        segments.append(segment("/c/b.mp4", 1010.0))
        assert self._replay(runner, frame_size).call_count == 1

    def test_a_gap_between_segments_is_not_filled_in(self):
        # nothing was recorded from 1010 to 1030
        segments = [segment("/c/a.mp4", 1000.0), segment("/c/c.mp4", 1030.0)]
        runner, frame_queue, frame_size = build_runner(segments)

        runner.trigger(1015.0)
        self._replay(runner, frame_size)
        times = [i[1] for i in self._drain(frame_queue)]

        assert times == [1000.0, 1000.3, 1000.5, 1030.0, 1030.3, 1030.5]

    def test_footage_that_ended_before_the_range_is_skipped_across_a_gap(self):
        segments = [segment("/c/a.mp4", 1000.0), segment("/c/c.mp4", 1030.0)]
        runner, frame_queue, frame_size = build_runner(segments)

        runner.trigger(1035.0)

        assert self._replay(runner, frame_size).call_count == 1
        assert self._drain(frame_queue)[0][1] == 1030.0

    def test_overlapping_segments_are_both_replayed_in_order(self):
        segments = [segment("/c/a.mp4", 1000.0, 10.5), segment("/c/b.mp4", 1010.2)]
        runner, frame_queue, frame_size = build_runner(segments)

        runner.trigger(1015.0)
        self._replay(runner, frame_size)
        times = [i[1] for i in self._drain(frame_queue)]

        assert len(times) == 6
        assert times[:3] == [1000.0, 1000.3, 1000.5]
        assert times[3:] == [1010.2, 1010.5, 1010.7]

    def test_a_short_segment_is_judged_by_its_own_end(self):
        # the stream stalled, so the file holds 4 seconds, not a full segment
        runner, frame_queue, frame_size = build_runner(
            [segment("/c/a.mp4", 1000.0, 4.0)]
        )

        # the range starts at 1006, after the footage ended at 1004
        runner.trigger(1016.0)

        assert self._replay(runner, frame_size).call_count == 0

    def test_a_missing_record_input_is_reported_once_and_does_not_loop(self):
        segments = [segment("/c/a.mp4", 1000.0), segment("/c/b.mp4", 1010.0)]
        runner, _, _ = build_runner(segments)
        runner.trigger(1015.0)

        with (
            patch.object(CameraConfig, "get_replay_ffmpeg_cmd", return_value=None),
            patch.object(runner, "_start_process") as start,
            self.assertLogs(LOGGER, level="WARNING") as logs,
        ):
            runner._replay_pending()

        start.assert_not_called()
        assert len(logs.output) == 1
        assert "record role" in logs.output[0]

    def test_does_nothing_without_a_trigger(self):
        runner, _, _ = build_runner([segment("/c/a.mp4", 1000.0)])

        assert runner._pending() is False

    def test_an_unreadable_segment_is_skipped_and_not_retried(self):
        runner, frame_queue, frame_size = build_runner([segment("/c/a.mp4", 1000.0)])

        runner.trigger(1008.0)
        first = self._replay(runner, frame_size, times=[], return_code=0)
        second = self._replay(runner, frame_size, times=[], return_code=0)

        assert first.call_count == 1
        assert second.call_count == 0
        assert frame_queue.empty()

    def test_a_clean_segment_without_frames_is_not_a_warning(self):
        runner, _, frame_size = build_runner([segment("/c/a.mp4", 1000.0)])

        runner.trigger(1008.0)

        with self.assertNoLogs(LOGGER, level="WARNING"):
            self._replay(runner, frame_size, times=[])

    def test_ffmpeg_failure_is_logged_with_its_exit_code_and_error(self):
        runner, frame_queue, frame_size = build_runner([segment("/c/a.mp4", 1000.0)])

        runner.trigger(1008.0)

        with self.assertLogs(LOGGER, level="WARNING") as logs:
            self._replay(
                runner,
                frame_size,
                times=[],
                return_code=183,
                errors=(b"[mov,mp4 @ 0x2] [error] moov atom not found\n",),
            )

        message = "\n".join(logs.output)
        assert "183" in message
        assert "moov atom not found" in message
        assert frame_queue.empty()

    def test_a_decode_error_after_some_frames_keeps_them_and_warns(self):
        runner, frame_queue, frame_size = build_runner([segment("/c/a.mp4", 1000.0)])

        runner.trigger(1008.0)

        with self.assertLogs(LOGGER, level="WARNING") as logs:
            self._replay(runner, frame_size, return_code=69)

        assert len(self._drain(frame_queue)) == 3
        assert "69" in "\n".join(logs.output)

    def test_a_truncated_last_frame_is_dropped_and_warned_about(self):
        runner, frame_queue, frame_size = build_runner([segment("/c/a.mp4", 1000.0)])

        runner.trigger(1008.0)

        with self.assertLogs(LOGGER, level="WARNING"):
            self._replay(runner, frame_size, trailing=b"\x00" * 100)

        assert len(self._drain(frame_queue)) == 3

    def test_a_frame_without_a_capture_time_is_skipped(self):
        runner, frame_queue, frame_size = build_runner([segment("/c/a.mp4", 1000.0)])

        runner.trigger(1008.0)

        with self.assertLogs(LOGGER, level="WARNING"):
            self._replay(runner, frame_size, times=["0", "nan", "0.5"])

        assert [i[1] for i in self._drain(frame_queue)] == [1000.0, 1000.5]

    def test_a_missing_shared_memory_buffer_skips_the_frame(self):
        runner, frame_queue, frame_size = build_runner([segment("/c/a.mp4", 1000.0)])
        runner.frame_manager.write.side_effect = lambda name: None

        runner.trigger(1008.0)

        with self.assertLogs(LOGGER, level="WARNING"):
            self._replay(runner, frame_size)

        assert frame_queue.empty()
        assert runner._frame_index == 0

    def test_frame_buffers_are_closed_by_the_writer_and_never_deleted(self):
        runner, frame_queue, frame_size = build_runner([segment("/c/a.mp4", 1000.0)])

        runner.trigger(1008.0)
        self._replay(runner, frame_size)
        names = [i[0] for i in self._drain(frame_queue)]

        assert names == ["front_door_frame0", "front_door_frame1", "front_door_frame2"]
        assert [c.args[0] for c in runner.frame_manager.close.call_args_list] == names
        runner.frame_manager.delete.assert_not_called()

    def test_frame_names_wrap_at_the_shared_memory_frame_count(self):
        segments = [segment(f"/c/{i}.mp4", 1000.0 + 10 * i) for i in range(4)]
        runner, frame_queue, frame_size = build_runner(segments)

        runner.trigger(1005.0)
        self._replay(runner, frame_size)
        names = [i[0] for i in self._drain(frame_queue)]

        assert len(names) == 12
        assert names[9] == "front_door_frame9"
        assert names[10] == "front_door_frame0"

    def test_a_full_queue_slows_replay_down_without_dropping_frames(self):
        runner, _, frame_size = build_runner([segment("/c/a.mp4", 1000.0)])
        runner.frame_queue = queue.Queue(maxsize=1)
        received: list[tuple] = []

        def consume():
            while len(received) < 3:
                received.append(runner.frame_queue.get(timeout=5))

        consumer = threading.Thread(target=consume, daemon=True)
        consumer.start()

        runner.trigger(1008.0)
        self._replay(runner, frame_size)
        consumer.join(timeout=5)

        assert [i[1] for i in received] == [1000.0, 1000.3, 1000.5]

    def test_stop_ends_a_replay_stuck_on_a_full_queue(self):
        runner, _, frame_size = build_runner([segment("/c/a.mp4", 1000.0)])
        runner.frame_queue = queue.Queue(maxsize=1)
        runner.frame_queue.put("occupied")
        stopper = threading.Timer(0.3, runner._local_stop.set)
        stopper.start()
        self.addCleanup(stopper.cancel)

        with patch.object(
            runner, "_start_process", return_value=fake_process(frame_size)
        ):
            runner._replay_segment(segment("/c/a.mp4", 1000.0))

        assert runner.frame_queue.qsize() == 1
        assert runner.active is False

    def test_resolution_comes_from_the_detect_config(self):
        runner, _, frame_size = build_runner([], detect={"width": 128, "height": 96})

        assert runner.frame_size == 96 * 3 // 2 * 128

    def test_stop_ends_the_thread_and_terminates_a_running_process(self):
        runner, _, _ = build_runner([])
        process = MagicMock()
        runner._process = process
        runner.start()

        runner.stop()

        process.terminate.assert_called_once()
        assert not runner.is_alive()

    def test_a_process_that_stops_producing_frames_is_killed(self):
        runner, _, _ = build_runner([])
        process = MagicMock()
        runner._read_started = 0.0

        with (
            patch("frigate.video.detect_replay.time") as fake_time,
            self.assertLogs(LOGGER, level="WARNING") as logs,
        ):
            fake_time.time.return_value = 1000.0
            runner._guard(process, threading.Event())

        process.kill.assert_called_once()
        assert "killing it" in logs.output[0]

    def test_the_guard_leaves_a_process_that_is_not_waiting_on_a_frame(self):
        runner, _, _ = build_runner([])
        process = MagicMock()
        done = threading.Event()
        runner._read_started = None
        threading.Timer(0.3, done.set).start()

        runner._guard(process, done)

        process.kill.assert_not_called()

    def test_ffmpeg_is_stopped_even_if_setting_up_the_helpers_fails(self):
        runner, _, frame_size = build_runner([segment("/c/a.mp4", 1000.0)])
        process = fake_process(frame_size)
        process.poll.return_value = None

        with (
            patch.object(runner, "_start_process", return_value=process),
            patch(
                "frigate.video.detect_replay.FfmpegStderr.start",
                side_effect=RuntimeError("cannot start new thread"),
            ),
            self.assertRaises(RuntimeError),
        ):
            runner._replay_segment(segment("/c/a.mp4", 1000.0))

        process.terminate.assert_called_once()
        assert runner._process is None
        assert runner.active is False

    def test_the_frame_buffer_is_closed_even_if_copying_into_it_fails(self):
        runner, frame_queue, _ = build_runner([])
        buffer = MagicMock()
        buffer.__setitem__.side_effect = ValueError("wrong size")
        runner.frame_manager.write.side_effect = lambda name: buffer

        with self.assertRaises(ValueError):
            runner._feed(b"\x00", 1000.0)

        runner.frame_manager.close.assert_called_once_with("front_door_frame0")
        assert frame_queue.empty()
        assert runner._frame_index == 0

    def test_the_replay_ffmpeg_command_is_built_for_the_segment(self):
        runner, _, _ = build_runner([])

        cmd = runner.config.get_replay_ffmpeg_cmd("/c/a.mp4")

        assert cmd[cmd.index("-i") + 1] == "/c/a.mp4"
        assert cmd[-1] == "pipe:"

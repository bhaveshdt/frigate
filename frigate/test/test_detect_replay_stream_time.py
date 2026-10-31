"""Tests for which stream replay reads and for the capture times it hands out.

By design replay reads the stream that records: the input with the record role,
and its main stream cache segments. A detect-only or sub stream input never
supplies footage. A frame's capture time is the start of its segment plus its
presentation time, and the tracker must only ever see times that move forward
and that belong to the footage they came with.
"""

import io
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from frigate.test.test_camera_watchdog_replay import build_config
from frigate.test.test_detect_replay import LOGGER, build_runner, fake_process, segment
from frigate.util import cache_frame
from frigate.util.cache_frame import (
    CacheSegment,
    clear_probe_cache,
    get_latest_finished_cache_segment,
    list_finished_cache_segments,
    probe_segment,
)
from frigate.video.detect_replay import FfmpegStderr, frame_timestamp


def replay_once(times: list[str], release_at: float | None = None):
    """Replay one 10 second segment starting at 1000, returns runner and times fed."""
    runner, frame_queue, frame_size = build_runner([segment("/c/a.mp4", 1000.0)])

    if release_at is None:
        runner.trigger(1008.0)
    else:
        runner.tail = 4.0
        runner.trigger(1001.0)
        runner.release(release_at)

    with (
        patch.object(
            runner,
            "_start_process",
            return_value=fake_process(frame_size, times=times),
        ),
        patch("frigate.video.detect_replay.time") as clock,
    ):
        clock.time.return_value = 1100.0
        runner._replay_pending()

    fed = []

    while not frame_queue.empty():
        fed.append(frame_queue.get()[1])

    return runner, fed


class TestReplaySource(unittest.TestCase):
    def test_replay_uses_the_record_inputs_settings_not_the_detect_inputs(self):
        camera = build_config(
            {"mode": "replay"},
            inputs=[
                {
                    "path": "rtsp://10.0.0.1:554/main",
                    "roles": ["record"],
                    "hwaccel_args": "preset-intel-qsv-h264",
                },
                {
                    "path": "rtsp://10.0.0.1:554/sub",
                    "roles": ["detect"],
                    "hwaccel_args": "preset-nvidia",
                },
            ],
        ).cameras["front_door"]

        joined = " ".join(camera.get_replay_ffmpeg_cmd("/tmp/cache/front_door@1.mp4"))

        assert "h264_qsv" in joined
        assert "cuda" not in joined
        assert "cuvid" not in joined

    def test_no_stream_is_decoded_continuously_in_replay_mode(self):
        camera = build_config(
            {"mode": "replay"},
            inputs=[
                {"path": "rtsp://10.0.0.1:554/main", "roles": ["record"]},
                {"path": "rtsp://10.0.0.1:554/sub", "roles": ["detect"]},
            ],
        ).cameras["front_door"]

        assert all("detect" not in cmd["roles"] for cmd in camera.ffmpeg_cmds)


class TestCacheSelection(unittest.TestCase):
    NOW = 1767225900.0

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _touch(self, name: str, age: float) -> str:
        path = os.path.join(self.tmp.name, name)
        open(path, "w").close()
        os.utime(path, (self.NOW - age, self.NOW - age))
        return path

    def _latest(self) -> str | None:
        return get_latest_finished_cache_segment(
            "front_door", self.tmp.name, now=self.NOW
        )

    def test_sub_stream_segments_are_never_used_when_the_main_stream_has_none(self):
        self._touch("front_door@sub@20260101000000+0000.mp4", 30)
        self._touch("front_door@sub@20260101000010+0000.mp4", 20)

        with patch.object(cache_frame, "probe_segment", return_value=(0.0, 10.0)):
            listing = list_finished_cache_segments(
                "front_door", self.tmp.name, in_use=lambda _: set()
            )

        assert listing == []
        assert self._latest() is None

    def test_a_newer_sub_stream_segment_does_not_change_the_latest_main_one(self):
        first = self._touch("front_door@20260101000000+0000.mp4", 30)
        self._touch("front_door@20260101000010+0000.mp4", 20)
        self._touch("front_door@sub@20260101000020+0000.mp4", 5)

        assert self._latest() == first

    def test_a_file_with_a_malformed_name_is_not_a_segment_for_the_latest_frame(self):
        first = self._touch("front_door@20260101000000+0000.mp4", 30)
        self._touch("front_door@20260101000010+0000.mp4", 20)
        # not a segment, so it must not make the newest real one look finished
        self._touch("front_door@notadate.mp4", 5)

        assert self._latest() == first


class TestCaptureTimes(unittest.TestCase):
    def setUp(self):
        clear_probe_cache()

    def test_a_negative_first_presentation_time_is_removed(self):
        first = CacheSegment("/c/a.mp4", 1000.0, 1010.0, -0.04)

        self.assertAlmostEqual(frame_timestamp(first, -0.04), 1000.0)
        self.assertAlmostEqual(frame_timestamp(first, 0.26), 1000.3)

    def test_a_negative_start_time_is_read_from_the_probe(self):
        result = MagicMock(
            returncode=0,
            stdout='{"format": {"start_time": "-0.040000", "duration": "9.96"}}',
        )

        with patch("subprocess.run", return_value=result):
            assert probe_segment("ffprobe", "/c/a.mp4", 1.0) == (-0.04, 9.96)

    def test_showinfo_lines_with_negative_and_missing_times_are_read(self):
        lines = (
            b"[Parsed_showinfo_1 @ 0x55d] [info] n:   0 pts: -1024 pts_time:-0.04 x\n"
            b"[Parsed_showinfo_1 @ 0x55d] [info] n:   1 pts:NOPTS pts_time:NOPTS\n"
            b"[Parsed_showinfo_1 @ 0x55d] [info] n:   2 pts:  3072 pts_time:0.12 x\n"
        )
        reader = FfmpegStderr(io.BytesIO(lines))

        reader.run()

        times = []

        while not reader.times.empty():
            times.append(reader.times.get())

        assert times == [-0.04, None, 0.12]

    def test_a_duplicate_capture_time_is_fed_once(self):
        runner, fed = replay_once(["0", "0.3", "0.3", "0.6"])

        assert fed == [1000.0, 1000.3, 1000.6]
        assert runner._cursor == 1010.0

    def test_a_frame_out_of_order_within_a_segment_is_not_fed(self):
        _, fed = replay_once(["0", "0.5", "0.3", "0.9"])

        assert fed == [1000.0, 1000.5, 1000.9]
        assert fed == sorted(set(fed))

    def test_a_frame_at_the_very_end_of_the_segment_is_kept(self):
        _, fed = replay_once(["0", "9.99"])

        assert fed == [1000.0, 1009.99]

    def test_a_capture_time_far_ahead_of_its_segment_is_not_fed(self):
        runner, fed = replay_once(["0", "0.3", "500.0", "500.3", "0.6"])

        # one bad time must not push the newest fed time ahead and make every
        # real frame after it look like one the tracker already has
        assert fed == [1000.0, 1000.3, 1000.6]
        assert runner.progress()[1] == 1000.6

    def test_frames_outside_their_segment_are_reported_not_dropped_silently(self):
        with self.assertLogs(LOGGER, level="WARNING") as logs:
            replay_once(["0", "0.3", "-500.0", "0.6"])

        assert any("/c/a.mp4" in line for line in logs.output)

    def test_a_far_ahead_capture_time_does_not_look_like_the_end_of_the_range(self):
        runner, fed = replay_once(["0", "500.0", "0.3"], release_at=1002.0)

        # the range ends at 1006, a time at 1500 is bad data and not a reason
        # to stop reading the segment
        assert fed == [1000.0, 1000.3]
        assert runner._cursor == 1010.0


if __name__ == "__main__":
    unittest.main()

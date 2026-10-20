"""Regression tests for replay coverage accounting and segment probe retries.

The replay cursor says how much footage has been dealt with. It may only pass
footage that was fed to the tracker, or that was given up on by a documented
rule, and a failed probe must not hide a segment forever.
"""

import os
import subprocess as sp
import tempfile
import unittest
from collections import deque
from unittest.mock import MagicMock, patch

from frigate.test.test_detect_replay import (
    LOGGER,
    build_runner,
    fake_process,
    segment,
)
from frigate.util import cache_frame
from frigate.util.cache_frame import (
    SegmentListing,
    clear_probe_cache,
    list_finished_cache_segments,
    probe_segment,
)

PROBE_OK = '{"format": {"start_time": "0.000000", "duration": "9.983000"}}'


def probe_result(stdout: str = "", returncode: int = 0) -> MagicMock:
    return MagicMock(stdout=stdout, returncode=returncode)


def probe_failures() -> dict[str, dict]:
    """The ways ffprobe can fail, as keyword arguments for patching run."""
    return {
        "timeout": {"side_effect": sp.TimeoutExpired("ffprobe", 10)},
        "missing binary": {"side_effect": FileNotFoundError},
        "nonzero exit": {"return_value": probe_result("", 1)},
        "invalid json": {"return_value": probe_result("not json")},
        "zero duration": {
            "return_value": probe_result(
                '{"format": {"start_time": "0", "duration": "0"}}'
            )
        },
    }


class TestProbeRetry(unittest.TestCase):
    def setUp(self):
        clear_probe_cache()

    @staticmethod
    def _probe(run: dict, now: float, mtime: float = 1.0):
        with (
            patch("subprocess.run", **run) as sub,
            patch.object(cache_frame, "time") as clock,
        ):
            clock.time.return_value = now
            result = probe_segment("ffprobe", "/c/a.mp4", mtime)

        return result, sub

    def test_a_failed_probe_is_not_remembered_forever(self):
        for name, failure in probe_failures().items():
            with self.subTest(failure=name):
                clear_probe_cache()

                first, _ = self._probe(failure, 1000.0)
                second, _ = self._probe(
                    {"return_value": probe_result(PROBE_OK)}, 1000.0 + 3600
                )

                assert first is None
                # same path, same mtime: only the earlier failure was cached
                assert second == (0.0, 9.983)

    def test_a_successful_probe_is_cached(self):
        with (
            patch("subprocess.run", return_value=probe_result(PROBE_OK)) as sub,
            patch.object(cache_frame, "time") as clock,
        ):
            clock.time.return_value = 1000.0

            for _ in range(20):
                assert probe_segment("ffprobe", "/c/a.mp4", 1.0) == (0.0, 9.983)

        assert sub.call_count == 1

    def test_a_broken_segment_is_not_probed_in_a_tight_loop(self):
        with (
            patch("subprocess.run", return_value=probe_result("", 1)) as sub,
            patch.object(cache_frame, "time") as clock,
        ):
            clock.time.return_value = 1000.0

            for _ in range(50):
                assert probe_segment("ffprobe", "/c/a.mp4", 1.0) is None

        assert sub.call_count == 1

    def test_retries_of_a_broken_segment_are_bounded(self):
        with (
            patch("subprocess.run", return_value=probe_result("", 1)) as sub,
            patch.object(cache_frame, "time") as clock,
        ):
            for hour in range(30):
                clock.time.return_value = 1000.0 + 3600 * hour
                assert probe_segment("ffprobe", "/c/a.mp4", 1.0) is None

        # it is tried again later, but not forever
        assert 1 < sub.call_count <= 10

    def test_a_rewritten_file_is_probed_again_after_retries_ran_out(self):
        with (
            patch("subprocess.run", return_value=probe_result("", 1)) as sub,
            patch.object(cache_frame, "time") as clock,
        ):
            for hour in range(30):
                clock.time.return_value = 1000.0 + 3600 * hour
                probe_segment("ffprobe", "/c/a.mp4", 1.0)

            before = sub.call_count
            clock.time.return_value = 1000.0 + 3600 * 31
            probe_segment("ffprobe", "/c/a.mp4", 2.0)

        assert sub.call_count == before + 1


class TestListingHoldsBackUnprobedFootage(unittest.TestCase):
    # 2026-01-01T00:00:00Z
    T0 = 1767225600.0

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.paths = []

        for second in (0, 10, 20):
            name = f"front_door@202601010000{second:02d}+0000.mp4"
            path = os.path.join(self.tmp.name, name)
            open(path, "w").close()
            self.paths.append(path)

    def _list(self, unreadable: str | None, exhausted: bool = False):
        def probe(_ffprobe, path, _mtime):
            return None if path == unreadable else (0.0, 9.98)

        with (
            patch("frigate.util.cache_frame.probe_segment", side_effect=probe),
            patch("frigate.util.cache_frame.probe_exhausted", return_value=exhausted),
        ):
            return list_finished_cache_segments(
                "front_door", self.tmp.name, in_use=lambda _: set()
            )

    def test_nothing_is_held_back_when_every_segment_can_be_probed(self):
        listing = self._list(None)

        assert [s.path for s in listing] == self.paths
        assert listing.blocked_from is None

    def test_a_segment_that_may_still_be_probed_says_where_it_begins(self):
        listing = self._list(self.paths[1])

        assert [s.path for s in listing] == [self.paths[0], self.paths[2]]
        assert listing.blocked_from == self.T0 + 10

    def test_a_segment_given_up_on_no_longer_holds_anything_back(self):
        listing = self._list(self.paths[1], exhausted=True)

        assert [s.path for s in listing] == [self.paths[0], self.paths[2]]
        assert listing.blocked_from is None


class TestReplayCoverage(unittest.TestCase):
    def _runner(self, segments):
        runner, frame_queue, frame_size = build_runner(segments)
        self.frame_size = frame_size
        return runner, frame_queue

    def _pass(self, runner, processes: deque, now: float = 1100.0) -> list:
        """One _replay_pending call, each ffmpeg start takes the next factory."""
        starts: list = []

        def start(cmd):
            starts.append(cmd)

            if processes:
                return processes.popleft()()

            return fake_process(self.frame_size)

        with (
            patch.object(runner, "_start_process", side_effect=start),
            patch("frigate.video.detect_replay.time") as clock,
        ):
            clock.time.return_value = now
            runner._replay_pending()

        return starts

    @staticmethod
    def _drain(frame_queue) -> list[float]:
        times = []

        while not frame_queue.empty():
            times.append(frame_queue.get()[1])

        return times

    def test_an_exception_before_the_first_frame_leaves_the_footage_uncovered(self):
        runner, _ = self._runner([segment("/c/a.mp4", 1000.0)])
        runner.trigger(1008.0)

        def boom():
            raise RuntimeError("boom")

        with self.assertLogs(LOGGER, level="ERROR"):
            starts = self._pass(runner, deque([boom]))

        assert len(starts) == 1
        # nothing of the segment was processed
        assert runner._cursor <= 1000.0

    def test_a_nonzero_exit_without_frames_leaves_the_footage_uncovered(self):
        runner, _ = self._runner([segment("/c/a.mp4", 1000.0)])
        runner.trigger(1008.0)
        processes = deque(
            [lambda: fake_process(self.frame_size, times=[], return_code=1)]
        )

        with self.assertLogs(LOGGER, level="WARNING"):
            starts = self._pass(runner, processes)

        assert len(starts) == 1
        assert runner._cursor <= 1000.0

    def test_a_failure_after_some_frames_covers_only_those_frames(self):
        runner, frame_queue = self._runner([segment("/c/a.mp4", 1000.0)])
        runner.trigger(1008.0)
        processes = deque(
            [
                lambda: fake_process(
                    self.frame_size, times=["0", "0.3", "0.5"], return_code=69
                ),
                lambda: fake_process(
                    self.frame_size, times=["0", "0.3", "0.5", "0.9", "1.4"]
                ),
            ]
        )

        with self.assertLogs(LOGGER, level="WARNING"):
            self._pass(runner, processes, now=1100.0)

        self.assertAlmostEqual(runner._cursor, 1000.5)

        # a later pass finishes the segment without feeding anything twice
        self._pass(runner, processes, now=1100.0 + 10_000)

        assert self._drain(frame_queue) == [1000.0, 1000.3, 1000.5, 1000.9, 1001.4]
        assert runner._cursor == 1010.0

    def test_truncated_output_covers_only_the_complete_frames(self):
        runner, _ = self._runner([segment("/c/a.mp4", 1000.0)])
        runner.trigger(1008.0)
        processes = deque(
            [lambda: fake_process(self.frame_size, trailing=b"\x00" * 100)]
        )

        with self.assertLogs(LOGGER, level="WARNING"):
            self._pass(runner, processes)

        self.assertAlmostEqual(runner._cursor, 1000.5)

    def test_a_frame_without_a_capture_time_covers_only_the_frames_before_it(self):
        runner, _ = self._runner([segment("/c/a.mp4", 1000.0)])
        runner.trigger(1008.0)
        processes = deque(
            [lambda: fake_process(self.frame_size, times=["0", "nan", "0.5"])]
        )

        with self.assertLogs(LOGGER, level="WARNING"):
            self._pass(runner, processes)

        self.assertAlmostEqual(runner._cursor, 1000.0)

    def test_stopping_midway_through_a_segment_does_not_cover_the_rest(self):
        runner, frame_queue = self._runner([segment("/c/a.mp4", 1000.0)])
        runner.trigger(1008.0)
        real_put = frame_queue.put

        def put(item, *args, **kwargs):
            real_put(item, *args, **kwargs)

            # shutdown arrives while the second frame is being handed over
            if frame_queue.qsize() == 2:
                runner._local_stop.set()

        frame_queue.put = put

        self._pass(runner, deque())

        assert self._drain(frame_queue) == [1000.0, 1000.3]
        self.assertAlmostEqual(runner._cursor, 1000.3)
        # a replacement runner starts from here, so it must not skip the rest
        assert runner.progress() == (runner._cursor, 1000.3)

    def test_a_clean_segment_without_frames_is_fully_covered(self):
        runner, _ = self._runner([segment("/c/a.mp4", 1000.0)])
        runner.trigger(1008.0)
        processes = deque([lambda: fake_process(self.frame_size, times=[])])

        starts = self._pass(runner, processes)

        assert len(starts) == 1
        assert runner._cursor == 1010.0

    def test_a_failed_segment_is_not_retried_in_a_tight_loop(self):
        runner, _ = self._runner([segment("/c/a.mp4", 1000.0)])
        runner.trigger(1008.0)
        processes = deque(
            [
                lambda: fake_process(self.frame_size, times=[], return_code=1)
                for _ in range(5)
            ]
        )

        with self.assertLogs(LOGGER, level="WARNING"):
            first = self._pass(runner, processes, now=1100.0)
            second = self._pass(runner, processes, now=1100.0)

        assert len(first) == 1
        assert second == []

    def test_a_segment_that_keeps_failing_is_retried_then_excluded_loudly(self):
        runner, _ = self._runner([segment("/c/a.mp4", 1000.0)])
        runner.trigger(1008.0)
        processes = deque(
            [
                lambda: fake_process(self.frame_size, times=[], return_code=1)
                for _ in range(100)
            ]
        )
        starts = 0

        with self.assertLogs(LOGGER, level="ERROR") as logs:
            for attempt in range(30):
                now = 1100.0 + 1000 * attempt
                starts += len(self._pass(runner, processes, now=now))

        # tried more than once, but never forever
        assert 1 < starts <= 10
        # giving up is the only way the cursor passes unprocessed footage
        assert runner._cursor == 1010.0
        assert any("/c/a.mp4" in line for line in logs.output)

    def test_a_failed_segment_holds_back_the_ones_after_it(self):
        segments = [segment("/c/a.mp4", 1000.0), segment("/c/b.mp4", 1010.0)]
        runner, frame_queue = self._runner(segments)
        runner.trigger(1015.0)
        processes = deque(
            [
                lambda: fake_process(self.frame_size, times=[], return_code=1),
                lambda: fake_process(self.frame_size),
                lambda: fake_process(self.frame_size),
            ]
        )

        with self.assertLogs(LOGGER, level="WARNING"):
            first = self._pass(runner, processes, now=1100.0)

        # the tracker needs frames in order, so b waits for a
        assert len(first) == 1
        assert frame_queue.empty()

        self._pass(runner, processes, now=1100.0 + 10_000)

        assert self._drain(frame_queue) == [
            1000.0,
            1000.3,
            1000.5,
            1010.0,
            1010.3,
            1010.5,
        ]
        assert runner._cursor == 1020.0

    def test_footage_that_cannot_be_probed_yet_is_not_skipped(self):
        first = segment("/c/a.mp4", 1000.0)
        later = segment("/c/c.mp4", 1020.0)
        runner, frame_queue = self._runner([])
        # /c/b.mp4 at 1010 is finished but ffprobe has not read it yet
        listing = SegmentListing([first, later])
        listing.blocked_from = 1010.0
        runner.list_segments = lambda *_: listing
        runner.trigger(1015.0)

        starts = self._pass(runner, deque())

        assert len(starts) == 1
        assert runner._cursor == 1010.0
        assert self._drain(frame_queue) == [1000.0, 1000.3, 1000.5]

        # once it can be read nothing is missing from the replay
        middle = segment("/c/b.mp4", 1010.0)
        runner.list_segments = lambda *_: [first, middle, later]
        self._pass(runner, deque())

        assert self._drain(frame_queue) == [
            1010.0,
            1010.3,
            1010.5,
            1020.0,
            1020.3,
            1020.5,
        ]
        assert runner._cursor == 1030.0


if __name__ == "__main__":
    unittest.main()

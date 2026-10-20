"""Tests for replay recovery paths: hardware fallback, frames that cannot be
written, and the watchdog's supervision of the replay thread."""

import unittest
from collections import deque
from unittest.mock import MagicMock, patch

from frigate.config import CameraConfig
from frigate.const import STREAM_TYPE_MAIN
from frigate.test.test_camera_watchdog_replay import build_watchdog
from frigate.test.test_detect_replay import LOGGER, build_runner, fake_process, segment
from frigate.video.detect_replay import HARDWARE_RETRY_AFTER_S


def build_command(path, software=False):
    return ["ffmpeg-cpu" if software else "ffmpeg-gpu", path]


def run_pass(runner, processes: deque, now: float = 1100.0) -> list[str]:
    """One _replay_pending call, returns the decoder of each ffmpeg started."""
    commands: list[str] = []

    def start(cmd):
        commands.append(cmd[0])
        return processes.popleft()

    with (
        patch.object(CameraConfig, "get_replay_ffmpeg_cmd", side_effect=build_command),
        patch.object(runner, "_start_process", side_effect=start),
        patch("frigate.video.detect_replay.time") as clock,
    ):
        clock.time.return_value = now
        runner._replay_pending()

    return commands


def drain(frame_queue) -> list[float]:
    times = []

    while not frame_queue.empty():
        times.append(frame_queue.get()[1])

    return times


class TestHardwareFallback(unittest.TestCase):
    def test_a_hardware_failure_after_some_frames_resumes_on_the_cpu(self):
        runner, frame_queue, frame_size = build_runner([segment("/c/a.mp4", 1000.0)])
        runner.trigger(1008.0)
        processes = deque(
            [
                # the hardware decoder dies after three frames
                fake_process(frame_size, times=["0", "0.3", "0.5"], return_code=69),
                # the CPU reads the whole segment
                fake_process(frame_size, times=["0", "0.3", "0.5", "0.9", "1.4"]),
            ]
        )

        with self.assertLogs(LOGGER, level="WARNING"):
            commands = run_pass(runner, processes)

        fed = drain(frame_queue)

        assert commands == ["ffmpeg-gpu", "ffmpeg-cpu"]
        # the rest of the segment, and nothing twice or out of order
        assert fed == [1000.0, 1000.3, 1000.5, 1000.9, 1001.4]
        assert runner._cursor == 1010.0
        # the decoder is the problem, so the next segments skip the hardware
        assert runner._software_until == 1100.0 + HARDWARE_RETRY_AFTER_S

    def test_when_the_cpu_fails_too_nothing_is_repeated_or_skipped(self):
        runner, frame_queue, frame_size = build_runner([segment("/c/a.mp4", 1000.0)])
        runner.trigger(1008.0)
        processes = deque(
            [
                fake_process(frame_size, times=["0", "0.3", "0.5"], return_code=69),
                fake_process(frame_size, times=["0", "0.3", "0.5"], return_code=69),
            ]
        )

        with self.assertLogs(LOGGER, level="WARNING"):
            commands = run_pass(runner, processes)

        assert commands == ["ffmpeg-gpu", "ffmpeg-cpu"]
        assert drain(frame_queue) == [1000.0, 1000.3, 1000.5]
        # the rest of the segment is still owed, so the cursor stays at the frames
        self.assertAlmostEqual(runner._cursor, 1000.5)
        # it was the footage that failed, not the decoder
        assert runner._software_until == 0.0


class TestUnwritableFrames(unittest.TestCase):
    def test_a_frame_that_cannot_be_written_is_not_skipped_over(self):
        runner, frame_queue, frame_size = build_runner([segment("/c/a.mp4", 1000.0)])
        runner.frame_manager.write.side_effect = lambda name: None
        runner.trigger(1008.0)

        with self.assertLogs(LOGGER, level="WARNING") as logs:
            run_pass(runner, deque([fake_process(frame_size)]))

        assert frame_queue.empty()
        # nothing reached the tracker, so nothing of the segment is consumed
        assert runner._cursor <= 1000.0
        # reported once for the segment, not once for every frame
        reports = [line for line in logs.output if "shared memory" in line]
        assert len(reports) == 1

    def test_the_segment_is_replayed_once_writing_works_again(self):
        runner, frame_queue, frame_size = build_runner([segment("/c/a.mp4", 1000.0)])
        working = runner.frame_manager.write.side_effect
        runner.frame_manager.write.side_effect = lambda name: None
        runner.trigger(1008.0)

        with self.assertLogs(LOGGER, level="WARNING"):
            run_pass(runner, deque([fake_process(frame_size)]), now=1100.0)

        runner.frame_manager.write.side_effect = working
        run_pass(runner, deque([fake_process(frame_size)]), now=1100.0 + 1000)

        assert drain(frame_queue) == [1000.0, 1000.3, 1000.5]
        assert runner._cursor == 1010.0


class TestReplaySupervision(unittest.TestCase):
    def setUp(self):
        self.watchdog = build_watchdog({"mode": "replay"})
        self.watchdog.detect_status = MagicMock()
        self.watchdog.replay_offline = MagicMock()

    @staticmethod
    def _alive(active: bool = False) -> MagicMock:
        runner = MagicMock()
        runner.is_alive.return_value = True
        runner.active = active
        return runner

    def _statuses(self) -> list[str]:
        return [
            call.args[0] for call in self.watchdog.detect_status.send.call_args_list
        ]

    def test_a_dead_replay_thread_is_reported_once_not_every_second(self):
        with (
            self.assertLogs("watchdog.front_door", level="ERROR") as logs,
            patch.object(self.watchdog, "start_replay") as start,
        ):
            results = [
                self.watchdog._check_replay(100.0 + second, can_restart=False)
                for second in range(5)
            ]

        assert len(logs.output) == 1
        assert results == [False] * 5
        start.assert_not_called()
        assert self._statuses() == ["idle"] * 5

    def test_a_dead_replay_thread_is_started_again_once_pacing_allows(self):
        with (
            self.assertLogs("watchdog.front_door", level="ERROR"),
            patch.object(self.watchdog, "start_replay") as start,
        ):
            restarted = self.watchdog._check_replay(100.0, can_restart=True)

        assert restarted is True
        start.assert_called_once()

    def test_an_attempt_that_had_to_wait_still_counts_as_a_restart(self):
        # the previous thread has not ended, so start_replay declines
        with (
            self.assertLogs("watchdog.front_door", level="ERROR"),
            patch.object(self.watchdog, "start_replay", return_value=False),
        ):
            restarted = self.watchdog._check_replay(100.0, can_restart=True)

        assert restarted is True

    def test_a_second_outage_is_reported_again(self):
        with (
            self.assertLogs("watchdog.front_door", level="ERROR") as logs,
            patch.object(self.watchdog, "start_replay"),
        ):
            self.watchdog._check_replay(100.0, can_restart=False)
            self.watchdog.replay_runner = self._alive()
            self.watchdog._check_replay(101.0, can_restart=False)
            self.watchdog.replay_runner = None
            self.watchdog._check_replay(102.0, can_restart=False)

        assert len(logs.output) == 2

    def test_an_idle_replay_thread_is_healthy_and_left_alone(self):
        self.watchdog.replay_runner = self._alive(active=False)

        with patch.object(self.watchdog, "start_replay") as start:
            restarted = self.watchdog._check_replay(100.0, can_restart=True)

        assert restarted is False
        start.assert_not_called()
        assert self._statuses() == ["idle"]
        assert self.watchdog.replay_offline.value == 0

    def test_a_replay_thread_that_is_decoding_is_online(self):
        self.watchdog.replay_runner = self._alive(active=True)

        self.watchdog._check_replay(100.0, can_restart=True)

        assert self._statuses() == ["online"]

    def test_a_stream_that_is_not_recording_shows_as_offline(self):
        self.watchdog.replay_runner = self._alive(active=True)
        self.watchdog.record_status[STREAM_TYPE_MAIN].last_status = "offline"

        self.watchdog._check_replay(100.0, can_restart=True)

        assert self._statuses() == ["offline"]
        assert self.watchdog.replay_offline.value == 1


if __name__ == "__main__":
    unittest.main()

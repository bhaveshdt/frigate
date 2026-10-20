"""Regression tests for replay runner ownership and the ffmpeg process lifecycle.

Only one runner may write a camera's replay frames at a time. A stop that timed
out is not proof the old thread ended, so a replacement waits until it has.
"""

import subprocess as sp
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from frigate.test.test_camera_watchdog_replay import build_watchdog
from frigate.test.test_detect_replay import build_runner, segment
from frigate.video.detect_replay import DetectReplayRunner


def stuck_runner() -> MagicMock:
    """A runner whose thread is still alive after it was asked to stop."""
    runner = MagicMock()
    runner.is_alive.return_value = True
    runner.progress.return_value = (1010.0, 1009.5)
    return runner


class TestRunnerOwnership(unittest.TestCase):
    def test_a_runner_that_did_not_stop_is_not_replaced(self):
        watchdog = build_watchdog({"mode": "replay"})
        watchdog.replay_runner = stuck_runner()

        watchdog.stop_replay()

        with patch("frigate.video.ffmpeg.DetectReplayRunner") as runner_class:
            watchdog.start_replay()

        runner_class.assert_not_called()

    def test_it_is_replaced_from_where_it_got_to_once_it_has_ended(self):
        watchdog = build_watchdog({"mode": "replay"})
        old = stuck_runner()
        watchdog.replay_runner = old
        watchdog.stop_replay()

        old.is_alive.return_value = False

        with patch("frigate.video.ffmpeg.DetectReplayRunner") as runner_class:
            watchdog.start_replay()

        runner_class.assert_called_once()
        assert runner_class.call_args.kwargs["cursor"] == 1010.0
        assert runner_class.call_args.kwargs["last_fed"] == 1009.5

    def test_progress_is_not_read_while_the_old_runner_can_still_change_it(self):
        watchdog = build_watchdog({"mode": "replay"})
        old = stuck_runner()
        watchdog.replay_runner = old

        watchdog.stop_replay()

        old.progress.assert_not_called()

    def test_a_restart_does_not_start_a_second_runner_beside_a_running_one(self):
        watchdog = build_watchdog({"mode": "replay"})
        watchdog.replay_runner = stuck_runner()

        with (
            patch("frigate.video.ffmpeg.DetectReplayRunner") as runner_class,
            patch("frigate.video.ffmpeg.start_or_restart_ffmpeg"),
            patch("frigate.video.ffmpeg.LogPipe"),
        ):
            watchdog.stop_all_ffmpeg()
            watchdog.start_all_ffmpeg()

        runner_class.assert_not_called()
        # recording is separate from replay and still starts
        assert len(watchdog.ffmpeg_other_processes) == 1

    def test_detect_events_do_not_reach_a_runner_that_is_stopping(self):
        watchdog = build_watchdog({"mode": "replay"})
        old = stuck_runner()
        watchdog.replay_runner = old
        watchdog.stop_replay()

        watchdog._on_detect_event("front_door", True, 500.0)

        old.trigger.assert_not_called()


class TestRunnerStopsBeforeSideEffects(unittest.TestCase):
    def test_a_stopped_runner_writes_nothing_to_shared_memory(self):
        runner, frame_queue, frame_size = build_runner([])
        ledger = [5.0] * 10
        runner.frame_manager.frame_times.return_value = ledger
        runner._local_stop.set()

        assert runner._feed(bytes(frame_size), 1000.0) is False

        runner.frame_manager.write.assert_not_called()
        # the slot may belong to a newer owner by now, so it is left alone
        assert ledger[0] == 5.0
        assert frame_queue.empty()


class TestRunnerStop(unittest.TestCase):
    def test_stop_kills_ffmpeg_that_ignores_terminate(self):
        runner, _, _ = build_runner([])
        release = threading.Event()
        self.addCleanup(release.set)
        process = MagicMock()
        # the thread is blocked reading the pipe until the process is gone
        process.kill.side_effect = release.set
        runner._process = process
        runner.run = lambda: release.wait(30)
        runner.start()

        with patch("frigate.video.detect_replay.STOP_GRACE_S", 0.2, create=True):
            stopped = runner.stop()

        process.terminate.assert_called_once()
        process.kill.assert_called_once()
        assert stopped is True
        assert not runner.is_alive()

    def test_stop_reports_a_thread_that_will_not_end(self):
        runner, _, _ = build_runner([])
        release = threading.Event()
        self.addCleanup(release.set)
        runner._process = MagicMock()
        runner.run = lambda: release.wait(30)
        runner.start()

        with (
            patch("frigate.video.detect_replay.STOP_GRACE_S", 0.1, create=True),
            self.assertLogs("frigate.video.detect_replay", level="WARNING"),
        ):
            stopped = runner.stop()

        assert stopped is False
        assert runner.is_alive()

    def test_stop_ends_a_replay_blocked_on_a_silent_ffmpeg(self):
        now = time.time()
        runner, _, _ = build_runner([segment("/c/a.mp4", now - 5, 10.0)])
        started = threading.Event()
        processes: list[sp.Popen] = []

        def start(cmd):
            # one process that never writes, like ffmpeg stuck on a hardware call
            process = sp.Popen(
                ["sh", "-c", "exec sleep 60"],
                stdin=sp.DEVNULL,
                stdout=sp.PIPE,
                stderr=sp.PIPE,
                start_new_session=True,
            )
            processes.append(process)
            self.addCleanup(process.kill)
            started.set()
            return process

        runner._start_process = start
        runner.trigger(now)
        runner.start()
        assert started.wait(10)

        runner.stop()

        assert not runner.is_alive()
        assert processes[0].poll() is not None
        assert runner._process is None
        assert runner.active is False

    def test_stopping_twice_is_harmless(self):
        runner, _, _ = build_runner([])
        runner.start()

        runner.stop()
        runner.stop()

        assert not runner.is_alive()


class TestReap(unittest.TestCase):
    def test_a_clean_exit_is_reported_and_nothing_is_signalled(self):
        process = MagicMock()
        process.wait.return_value = 0
        process.poll.return_value = 0
        process.returncode = 0

        assert DetectReplayRunner._reap(process, ended=True) == 0

        process.terminate.assert_not_called()
        process.kill.assert_not_called()
        process.stdout.close.assert_called_once()

    def test_ffmpeg_that_will_not_finish_by_itself_is_terminated(self):
        process = MagicMock()
        process.poll.return_value = None
        process.wait.side_effect = [sp.TimeoutExpired("ffmpeg", 10), 0]
        process.returncode = -15

        assert DetectReplayRunner._reap(process, ended=True) == -15

        process.terminate.assert_called_once()
        process.kill.assert_not_called()

    def test_ffmpeg_that_ignores_terminate_is_killed(self):
        process = MagicMock()
        process.poll.return_value = None
        process.wait.side_effect = [sp.TimeoutExpired("ffmpeg", 5), 0]
        process.returncode = -9

        assert DetectReplayRunner._reap(process, ended=False) == -9

        process.terminate.assert_called_once()
        process.kill.assert_called_once()
        process.stdout.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()

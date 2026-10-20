"""Tests for carrying a replay session across a restart of the replay runner.

A session is not over until its tail has been replayed. The tail feeds the
tracker the frames that end its objects, so a runner that replaces one that
died or was stopped must carry on with it.
"""

import unittest
from unittest.mock import MagicMock, patch

from frigate.test.test_camera_watchdog_replay import build_watchdog
from frigate.test.test_detect_replay import build_runner, fake_process, segment
from frigate.video.detect_replay import ReplaySession

INF = float("inf")
# detection is still wanted
OPEN = ReplaySession(990.0, INF, True, ((990.0, INF),))
# detection was turned off at 1002 and its tail has not been replayed
RELEASED = ReplaySession(990.0, 1006.0, False, ((990.0, 1002.0),))


def ended_runner(session: ReplaySession | None) -> MagicMock:
    runner = MagicMock()
    runner.is_alive.return_value = False
    runner.progress.return_value = (1010.0, 1009.5)
    runner.session.return_value = session
    return runner


class TestRunnerSession(unittest.TestCase):
    def test_a_runner_with_nothing_pending_has_no_session(self):
        runner, _, _ = build_runner([])

        assert runner.session() is None

    def test_a_session_still_wanted_is_reported(self):
        runner, _, _ = build_runner([])
        runner.trigger(1000.0)

        session = runner.session()

        start = 1000.0 - runner.pre_roll
        assert session == ReplaySession(start, INF, True, ((start, INF),))

    def test_a_released_session_is_reported_until_its_tail_is_replayed(self):
        runner, _, _ = build_runner([])
        runner.tail = 4.0
        runner.trigger(1000.0)
        runner.release(1002.0)

        session = runner.session()

        start = 1000.0 - runner.pre_roll
        assert session == ReplaySession(start, 1006.0, False, ((start, 1002.0),))

        runner._cursor = 1006.0

        assert runner.session() is None

    def test_resume_restores_what_is_wanted_and_what_is_owed(self):
        runner, _, _ = build_runner([])

        runner.resume(RELEASED)

        assert runner._pending()
        assert runner._range_start == 990.0
        # frames inside the window are detection frames, the tail is not
        assert runner._windows.contains(995.0)
        assert not runner._windows.contains(1004.0)

    def test_a_replacement_replays_the_tail_the_old_runner_still_owed(self):
        old, _, _ = build_runner([segment("/c/a.mp4", 1000.0)])
        old.tail = 4.0
        old.trigger(1001.0)
        old.release(1002.0)
        session = old.session()

        new, frame_queue, frame_size = build_runner([segment("/c/a.mp4", 1000.0)])

        def replay() -> list[float]:
            process = fake_process(frame_size, times=["0", "0.5", "3", "5", "7"])

            with (
                patch.object(new, "_start_process", return_value=process),
                patch("frigate.video.detect_replay.time") as clock,
            ):
                clock.time.return_value = 1100.0
                new._replay_pending()

            fed = []

            while not frame_queue.empty():
                fed.append(frame_queue.get()[1])

            return fed

        # without the session there is nothing to do, and the tracker never
        # sees the frames that end its objects
        assert not new._pending()
        assert replay() == []

        new.resume(session)

        assert new._pending()
        # up to the end of the tail at 1006, and no further
        assert replay() == [1000.0, 1000.5, 1003.0, 1005.0]


class TestWatchdogCarriesTheSession(unittest.TestCase):
    def _watchdog(self, enabled: bool = True):
        watchdog = build_watchdog({"mode": "replay"})
        watchdog.config.detect.enabled = enabled
        return watchdog

    def test_a_runner_that_died_hands_its_session_to_the_replacement(self):
        watchdog = self._watchdog()
        watchdog.replay_runner = ended_runner(RELEASED)

        with patch("frigate.video.ffmpeg.DetectReplayRunner") as runner_class:
            watchdog.start_replay()

        runner_class.return_value.resume.assert_called_once_with(RELEASED)

    def test_a_runner_stopped_on_purpose_hands_its_session_on_too(self):
        watchdog = self._watchdog()
        watchdog.replay_runner = ended_runner(RELEASED)
        watchdog.stop_replay()

        with patch("frigate.video.ffmpeg.DetectReplayRunner") as runner_class:
            watchdog.start_replay()

        runner_class.return_value.resume.assert_called_once_with(RELEASED)

    def test_the_session_is_resumed_before_the_thread_starts(self):
        watchdog = self._watchdog()
        watchdog.replay_runner = ended_runner(RELEASED)

        with patch("frigate.video.ffmpeg.DetectReplayRunner") as runner_class:
            watchdog.start_replay()

        names = [call[0] for call in runner_class.return_value.mock_calls]

        assert names.index("resume") < names.index("start")

    def test_nothing_is_resumed_when_no_session_was_pending(self):
        watchdog = self._watchdog()
        watchdog.replay_runner = ended_runner(None)

        with patch("frigate.video.ffmpeg.DetectReplayRunner") as runner_class:
            watchdog.start_replay()

        runner_class.return_value.resume.assert_not_called()

    def test_a_session_is_handed_over_once(self):
        watchdog = self._watchdog()
        watchdog.replay_runner = ended_runner(RELEASED)

        with patch("frigate.video.ffmpeg.DetectReplayRunner") as runner_class:
            watchdog.start_replay()

            # the replacement dies too, with nothing pending this time
            runner_class.return_value.is_alive.return_value = False
            runner_class.return_value.progress.return_value = (1010.0, 1009.5)
            runner_class.return_value.session.return_value = None
            watchdog.start_replay()

        runner_class.return_value.resume.assert_called_once_with(RELEASED)

    def test_a_release_that_arrived_during_the_restart_is_applied(self):
        watchdog = self._watchdog(enabled=False)
        watchdog.config.detect.mark_event(1002.0)
        watchdog.replay_runner = ended_runner(OPEN)

        with patch("frigate.video.ffmpeg.DetectReplayRunner") as runner_class:
            watchdog.start_replay()

        runner_class.return_value.release.assert_called_once_with(1002.0)

    def test_a_session_still_wanted_is_not_released_while_detection_is_on(self):
        watchdog = self._watchdog(enabled=True)
        watchdog.replay_runner = ended_runner(OPEN)

        with patch("frigate.video.ffmpeg.DetectReplayRunner") as runner_class:
            watchdog.start_replay()

        runner_class.return_value.release.assert_not_called()

    def test_a_session_that_was_already_released_is_not_released_again(self):
        watchdog = self._watchdog(enabled=False)
        watchdog.config.detect.mark_event(1002.0)
        watchdog.replay_runner = ended_runner(RELEASED)

        with patch("frigate.video.ffmpeg.DetectReplayRunner") as runner_class:
            watchdog.start_replay()

        runner_class.return_value.release.assert_not_called()


if __name__ == "__main__":
    unittest.main()

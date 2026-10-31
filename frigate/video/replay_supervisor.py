"""Supervision of the detect replay runner of one camera."""

import logging
import time
from collections.abc import Callable
from typing import NamedTuple

from frigate.config import CameraConfig
from frigate.video.detect_replay import DetectReplayRunner, ReplaySession


class ReplayCheck(NamedTuple):
    """The outcome of one supervision pass."""

    # detect status to report: online, idle or offline
    status: str
    # whether a start was attempted, so the caller paces the next one
    restarted: bool
    # whether a live runner was there to judge, so record_offline applied
    running: bool


class ReplaySupervisor:
    """Owns the replay runner of a camera and keeps it running.

    It is called only from the camera watchdog thread: the watchdog loop, the
    config update callback it drives, and start and stop. Its state is
    confined to that thread, so it takes no lock. The runner has its own lock
    for what the runner thread shares.

    The supervisor never reports a status. The watchdog sends the detect
    status and sets the replay offline flag from what check returns.
    """

    def __init__(
        self,
        config: CameraConfig,
        runner_factory: Callable[[float, float], DetectReplayRunner],
        logger: logging.Logger,
    ) -> None:
        """The factory builds a runner from where the last one got to."""
        self.config = config
        self.logger = logger
        self._runner_factory = runner_factory
        self.runner: DetectReplayRunner | None = None
        # (cursor, last fed capture time) of the last runner, so the next one
        # never feeds the tracker a frame it already has
        self._progress: tuple[float, float] = (0.0, 0.0)
        # the session the last runner left unfinished, for its replacement
        self._session: ReplaySession | None = None
        # A runner that was asked to stop whose thread is still alive. It can
        # still write frames, so no replacement starts until it has ended
        self._retiring: DetectReplayRunner | None = None
        # whether the current outage of the replay thread was already reported
        self._down_logged = False

    def check(self, now: float, can_restart: bool, record_offline: bool) -> ReplayCheck:
        """Supervise the replay thread.

        A thread with nothing to do is idle and healthy. A dead one is reported
        once per outage, not on every pass of the watchdog, and is started again
        when the restart pacing allows. An attempt counts as a restart even when
        start declines because the previous thread has not ended, so the
        retries stay paced.

        With no detect stream to lose, a dead camera shows up as a recording
        stream that is down, which the caller passes as record_offline.
        """
        runner = self.runner

        if runner is None or not runner.is_alive():
            if not self._down_logged:
                self._down_logged = True
                self.logger.error(
                    "Replay thread is not running for %s", self.config.name
                )

            if not can_restart:
                return ReplayCheck("idle", False, False)

            self.start()
            return ReplayCheck("idle", True, False)

        self._down_logged = False

        if record_offline:
            return ReplayCheck("offline", False, True)

        return ReplayCheck("online" if runner.active else "idle", False, True)

    def start(self) -> bool:
        """Start the thread that replays recorded footage for detection.

        Returns False when the previous runner's thread is still alive. Two
        runners must never write the same frame ring, and a stop that timed out
        is not proof that the old thread ended, so the replacement waits.
        """
        previous = self._retiring if self._retiring is not None else self.runner

        if previous is not None:
            if previous.is_alive():
                self.logger.warning(
                    "The previous replay thread for %s has not ended, not "
                    "starting another",
                    self.config.name,
                )
                return False

            # a runner that ended is replaced from where it got to
            self._progress = previous.progress()
            self._session = previous.session()
            self._retiring = None

        cursor, last_fed = self._progress
        self.runner = self._runner_factory(cursor, last_fed)
        # a session that was not over when the last runner ended carries on, so
        # a restart does not lose the frames that end the tracker's objects
        session, self._session = self._session, None

        if session is not None:
            self.runner.resume(session)

        self.runner.start()

        if session is not None and session.demand and not self.config.detect.enabled:
            # the release arrived while no runner was there to take it
            released_at = self.config.detect.event_time
            self.runner.release(released_at if released_at is not None else time.time())

        # detection already enabled, such as from the config or a toggle that
        # arrived while the runner was being restarted, is a standing trigger
        if self.config.detect.enabled:
            event_time = self.config.detect.event_time
            self.runner.trigger(event_time if event_time is not None else time.time())

        return True

    def stop(self) -> None:
        """Stop the replay runner and keep what it had not finished.

        The runner is dropped so detect events no longer reach it. When its
        thread ended in time, its progress and unfinished session carry over
        to the next runner. When it did not, its state is not read and the
        runner is held as retiring so the replacement waits (see start).
        """
        runner = self.runner

        if runner is None:
            return

        runner.stop()
        # detect events no longer reach it
        self.runner = None

        if runner.is_alive():
            # a timeout is not proof that the thread ended. It may still write
            # frames and change its progress, so its progress is not read and
            # its replacement waits (see start)
            self.logger.warning(
                "Replay thread for %s did not stop in time", self.config.name
            )
            self._retiring = runner
        else:
            self._progress = runner.progress()
            self._session = runner.session()

    def on_detect_event(self, enabled: bool, event_time: float | None) -> None:
        """Pass a detect toggle on to the runner.

        In replay mode the toggle is the hardware event. Every toggle arrives
        here in order, so one that flips back before the next watchdog tick is
        still replayed, and each carries the time the event was received
        rather than the time the watchdog got to it.
        """
        if self.runner is None:
            return

        timestamp = event_time if event_time is not None else time.time()

        if enabled:
            self.runner.trigger(timestamp)
        else:
            self.runner.release(timestamp)

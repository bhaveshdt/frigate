"""Tests for dynamic camera config updates recreating ffmpeg commands."""

import pickle
import unittest
from unittest.mock import patch

from frigate.config import CameraConfig, FrigateConfig
from frigate.config.camera.updater import (
    CameraConfigUpdateEnum,
    CameraConfigUpdateSubscriber,
)
from frigate.const import SUB_CACHE_TAG


def _build_scene_frigate_config(scene: str | None) -> FrigateConfig:
    detect = {"height": 1080, "width": 1920, "fps": 5}
    if scene is not None:
        detect["scene"] = scene
    return FrigateConfig(
        **{
            "mqtt": {"host": "mqtt"},
            "models": [
                {"devices": ["cpu"]},
                {"scene": "outdoor", "devices": ["openvino:CPU"]},
            ],
            "cameras": {
                "front_door": {
                    "ffmpeg": {
                        "inputs": [
                            {"path": "rtsp://10.0.0.1:554/video", "roles": ["detect"]}
                        ]
                    },
                    "detect": detect,
                }
            },
        }
    )


def _build_camera_config(sub_enabled: bool) -> CameraConfig:
    config = FrigateConfig(
        **{
            "mqtt": {"host": "mqtt"},
            "cameras": {
                "front_door": {
                    "ffmpeg": {
                        "inputs": [
                            {
                                "path": "rtsp://10.0.0.1:554/video",
                                "roles": ["detect", "record"],
                            },
                            {
                                "path": "rtsp://10.0.0.1:554/video2",
                                "roles": ["record_sub"],
                            },
                        ]
                    },
                    "record": {"enabled": True, "sub": {"enabled": sub_enabled}},
                }
            },
        }
    )
    return config.cameras["front_door"]


def _has_sub_output(camera_config: CameraConfig) -> bool:
    return any(
        SUB_CACHE_TAG in part for c in camera_config.ffmpeg_cmds for part in c["cmd"]
    )


class TestRecordUpdateRecreatesFfmpegCmds(unittest.TestCase):
    def setUp(self):
        # avoid binding a real ZMQ socket; updates are fed directly through
        # the mocked subscriber below
        patcher = patch("frigate.config.camera.updater.ConfigSubscriber")
        patcher.start()
        self.addCleanup(patcher.stop)

    def _push_record_update(
        self, subscriber: CameraConfigUpdateSubscriber, record_config
    ) -> None:
        subscriber.subscriber.check_for_update.side_effect = [
            ("config/cameras/front_door/record", record_config),
            (None, None),
        ]
        subscriber.check_for_updates()

    def test_enabling_sub_recreates_ffmpeg_cmds(self):
        camera_config = _build_camera_config(sub_enabled=False)
        subscriber = CameraConfigUpdateSubscriber(
            None, {"front_door": camera_config}, [CameraConfigUpdateEnum.record]
        )
        assert not _has_sub_output(camera_config)

        self._push_record_update(
            subscriber, _build_camera_config(sub_enabled=True).record
        )

        assert _has_sub_output(camera_config)

    def test_disabling_sub_recreates_ffmpeg_cmds(self):
        camera_config = _build_camera_config(sub_enabled=True)
        subscriber = CameraConfigUpdateSubscriber(
            None, {"front_door": camera_config}, [CameraConfigUpdateEnum.record]
        )
        assert _has_sub_output(camera_config)

        self._push_record_update(
            subscriber, _build_camera_config(sub_enabled=False).record
        )

        assert not _has_sub_output(camera_config)

    @patch("frigate.detectors.detector_config.load_labels")
    def test_removed_camera_readded_without_scene_gets_fresh_model(self, mock_labels):
        mock_labels.return_value = {}
        config = _build_scene_frigate_config("outdoor")
        subscriber = CameraConfigUpdateSubscriber(
            config, {}, [CameraConfigUpdateEnum.add, CameraConfigUpdateEnum.remove]
        )
        assert config.model_for_camera("front_door").scene == "outdoor"

        subscriber.subscriber.check_for_update.side_effect = [
            ("config/cameras/front_door/remove", config.cameras["front_door"]),
            (None, None),
        ]
        subscriber.check_for_updates()

        # recreating the camera through the wizard leaves the scene unset,
        # so the removed camera's cached model must not carry over
        readded = _build_scene_frigate_config(None).cameras["front_door"]
        subscriber.subscriber.check_for_update.side_effect = [
            ("config/cameras/front_door/add", readded),
            (None, None),
        ]
        subscriber.check_for_updates()

        assert config.model_for_camera("front_door").scene == "default"

    def test_unchanged_record_update_keeps_existing_cmds(self):
        camera_config = _build_camera_config(sub_enabled=False)
        subscriber = CameraConfigUpdateSubscriber(
            None, {"front_door": camera_config}, [CameraConfigUpdateEnum.record]
        )
        cmds_before = camera_config.ffmpeg_cmds

        # neither enabled_in_config nor sub.enabled changed, so the
        # commands should not be rebuilt
        self._push_record_update(
            subscriber, _build_camera_config(sub_enabled=False).record
        )

        assert camera_config.ffmpeg_cmds is cmds_before


class TestDetectUpdateEvents(unittest.TestCase):
    def setUp(self):
        patcher = patch("frigate.config.camera.updater.ConfigSubscriber")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.events: list[tuple[str, bool, float | None]] = []
        self.camera_config = _build_camera_config(sub_enabled=False)
        self.subscriber = CameraConfigUpdateSubscriber(
            None,
            {"front_door": self.camera_config},
            [CameraConfigUpdateEnum.detect, CameraConfigUpdateEnum.record],
            on_detect_event=lambda *event: self.events.append(event),
        )

    @staticmethod
    def _detect(enabled: bool, event_time: float | None = None):
        detect = _build_camera_config(sub_enabled=False).detect
        detect.enabled = enabled

        if event_time is not None:
            detect.mark_event(event_time)

        return detect

    def _push(self, *updates) -> None:
        self.subscriber.subscriber.check_for_update.side_effect = [
            *updates,
            (None, None),
        ]
        self.subscriber.check_for_updates()

    def test_every_detect_update_is_reported_with_its_event_time(self):
        self._push(("config/cameras/front_door/detect", self._detect(True, 100.0)))

        assert self.events == [("front_door", True, 100.0)]
        assert self.camera_config.detect.enabled is True

    def test_a_toggle_that_flips_back_within_one_check_is_not_lost(self):
        # polling the config after both arrived would see no change at all
        self._push(
            ("config/cameras/front_door/detect", self._detect(True, 100.0)),
            ("config/cameras/front_door/detect", self._detect(False, 100.4)),
        )

        assert self.events == [
            ("front_door", True, 100.0),
            ("front_door", False, 100.4),
        ]

    def test_an_update_without_an_event_time_reports_none(self):
        self._push(("config/cameras/front_door/detect", self._detect(False)))

        assert self.events == [("front_door", False, None)]

    def test_other_update_types_are_not_detect_events(self):
        self._push(
            (
                "config/cameras/front_door/record",
                _build_camera_config(sub_enabled=False).record,
            )
        )

        assert self.events == []

    def test_the_event_time_survives_the_trip_between_processes(self):
        # config updates are published with send_pyobj, which pickles them
        detect = self._detect(True, 100.0)

        assert pickle.loads(pickle.dumps(detect)).event_time == 100.0

    def test_the_event_time_is_not_part_of_the_config(self):
        detect = self._detect(True, 100.0)

        assert "event_time" not in detect.model_dump()


def _build_mode_camera(
    mode: str, record_enabled: bool = True, continuous_days: float = 0
) -> CameraConfig:
    config = FrigateConfig(
        **{
            "mqtt": {"host": "mqtt"},
            "cameras": {
                "front_door": {
                    "ffmpeg": {
                        "inputs": [
                            {
                                "path": "rtsp://10.0.0.1:554/video",
                                "roles": ["detect", "record"],
                            }
                        ]
                    },
                    "detect": {"width": 960, "height": 432, "mode": mode},
                    "record": {
                        "enabled": record_enabled,
                        "continuous": {"days": continuous_days},
                    },
                }
            },
        }
    )
    return config.cameras["front_door"]


class TestDetectModeUpdates(unittest.TestCase):
    def setUp(self):
        patcher = patch("frigate.config.camera.updater.ConfigSubscriber")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.events: list[tuple[str, bool, float | None]] = []

    def _subscribe(self, camera_config: CameraConfig) -> CameraConfigUpdateSubscriber:
        return CameraConfigUpdateSubscriber(
            None,
            {"front_door": camera_config},
            [CameraConfigUpdateEnum.detect],
            on_detect_event=lambda *event: self.events.append(event),
        )

    @staticmethod
    def _push(subscriber: CameraConfigUpdateSubscriber, detect) -> None:
        subscriber.subscriber.check_for_update.side_effect = [
            ("config/cameras/front_door/detect", detect),
            (None, None),
        ]
        subscriber.check_for_updates()

    def test_switching_to_replay_removes_the_detect_output(self):
        camera_config = _build_mode_camera("continuous")
        assert "detect" in camera_config.ffmpeg_cmds[0]["roles"]
        detect = _build_mode_camera("replay").detect

        self._push(self._subscribe(camera_config), detect)

        assert camera_config.detect_replay is True
        assert "detect" not in camera_config.ffmpeg_cmds[0]["roles"]
        assert len(self.events) == 1

    def test_switching_back_to_continuous_restores_the_detect_output(self):
        camera_config = _build_mode_camera("replay")
        assert "detect" not in camera_config.ffmpeg_cmds[0]["roles"]

        self._push(
            self._subscribe(camera_config), _build_mode_camera("continuous").detect
        )

        assert camera_config.detect_replay is False
        assert "detect" in camera_config.ffmpeg_cmds[0]["roles"]

    def test_replay_is_refused_when_the_camera_does_not_record(self):
        # a runtime update skips the camera validator, so the updater has to
        # hold the same rule or replay would be left with no footage
        camera_config = _build_mode_camera("continuous", record_enabled=False)
        subscriber = self._subscribe(camera_config)

        with self.assertLogs("frigate.config.camera.updater", level="ERROR") as logs:
            self._push(subscriber, _build_mode_camera("replay").detect)

        assert "requires recording" in "\n".join(logs.output)
        assert camera_config.detect_replay is False
        assert "detect" in camera_config.ffmpeg_cmds[0]["roles"]
        # the watchdog is not told about an update that was not applied
        assert self.events == []

    def test_replay_is_refused_when_continuous_retention_is_configured(self):
        # the maintainer would quietly trim the idle footage it promises to keep
        camera_config = _build_mode_camera("continuous", continuous_days=7)
        subscriber = self._subscribe(camera_config)

        with self.assertLogs("frigate.config.camera.updater", level="ERROR") as logs:
            self._push(subscriber, _build_mode_camera("replay").detect)

        assert "continuous recording retention" in "\n".join(logs.output)
        assert camera_config.detect_replay is False
        assert self.events == []

    def test_an_update_that_keeps_replay_is_not_checked_again(self):
        # toggles from hardware events must never be refused for the mode
        camera_config = _build_mode_camera("replay")
        detect = _build_mode_camera("replay").detect
        detect.enabled = True
        detect.mark_event(100.0)

        self._push(self._subscribe(camera_config), detect)

        assert self.events == [("front_door", True, 100.0)]

    def test_replay_blocker_matches_what_the_validator_enforces(self):
        assert _build_mode_camera("replay").replay_blocker() is None
        assert (
            "requires recording"
            in _build_mode_camera("continuous", record_enabled=False).replay_blocker()
        )
        # an explicit value overrides the config, for runtime checks
        assert (
            _build_mode_camera("continuous").replay_blocker(record_enabled=False)
            is not None
        )

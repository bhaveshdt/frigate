"""Convenience classes for updating configurations dynamically."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from frigate.comms.config_updater import ConfigPublisher, ConfigSubscriber
from frigate.config import CameraConfig, FrigateConfig
from frigate.config.camera.detect import DetectModeEnum

logger = logging.getLogger(__name__)


class CameraConfigUpdateEnum(str, Enum):
    """Supported camera config update types."""

    add = "add"  # for adding a camera
    audio = "audio"
    audio_transcription = "audio_transcription"
    autotracking = "autotracking"  # ptz autotracking only, without an onvif reinit
    birdseye = "birdseye"
    detect = "detect"
    enabled = "enabled"
    ffmpeg = "ffmpeg"
    live = "live"
    motion = "motion"  # includes motion and motion masks
    mqtt = "mqtt"
    notifications = "notifications"
    objects = "objects"
    object_genai = "object_genai"
    onvif = "onvif"
    record = "record"
    refresh = "refresh"  # signals the camera maintainer to recycle the camera process
    remove = "remove"  # for removing a camera
    review = "review"
    review_genai = "review_genai"
    semantic_search = "semantic_search"  # for semantic search triggers
    face_recognition = "face_recognition"
    lpr = "lpr"
    snapshots = "snapshots"
    timestamp_style = "timestamp_style"
    ui = "ui"
    zones = "zones"


@dataclass
class CameraConfigUpdateTopic:
    update_type: CameraConfigUpdateEnum
    camera: str

    @property
    def topic(self) -> str:
        return f"config/cameras/{self.camera}/{self.update_type.name}"


class CameraConfigUpdatePublisher:
    def __init__(self):
        self.publisher = ConfigPublisher()

    def publish_update(self, topic: CameraConfigUpdateTopic, config: Any) -> None:
        self.publisher.publish(topic.topic, config)

    def stop(self) -> None:
        self.publisher.stop()


class CameraConfigUpdateSubscriber:
    def __init__(
        self,
        config: FrigateConfig | None,
        camera_configs: dict[str, CameraConfig],
        topics: list[CameraConfigUpdateEnum],
        on_detect_event: Callable[[str, bool, float | None], None] | None = None,
    ):
        self.config = config
        self.camera_configs = camera_configs
        self.topics = topics
        # called once for every detect update, in the order they were published,
        # with the camera, whether detection is now enabled, and the time the
        # toggle was received. Polling the config instead would only see the
        # state at the moment of the poll, so a toggle that flips back within
        # one poll would be lost and the time would be that of the poll.
        self.on_detect_event = on_detect_event

        base_topic = "config/cameras"

        # global subscribers must hear every camera; only narrow per-camera workers
        is_global_subscriber = (
            CameraConfigUpdateEnum.add in self.topics
            or CameraConfigUpdateEnum.remove in self.topics
        )
        if not is_global_subscriber and len(self.camera_configs) == 1:
            base_topic += f"/{list(self.camera_configs.keys())[0]}"

        self.subscriber = ConfigSubscriber(
            base_topic,
            exact=False,
        )

    def __update_config(
        self, camera: str, update_type: CameraConfigUpdateEnum, updated_config: Any
    ) -> None:
        if update_type == CameraConfigUpdateEnum.add:
            shared = self.config.cameras.setdefault(camera, updated_config)
            self.camera_configs[camera] = shared
            return
        elif update_type == CameraConfigUpdateEnum.remove:
            self.config.cameras.pop(camera, None)
            self.config.drop_camera_model(camera)
            self.camera_configs.pop(camera, None)
            return

        config = self.camera_configs.get(camera)

        if not config:
            return

        if update_type == CameraConfigUpdateEnum.audio:
            config.audio = updated_config
        elif update_type == CameraConfigUpdateEnum.ffmpeg:
            config.ffmpeg = updated_config
            config.recreate_ffmpeg_cmds()
        elif update_type == CameraConfigUpdateEnum.audio_transcription:
            config.audio_transcription = updated_config
        elif update_type == CameraConfigUpdateEnum.birdseye:
            config.birdseye = updated_config
        elif update_type == CameraConfigUpdateEnum.detect:
            old_mode = config.detect.mode
            new_mode = getattr(updated_config, "mode", old_mode)

            # a runtime update skips the camera validator, so a switch to
            # replay is checked here against what is actually being recorded
            if new_mode != old_mode and new_mode == DetectModeEnum.replay:
                reason = config.replay_blocker(bool(config.record.enabled_in_config))

                if reason is not None:
                    logger.error(
                        "Not switching %s to detect mode replay: %s", camera, reason
                    )
                    return

            config.detect = updated_config
            # the detect output of the ffmpeg commands is gated on the mode
            if old_mode != new_mode:
                config.recreate_ffmpeg_cmds()

            if self.on_detect_event is not None:
                self.on_detect_event(
                    camera,
                    updated_config.enabled,
                    getattr(updated_config, "event_time", None),
                )
        elif update_type == CameraConfigUpdateEnum.enabled:
            config.enabled = updated_config
        elif update_type == CameraConfigUpdateEnum.object_genai:
            config.objects.genai = updated_config
        elif update_type == CameraConfigUpdateEnum.live:
            config.live = updated_config
        elif update_type == CameraConfigUpdateEnum.motion:
            config.motion = updated_config
        elif update_type == CameraConfigUpdateEnum.notifications:
            config.notifications = updated_config
        elif update_type == CameraConfigUpdateEnum.objects:
            config.objects = updated_config
        elif update_type == CameraConfigUpdateEnum.record:
            old_enabled_in_config = config.record.enabled_in_config
            old_sub_enabled = config.record.sub.enabled
            config.record = updated_config
            # the record and record_sub ffmpeg outputs are gated on these
            if (
                old_enabled_in_config != updated_config.enabled_in_config
                or old_sub_enabled != updated_config.sub.enabled
            ):
                config.recreate_ffmpeg_cmds()
        elif update_type == CameraConfigUpdateEnum.review:
            config.review = updated_config
        elif update_type == CameraConfigUpdateEnum.review_genai:
            config.review.genai = updated_config
        elif update_type == CameraConfigUpdateEnum.semantic_search:
            config.semantic_search = updated_config
        elif update_type == CameraConfigUpdateEnum.face_recognition:
            config.face_recognition = updated_config
        elif update_type == CameraConfigUpdateEnum.lpr:
            config.lpr = updated_config
        elif update_type == CameraConfigUpdateEnum.snapshots:
            config.snapshots = updated_config
        elif update_type == CameraConfigUpdateEnum.onvif:
            config.onvif = updated_config
        elif update_type == CameraConfigUpdateEnum.autotracking:
            config.onvif.autotracking = updated_config
        elif update_type == CameraConfigUpdateEnum.timestamp_style:
            config.timestamp_style = updated_config
        elif update_type == CameraConfigUpdateEnum.zones:
            config.zones = updated_config

    def check_for_updates(self) -> dict[str, list[str]]:
        updated_topics: dict[str, list[str]] = {}

        # get all updates available
        while True:
            update_topic, update_config = self.subscriber.check_for_update()

            if update_topic is None or update_config is None:
                break

            _, _, camera, raw_type = update_topic.split("/")
            update_type = CameraConfigUpdateEnum[raw_type]

            if update_type in self.topics:
                if update_type.name in updated_topics:
                    updated_topics[update_type.name].append(camera)
                else:
                    updated_topics[update_type.name] = [camera]

                self.__update_config(camera, update_type, update_config)

        return updated_topics

    def stop(self) -> None:
        self.subscriber.stop()

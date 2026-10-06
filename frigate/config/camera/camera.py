import os
import re
from enum import Enum

from pydantic import Field, PrivateAttr, model_validator

from frigate.const import (
    CACHE_DIR,
    CACHE_SEGMENT_FORMAT,
    LIBAVFORMAT_VERSION_MAJOR,
    REGEX_CAMERA_NAME,
    SUB_CACHE_TAG,
)
from frigate.ffmpeg_presets import (
    parse_preset_hardware_acceleration_decode,
    parse_preset_hardware_acceleration_scale,
    parse_preset_input,
    parse_preset_output_record,
)
from frigate.util.builtin import (
    escape_special_characters,
    generate_color_palette,
    get_ffmpeg_arg_list,
)

from ..base import FrigateBaseModel
from ..classification import (
    CameraAudioTranscriptionConfig,
    CameraFaceRecognitionConfig,
    CameraLicensePlateRecognitionConfig,
    CameraSemanticSearchConfig,
)
from .audio import AudioConfig
from .birdseye import BirdseyeCameraConfig
from .detect import DetectConfig, DetectModeEnum
from .ffmpeg import CameraFfmpegConfig, CameraInput, CameraRoleEnum
from .live import CameraLiveConfig
from .motion import MotionConfig
from .mqtt import CameraMqttConfig
from .notification import NotificationConfig
from .objects import ObjectConfig
from .onvif import OnvifConfig
from .profile import CameraProfileConfig
from .record import RecordConfig
from .review import ReviewConfig
from .snapshots import SnapshotsConfig
from .timestamp import TimestampStyleConfig
from .ui import CameraUiConfig
from .zone import ZoneConfig

__all__ = ["CameraConfig"]

# ffmpeg flags that only make sense on a live stream and break a recorded file.
# dump_extra re-inserts parameter sets that an mp4 already carries, which makes
# the hardware decoder reject every packet
STREAM_ONLY_ARGS = ("-bsf:v", "dump_extra")

REPLAY_PASSTHROUGH_ARGS = (
    ["-fps_mode", "passthrough"]
    if LIBAVFORMAT_VERSION_MAJOR >= 59
    else ["-vsync", "0"]
)


class CameraTypeEnum(str, Enum):
    generic = "generic"
    lpr = "lpr"


class CameraConfig(FrigateBaseModel):
    name: str | None = Field(
        None,
        title="Camera name",
        description="Camera name is required",
        pattern=REGEX_CAMERA_NAME,
    )

    friendly_name: str | None = Field(
        None,
        title="Friendly name",
        description="Camera friendly name used in the Frigate UI",
    )

    @model_validator(mode="before")
    @classmethod
    def handle_friendly_name(cls, values):
        if isinstance(values, dict) and "friendly_name" in values:
            pass
        return values

    enabled: bool = Field(default=True, title="Enabled", description="Enabled")

    # Options with global fallback
    audio: AudioConfig = Field(
        default_factory=AudioConfig,
        title="Audio detection",
        description="Settings for audio-based event detection for this camera.",
    )
    audio_transcription: CameraAudioTranscriptionConfig = Field(
        default_factory=CameraAudioTranscriptionConfig,
        title="Audio transcription",
        description="Settings for live and speech audio transcription used for events and live captions.",
    )
    birdseye: BirdseyeCameraConfig = Field(
        default_factory=BirdseyeCameraConfig,
        title="Birdseye",
        description="Settings for the Birdseye composite view that composes multiple camera feeds into a single layout.",
    )
    detect: DetectConfig = Field(
        default_factory=DetectConfig,
        title="Object Detection",
        description="Settings for the detection/detect role used to run object detection and initialize trackers.",
    )
    face_recognition: CameraFaceRecognitionConfig = Field(
        default_factory=CameraFaceRecognitionConfig,
        title="Face recognition",
        description="Settings for face detection and recognition for this camera.",
    )
    ffmpeg: CameraFfmpegConfig = Field(
        title="Streams (FFmpeg)",
        description="Camera stream inputs and FFmpeg options, including binary path, args, hwaccel, and per-role output args.",
    )
    live: CameraLiveConfig = Field(
        default_factory=CameraLiveConfig,
        title="Live playback",
        description="Settings used by the Web UI to control live stream selection, resolution and quality.",
    )
    lpr: CameraLicensePlateRecognitionConfig = Field(
        default_factory=CameraLicensePlateRecognitionConfig,
        title="License Plate Recognition",
        description="License plate recognition settings including detection thresholds, formatting, and known plates.",
    )
    motion: MotionConfig = Field(
        None,
        title="Motion detection",
        description="Default motion detection settings for this camera.",
    )
    objects: ObjectConfig = Field(
        default_factory=ObjectConfig,
        title="Objects",
        description="Object tracking defaults including which labels to track and per-object filters.",
    )
    record: RecordConfig = Field(
        default_factory=RecordConfig,
        title="Recording",
        description="Recording and retention settings for this camera.",
    )
    review: ReviewConfig = Field(
        default_factory=ReviewConfig,
        title="Review",
        description="Settings that control alerts, detections, and GenAI review summaries used by the UI and storage for this camera.",
    )
    semantic_search: CameraSemanticSearchConfig = Field(
        default_factory=CameraSemanticSearchConfig,
        title="Semantic Search",
        description="Settings for semantic search which builds and queries object embeddings to find similar items.",
    )
    snapshots: SnapshotsConfig = Field(
        default_factory=SnapshotsConfig,
        title="Snapshots",
        description="Settings for API-generated snapshots of tracked objects for this camera.",
    )
    timestamp_style: TimestampStyleConfig = Field(
        default_factory=TimestampStyleConfig,
        title="Timestamp style",
        description="Styling options for timestamps applied to snapshots and Debug view.",
    )

    # Options without global fallback
    best_image_timeout: int = Field(
        default=60,
        title="Best image timeout",
        description="How long to wait for the image with the highest confidence score.",
    )
    mqtt: CameraMqttConfig = Field(
        default_factory=CameraMqttConfig,
        title="MQTT",
        description="MQTT image publishing settings.",
    )
    notifications: NotificationConfig = Field(
        default_factory=NotificationConfig,
        title="Notifications",
        description="Settings to enable and control notifications for this camera.",
    )
    onvif: OnvifConfig = Field(
        default_factory=OnvifConfig,
        title="ONVIF",
        description="ONVIF connection and PTZ autotracking settings for this camera.",
    )
    type: CameraTypeEnum = Field(
        default=CameraTypeEnum.generic,
        title="Camera type",
        description="Camera Type",
    )
    ui: CameraUiConfig = Field(
        default_factory=CameraUiConfig,
        title="Camera UI",
        description="Display ordering and visibility for this camera in the UI. Ordering affects the default dashboard. For more granular control, use camera groups.",
    )
    webui_url: str | None = Field(
        None,
        title="Camera URL",
        description="URL to visit the camera directly from system page",
    )

    profiles: dict[str, CameraProfileConfig] = Field(
        default_factory=dict,
        title="Profiles",
        description="Named config profiles with partial overrides that can be activated at runtime.",
    )
    zones: dict[str, ZoneConfig] = Field(
        default_factory=dict,
        title="Zones",
        description="Zones allow you to define a specific area of the frame so you can determine whether or not an object is within a particular area.",
    )
    enabled_in_config: bool | None = Field(
        default=None,
        title="Original camera state",
        description="Keep track of original state of camera.",
    )

    _ffmpeg_cmds: list[dict[str, list[str]]] = PrivateAttr()

    def __init__(self, **config):
        # Set zone colors
        if "zones" in config:
            colors = generate_color_palette(len(config["zones"]))

            config["zones"] = {
                name: {**z, "color": color}
                for (name, z), color in zip(config["zones"].items(), colors)
            }

        # add roles to the input if there is only one
        if len(config["ffmpeg"]["inputs"]) == 1:
            existing_roles = config["ffmpeg"]["inputs"][0].get("roles", [])

            config["ffmpeg"]["inputs"][0]["roles"] = [
                "record",
                "detect",
            ]

            if "audio" in existing_roles:
                config["ffmpeg"]["inputs"][0]["roles"].append("audio")

            # kept so role validation can report the real problem rather than
            # claiming the role was never assigned
            if "record_sub" in existing_roles:
                config["ffmpeg"]["inputs"][0]["roles"].append("record_sub")

        super().__init__(**config)

    @model_validator(mode="after")
    def validate_detect_mode(self) -> "CameraConfig":
        if self.detect.mode == DetectModeEnum.replay:
            reason = self.replay_blocker()

            if reason is not None:
                raise ValueError(reason)

        return self

    def replay_blocker(self, record_enabled: bool | None = None) -> str | None:
        """Why detect mode replay cannot run on this camera, or None if it can.

        Replay analyzes recorded footage, so it needs recording and an input
        that records. It also cannot honor continuous or motion retention: the
        recording maintainer only keeps footage for those once detection has
        reported on it, and between hardware events nothing is analyzed, so
        that footage would be trimmed from the cache without a word. Refusing
        the combination is better than quietly recording less than configured.

        This is the one definition of the rule: the config validator applies
        it when a config is loaded, and the runtime updater applies it before a
        camera is switched to replay, which skips the validator.

        Args:
            record_enabled: Whether the camera records, when that differs from
                the config's own value (a runtime update is judged on what is
                actually being recorded).
        """
        if not (self.record.enabled if record_enabled is None else record_enabled):
            return "detect -> mode replay requires recording to be enabled because replay analyzes recorded footage"

        if not any(CameraRoleEnum.record in i.roles for i in self.ffmpeg.inputs):
            return "detect -> mode replay requires an input with the record role"

        retention = {
            "continuous": self.record.continuous.days,
            "motion": self.record.motion.days,
        }

        if self.record.sub.enabled:
            retention["sub stream continuous"] = self.record.sub.continuous.days
            retention["sub stream motion"] = self.record.sub.motion.days

        configured = [name for name, days in retention.items() if days > 0]

        if configured:
            return (
                f"detect -> mode replay cannot be used with {' or '.join(configured)} "
                "recording retention because footage between hardware events is "
                "never analyzed and would not be kept. Set those retention days "
                "to 0 and use alert and detection retention instead"
            )

        return None

    @property
    def detect_replay(self) -> bool:
        """Whether detection replays recorded footage instead of decoding a stream."""
        return self.detect.mode == DetectModeEnum.replay

    @property
    def frame_shape(self) -> tuple[int, int]:
        return self.detect.height, self.detect.width

    @property
    def frame_shape_yuv(self) -> tuple[int, int]:
        return self.detect.height * 3 // 2, self.detect.width

    @property
    def ffmpeg_cmds(self) -> list[dict[str, list[str]]]:
        return self._ffmpeg_cmds

    def get_formatted_name(self) -> str:
        """Return the friendly name if set, otherwise return a formatted version of the camera name."""
        if self.friendly_name:
            return self.friendly_name
        return self.name.replace("_", " ").title() if self.name else ""

    def create_ffmpeg_cmds(self):
        if "_ffmpeg_cmds" in self:
            return
        self._build_ffmpeg_cmds()

    def recreate_ffmpeg_cmds(self):
        """Force regeneration of ffmpeg commands from current config."""
        self._build_ffmpeg_cmds()

    def _build_ffmpeg_cmds(self):
        """Build ffmpeg commands from the current ffmpeg config."""
        ffmpeg_cmds = []
        for ffmpeg_input in self.ffmpeg.inputs:
            ffmpeg_cmd = self._get_ffmpeg_cmd(ffmpeg_input)
            if ffmpeg_cmd is None:
                continue

            # replay mode has no detect output, so the process is not a detect process
            roles = [
                role
                for role in ffmpeg_input.roles
                if not (self.detect_replay and role == CameraRoleEnum.detect)
            ]
            ffmpeg_cmds.append({"roles": roles, "cmd": ffmpeg_cmd})
        self._ffmpeg_cmds = ffmpeg_cmds

    def _get_hwaccel_args(self, ffmpeg_input: CameraInput) -> list[str]:
        """Hardware acceleration decode args for an input."""
        camera_arg = (
            self.ffmpeg.hwaccel_args if self.ffmpeg.hwaccel_args != "auto" else None
        )
        return get_ffmpeg_arg_list(
            parse_preset_hardware_acceleration_decode(
                ffmpeg_input.hwaccel_args,
                self.detect.fps,
                self.detect.width,
                self.detect.height,
                self.ffmpeg.gpu,
            )
            or ffmpeg_input.hwaccel_args
            or parse_preset_hardware_acceleration_decode(
                camera_arg,
                self.detect.fps,
                self.detect.width,
                self.detect.height,
                self.ffmpeg.gpu,
            )
            or camera_arg
            or []
        )

    def get_replay_ffmpeg_cmd(self, segment_path: str) -> list[str] | None:
        """Build the ffmpeg command that decodes a recorded main stream segment.

        Uses the same hardware decode and scale presets as the detect role, so
        frames come out at detect width and height as raw yuv420p on stdout.
        Stream input args are left out because they are for live streams.
        """
        ffmpeg_input = next(
            (i for i in self.ffmpeg.inputs if CameraRoleEnum.record in i.roles), None
        )

        if ffmpeg_input is None:
            return None

        scale_args = self._replay_filter_args(
            parse_preset_hardware_acceleration_scale(
                ffmpeg_input.hwaccel_args or self.ffmpeg.hwaccel_args,
                get_ffmpeg_arg_list(self.ffmpeg.output_args.detect),
                self.detect.fps,
                self.detect.width,
                self.detect.height,
            ),
            self.detect.fps,
        )
        global_args = self._replay_global_args(
            get_ffmpeg_arg_list(ffmpeg_input.global_args or self.ffmpeg.global_args)
        )
        hwaccel_args = self._strip_stream_only_args(
            self._get_hwaccel_args(ffmpeg_input)
        )

        cmd = (
            [self.ffmpeg.ffmpeg_path]
            + global_args
            + hwaccel_args
            + ["-i", segment_path]
            + scale_args
            + ["pipe:"]
        )

        return [part for part in cmd if part != ""]

    @staticmethod
    def _strip_stream_only_args(args: list[str]) -> list[str]:
        """Remove args that are only valid when reading a live stream."""
        stripped: list[str] = []
        index = 0

        while index < len(args):
            if tuple(args[index : index + 2]) == STREAM_ONLY_ARGS:
                index += 2
                continue

            stripped.append(args[index])
            index += 1

        return stripped

    @staticmethod
    def _replay_global_args(args: list[str]) -> list[str]:
        """Log at info level with a level tag on every line.

        The replay reads the capture time of each frame from showinfo, which
        logs at info level. The tag lets the reader tell those lines apart from
        warnings and errors.
        """
        kept: list[str] = []
        index = 0

        while index < len(args):
            if args[index] in ("-loglevel", "-v"):
                index += 2
                continue

            kept.append(args[index])
            index += 1

        return kept + ["-loglevel", "level+info"]

    @staticmethod
    def _replay_filter_args(scale_args: list[str], fps: int | float) -> list[str]:
        """Pick frames by their real capture time instead of resampling them.

        The cache stamps frames with their arrival time, so spacing is uneven
        and an fps filter would duplicate frames to fill gaps, giving copies a
        made up time. Instead the first frame in each 1/fps window of the
        segment is kept and every other frame is dropped before any scaling or
        download. showinfo then logs the untouched presentation time of each
        kept frame, which is read back as its timestamp. It must come before
        the hardware scaler, since vpp_qsv rewrites the time base to the frame
        rate and rounds the time of every frame to it. After that the time is
        renumbered, because the scaler's rounding can give two kept frames the
        same time, which the muxer rejects as non monotonic.
        """
        args = list(scale_args)

        # an output frame rate would make ffmpeg duplicate and drop frames
        while "-r" in args:
            index = args.index("-r")
            del args[index : index + 2]

        select = (
            f"select=isnan(prev_selected_t)+"
            f"gt(floor(t*{fps})\\,floor(prev_selected_t*{fps}))"
        )
        prefix = f"{select},showinfo,setpts=N/TB"

        if "-vf" not in args:
            return REPLAY_PASSTHROUGH_ARGS + ["-vf", prefix] + args

        index = args.index("-vf") + 1
        video_filter = re.sub(r"framerate=[\d.]+:", "", args[index])
        video_filter = re.sub(r":framerate=[\d.]+", "", video_filter)
        video_filter = re.sub(r"(^|,)fps=[\d.]+(?=,|$)", "", video_filter).lstrip(",")
        args[index] = f"{prefix},{video_filter}" if video_filter else prefix

        return REPLAY_PASSTHROUGH_ARGS + args

    def _get_ffmpeg_cmd(self, ffmpeg_input: CameraInput):
        ffmpeg_output_args = []
        # replay mode decodes recorded segments on demand instead of a detect output
        if "detect" in ffmpeg_input.roles and not self.detect_replay:
            detect_args = get_ffmpeg_arg_list(self.ffmpeg.output_args.detect)
            scale_detect_args = parse_preset_hardware_acceleration_scale(
                ffmpeg_input.hwaccel_args or self.ffmpeg.hwaccel_args,
                detect_args,
                self.detect.fps,
                self.detect.width,
                self.detect.height,
            )

            ffmpeg_output_args = scale_detect_args + ffmpeg_output_args + ["pipe:"]

        if "record" in ffmpeg_input.roles and self.record.enabled:
            record_args = get_ffmpeg_arg_list(
                parse_preset_output_record(
                    self.ffmpeg.output_args.record,
                    self.ffmpeg.apple_compatibility,
                )
                or self.ffmpeg.output_args.record
            )

            ffmpeg_output_args = (
                record_args
                + [f"{os.path.join(CACHE_DIR, self.name)}@{CACHE_SEGMENT_FORMAT}.mp4"]
                + ffmpeg_output_args
            )

        if (
            "record_sub" in ffmpeg_input.roles
            and self.record.enabled
            and self.record.sub.enabled
        ):
            sub_output_args = self.ffmpeg.output_args.effective_record_sub
            record_args = get_ffmpeg_arg_list(
                parse_preset_output_record(
                    sub_output_args,
                    self.ffmpeg.apple_compatibility,
                )
                or sub_output_args
            )

            ffmpeg_output_args = (
                record_args
                + [
                    f"{os.path.join(CACHE_DIR, self.name)}{SUB_CACHE_TAG}@{CACHE_SEGMENT_FORMAT}.mp4"
                ]
                + ffmpeg_output_args
            )

        # if there aren't any outputs enabled for this input
        if len(ffmpeg_output_args) == 0:
            return None

        global_args = get_ffmpeg_arg_list(
            ffmpeg_input.global_args or self.ffmpeg.global_args
        )

        hwaccel_args = self._get_hwaccel_args(ffmpeg_input)
        input_args = get_ffmpeg_arg_list(
            parse_preset_input(ffmpeg_input.input_args, self.detect.fps)
            or ffmpeg_input.input_args
            or parse_preset_input(self.ffmpeg.input_args, self.detect.fps)
            or self.ffmpeg.input_args
        )

        cmd = (
            [self.ffmpeg.ffmpeg_path]
            + global_args
            + (
                hwaccel_args
                if "detect" in ffmpeg_input.roles and not self.detect_replay
                else []
            )
            + input_args
            + ["-i", escape_special_characters(ffmpeg_input.path)]
            + ffmpeg_output_args
        )

        return [part for part in cmd if part != ""]

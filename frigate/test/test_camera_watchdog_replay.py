"""Tests for detect mode replay in the camera watchdog and camera config."""

import threading
import unittest
from unittest.mock import MagicMock, patch

from pydantic import ValidationError

from frigate.config import CameraConfig, FrigateConfig
from frigate.video.ffmpeg import CameraWatchdog, _CombinedStopEvent


def build_config(
    detect: dict | None = None,
    inputs: list[dict] | None = None,
    record: dict | None = None,
    hwaccel: str | None = None,
) -> FrigateConfig:
    ffmpeg: dict = {
        "inputs": inputs
        or [
            {
                "path": "rtsp://10.0.0.1:554/video",
                "roles": ["record", "detect"],
            }
        ]
    }
    if hwaccel:
        ffmpeg["hwaccel_args"] = hwaccel

    return FrigateConfig(
        **{
            "mqtt": {"host": "mqtt"},
            "cameras": {
                "front_door": {
                    "ffmpeg": ffmpeg,
                    "detect": {"width": 960, "height": 432, **(detect or {})},
                    "record": record or {"enabled": True},
                }
            },
        }
    )


def build_watchdog(detect: dict | None = None, **kwargs) -> CameraWatchdog:
    camera_config = build_config(detect, **kwargs).cameras["front_door"]

    with (
        patch("frigate.video.ffmpeg.LogPipe"),
        patch("frigate.video.ffmpeg.InterProcessRequestor"),
        patch("frigate.video.ffmpeg.RecordingsDataSubscriber"),
        patch("frigate.video.ffmpeg.CameraConfigUpdateSubscriber"),
    ):
        watchdog = CameraWatchdog(
            camera_config,
            1,
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
        )

    watchdog.requestor = MagicMock()
    return watchdog


class TestDetectModeConfig(unittest.TestCase):
    def test_mode_defaults_to_continuous(self):
        camera = build_config().cameras["front_door"]

        assert camera.detect.mode == "continuous"
        assert camera.detect_replay is False

    def test_replay_is_valid(self):
        camera = build_config({"mode": "replay"}).cameras["front_door"]

        assert camera.detect_replay is True

    def test_invalid_mode_is_rejected(self):
        with self.assertRaises(ValidationError):
            build_config({"mode": "sometimes"})

    def test_replay_requires_recording(self):
        with self.assertRaises(ValueError):
            build_config({"mode": "replay"}, record={"enabled": False})

    def test_replay_rejects_continuous_retention(self):
        # the recording maintainer would trim idle footage from the cache
        with self.assertRaisesRegex(ValueError, "continuous recording retention"):
            build_config(
                {"mode": "replay"},
                record={"enabled": True, "continuous": {"days": 1}},
            )

    def test_replay_rejects_motion_retention(self):
        with self.assertRaisesRegex(ValueError, "motion recording retention"):
            build_config(
                {"mode": "replay"},
                record={"enabled": True, "motion": {"days": 1}},
            )

    def test_replay_rejects_sub_stream_retention_only_when_the_sub_stream_records(
        self,
    ):
        inputs = [
            {"path": "rtsp://10.0.0.1:554/main", "roles": ["record", "detect"]},
            {"path": "rtsp://10.0.0.1:554/sub", "roles": ["record_sub"]},
        ]
        record = {
            "enabled": True,
            "sub": {"enabled": True, "continuous": {"days": 1}},
        }

        with self.assertRaisesRegex(ValueError, "sub stream continuous"):
            build_config({"mode": "replay"}, inputs=inputs, record=record)

        # a sub stream that is not recording has nothing to retain
        record["sub"]["enabled"] = False
        camera = build_config({"mode": "replay"}, inputs=inputs, record=record)

        assert camera.cameras["front_door"].detect_replay is True

    def test_replay_keeps_event_retention(self):
        camera = build_config(
            {"mode": "replay"},
            record={
                "enabled": True,
                "alerts": {"retain": {"days": 30, "mode": "all"}},
                "detections": {"retain": {"days": 30}},
            },
        ).cameras["front_door"]

        assert camera.detect_replay is True

    def test_continuous_mode_keeps_continuous_retention(self):
        camera = build_config(
            {"mode": "continuous"},
            record={"enabled": True, "continuous": {"days": 7}},
        ).cameras["front_door"]

        assert camera.record.continuous.days == 7

    def test_on_demand_is_gone(self):
        with self.assertRaises(ValidationError):
            build_config({"on_demand": True})


class TestReplayFfmpegCommands(unittest.TestCase):
    def test_continuous_keeps_the_detect_output(self):
        camera = build_config().cameras["front_door"]
        (cmd,) = camera.ffmpeg_cmds

        assert "detect" in cmd["roles"]
        assert "pipe:" in cmd["cmd"]

    def test_replay_drops_the_detect_output_but_keeps_recording(self):
        camera = build_config(
            {"mode": "replay"}, hwaccel="preset-intel-qsv-h264"
        ).cameras["front_door"]
        (cmd,) = camera.ffmpeg_cmds

        assert "detect" not in cmd["roles"]
        assert "record" in cmd["roles"]
        assert "pipe:" not in cmd["cmd"]
        assert "h264_qsv" not in cmd["cmd"]
        assert any("front_door@" in part for part in cmd["cmd"])

    def test_replay_detect_only_input_is_not_started(self):
        camera = build_config(
            {"mode": "replay"},
            inputs=[
                {"path": "rtsp://10.0.0.1:554/main", "roles": ["record"]},
                {"path": "rtsp://10.0.0.1:554/sub", "roles": ["detect"]},
            ],
        ).cameras["front_door"]

        assert len(camera.ffmpeg_cmds) == 1
        assert "record" in camera.ffmpeg_cmds[0]["roles"]

    def test_replay_command_decodes_with_qsv_and_scales_before_download(self):
        camera = build_config(
            {"mode": "replay", "fps": 6}, hwaccel="preset-intel-qsv-h264"
        ).cameras["front_door"]

        cmd = camera.get_replay_ffmpeg_cmd("/tmp/cache/front_door@1.mp4")
        joined = " ".join(cmd)

        assert cmd[cmd.index("-i") + 1] == "/tmp/cache/front_door@1.mp4"
        assert "-c:v h264_qsv" in joined
        # resize on the gpu to the detect dimensions, then download
        assert "vpp_qsv=w=960:h=432" in joined
        assert joined.index("vpp_qsv") < joined.index("hwdownload")
        assert "rawvideo" in joined
        assert cmd[-1] == "pipe:"
        # live stream input args do not apply to a file
        assert "-rtsp_transport" not in cmd

    def test_replay_command_picks_frames_by_time_before_the_hardware_scaler(self):
        camera = build_config(
            {"mode": "replay", "fps": 6}, hwaccel="preset-intel-qsv-h264"
        ).cameras["front_door"]

        cmd = camera.get_replay_ffmpeg_cmd("/tmp/cache/front_door@1.mp4")
        video_filter = cmd[cmd.index("-vf") + 1]

        # frames are dropped on their timestamp, then their time is logged,
        # both before vpp_qsv, which would round the time to the frame rate
        assert video_filter.startswith(
            "select=isnan(prev_selected_t)+gt(floor(t*6)\\,floor(prev_selected_t*6)),"
            "showinfo,setpts=N/TB,vpp_qsv="
        )
        assert "fps=" not in video_filter
        # no forced output rate, which would duplicate and drop frames
        assert "-r" not in cmd
        assert cmd[cmd.index("-fps_mode") + 1] == "passthrough" or (
            cmd[cmd.index("-vsync") + 1] == "0"
        )

    def test_replay_command_logs_tagged_info_so_frame_times_can_be_read(self):
        camera = build_config({"mode": "replay"}).cameras["front_door"]

        cmd = camera.get_replay_ffmpeg_cmd("/tmp/cache/a.mp4")

        assert cmd[cmd.index("-loglevel") + 1] == "level+info"
        assert cmd.count("-loglevel") == 1

    def test_replay_command_never_uses_live_stream_only_bitstream_filters(self):
        camera = build_config(
            {"mode": "replay"}, hwaccel="preset-intel-qsv-h264"
        ).cameras["front_door"]
        live_args = ["-hwaccel", "qsv", "-c:v", "h264_qsv", "-bsf:v", "dump_extra"]

        with patch.object(CameraConfig, "_get_hwaccel_args", return_value=live_args):
            cmd = camera.get_replay_ffmpeg_cmd("/tmp/cache/a.mp4")

        # dump_extra makes the decoder reject every packet of an mp4 file
        assert "dump_extra" not in cmd
        assert "-bsf:v" not in cmd
        assert "h264_qsv" in cmd

    def test_replay_command_falls_back_to_cpu_decode(self):
        camera = build_config({"mode": "replay"}, hwaccel=None).cameras["front_door"]

        cmd = camera.get_replay_ffmpeg_cmd("/tmp/cache/a.mp4")
        joined = " ".join(cmd)

        assert "showinfo,setpts=N/TB,scale=960:432" in cmd[cmd.index("-vf") + 1]
        assert "qsv" not in joined


class TestReplayFilterArgs(unittest.TestCase):
    SELECT = "select=isnan(prev_selected_t)+gt(floor(t*6)\\,floor(prev_selected_t*6))"

    def _filter(self, args: list[str]) -> list[str]:
        result = CameraConfig._replay_filter_args(args, 6)
        return result[result.index("-vf") + 1 :]

    def test_removes_the_fps_filter_wherever_it_is(self):
        for video_filter in (
            "fps=6,scale=960:432",
            "scale=960:432,fps=6",
            "vpp_qsv=w=960:h=432,hwdownload,format=nv12,fps=6,format=yuv420p",
        ):
            result = self._filter(["-vf", video_filter, "-f", "rawvideo"])

            assert "fps=" not in result[0]
            assert result[0].startswith(f"{self.SELECT},showinfo,")
            assert result[1:] == ["-f", "rawvideo"]

    def test_removes_framerate_options_from_the_scaler(self):
        result = self._filter(
            ["-vf", "scale_vaapi=w=960:h=432:framerate=6:format=nv12"]
        )

        assert (
            result[0]
            == f"{self.SELECT},showinfo,setpts=N/TB,scale_vaapi=w=960:h=432:format=nv12"
        )

    def test_a_filter_that_was_only_fps_leaves_just_the_selection(self):
        assert self._filter(["-vf", "fps=6"]) == [f"{self.SELECT},showinfo,setpts=N/TB"]

    def test_adds_a_filter_when_the_preset_has_none(self):
        # presets that scale in the decoder only set an output rate
        result = CameraConfig._replay_filter_args(["-r", "6", "-f", "rawvideo"], 6)

        assert "-r" not in result
        assert result[result.index("-vf") + 1] == f"{self.SELECT},showinfo,setpts=N/TB"
        assert result[-2:] == ["-f", "rawvideo"]

    def test_does_not_change_the_arguments_it_is_given(self):
        args = ["-r", "6", "-vf", "fps=6,scale=960:432"]

        CameraConfig._replay_filter_args(args, 6)

        assert args == ["-r", "6", "-vf", "fps=6,scale=960:432"]


class TestReplayGlobalAndHwaccelArgs(unittest.TestCase):
    def test_replaces_any_existing_log_level(self):
        args = ["-hide_banner", "-loglevel", "warning", "-v", "error", "-threads", "2"]

        assert CameraConfig._replay_global_args(args) == [
            "-hide_banner",
            "-threads",
            "2",
            "-loglevel",
            "level+info",
        ]

    def test_strips_only_the_stream_only_pair(self):
        args = ["-hwaccel", "qsv", "-bsf:v", "dump_extra", "-c:v", "h264_qsv"]

        assert CameraConfig._strip_stream_only_args(args) == [
            "-hwaccel",
            "qsv",
            "-c:v",
            "h264_qsv",
        ]
        # a different bitstream filter is left alone
        assert CameraConfig._strip_stream_only_args(["-bsf:v", "h264_mp4toannexb"]) == [
            "-bsf:v",
            "h264_mp4toannexb",
        ]


class TestCameraWatchdogReplay(unittest.TestCase):
    def test_start_all_starts_replay_instead_of_detect(self):
        watchdog = build_watchdog({"mode": "replay"})

        with (
            patch.object(watchdog, "start_ffmpeg_detect") as start_detect,
            patch.object(watchdog, "start_replay") as start_replay,
            patch("frigate.video.ffmpeg.start_or_restart_ffmpeg"),
            patch("frigate.video.ffmpeg.LogPipe"),
        ):
            watchdog.start_all_ffmpeg()

        start_detect.assert_not_called()
        start_replay.assert_called_once()
        # recording still starts
        assert len(watchdog.ffmpeg_other_processes) == 1

    def test_start_all_starts_detect_when_continuous(self):
        watchdog = build_watchdog()

        with (
            patch.object(watchdog, "start_ffmpeg_detect") as start_detect,
            patch.object(watchdog, "start_replay") as start_replay,
            patch("frigate.video.ffmpeg.start_or_restart_ffmpeg"),
            patch("frigate.video.ffmpeg.LogPipe"),
        ):
            watchdog.start_all_ffmpeg()

        start_detect.assert_called_once()
        start_replay.assert_not_called()

    def test_start_replay_triggers_when_detect_is_already_enabled(self):
        watchdog = build_watchdog({"mode": "replay", "enabled": True})

        with patch("frigate.video.ffmpeg.DetectReplayRunner") as runner_class:
            watchdog.start_replay()

        runner_class.return_value.start.assert_called_once()
        runner_class.return_value.trigger.assert_called_once()

    def test_start_replay_stays_idle_when_detect_is_disabled(self):
        watchdog = build_watchdog({"mode": "replay", "enabled": False})

        with patch("frigate.video.ffmpeg.DetectReplayRunner") as runner_class:
            watchdog.start_replay()

        runner_class.return_value.trigger.assert_not_called()

    def test_start_replay_triggers_from_the_time_of_the_event(self):
        watchdog = build_watchdog({"mode": "replay", "enabled": True})
        watchdog.config.detect.mark_event(500.0)

        with patch("frigate.video.ffmpeg.DetectReplayRunner") as runner_class:
            watchdog.start_replay()

        runner_class.return_value.trigger.assert_called_once_with(500.0)

    def test_a_detect_on_event_triggers_the_runner_at_the_event_time(self):
        watchdog = build_watchdog({"mode": "replay"})
        watchdog.replay_runner = MagicMock()

        watchdog._on_detect_event("front_door", True, 500.0)

        watchdog.replay_runner.trigger.assert_called_once_with(500.0)
        watchdog.replay_runner.release.assert_not_called()

    def test_a_detect_off_event_releases_the_runner_at_the_event_time(self):
        watchdog = build_watchdog({"mode": "replay"})
        watchdog.replay_runner = MagicMock()

        watchdog._on_detect_event("front_door", False, 520.0)

        watchdog.replay_runner.release.assert_called_once_with(520.0)
        watchdog.replay_runner.trigger.assert_not_called()

    def test_a_detect_event_without_a_time_uses_the_time_it_was_handled(self):
        watchdog = build_watchdog({"mode": "replay"})
        watchdog.replay_runner = MagicMock()

        with patch("frigate.video.ffmpeg.time") as fake_time:
            fake_time.time.return_value = 700.0
            watchdog._on_detect_event("front_door", True, None)

        watchdog.replay_runner.trigger.assert_called_once_with(700.0)

    def test_detect_events_for_other_cameras_are_ignored(self):
        watchdog = build_watchdog({"mode": "replay"})
        watchdog.replay_runner = MagicMock()

        watchdog._on_detect_event("back_yard", True, 500.0)

        watchdog.replay_runner.trigger.assert_not_called()

    def test_a_detect_event_without_a_runner_is_ignored(self):
        # continuous mode, or replay between a stop and a restart
        watchdog = build_watchdog()
        assert watchdog.replay_runner is None

        watchdog._on_detect_event("front_door", True, 500.0)

    def test_the_watchdog_listens_for_detect_events(self):
        camera_config = build_config({"mode": "replay"}).cameras["front_door"]

        with (
            patch("frigate.video.ffmpeg.LogPipe"),
            patch("frigate.video.ffmpeg.InterProcessRequestor"),
            patch("frigate.video.ffmpeg.RecordingsDataSubscriber"),
            patch("frigate.video.ffmpeg.CameraConfigUpdateSubscriber") as subscriber,
        ):
            watchdog = CameraWatchdog(
                camera_config, 1, *[MagicMock() for _ in range(8)]
            )

        callback = subscriber.call_args.kwargs["on_detect_event"]
        assert callback == watchdog._on_detect_event

    def test_start_replay_does_not_hand_the_runner_the_detect_log_pipe(self):
        watchdog = build_watchdog({"mode": "replay"})

        with patch("frigate.video.ffmpeg.DetectReplayRunner") as runner_class:
            watchdog.start_replay()

        assert watchdog.logpipe not in runner_class.call_args.args
        assert len(runner_class.call_args.args) == 5

    def test_stop_all_stops_the_replay_thread(self):
        watchdog = build_watchdog({"mode": "replay"})
        runner = MagicMock()
        watchdog.replay_runner = runner

        watchdog.stop_all_ffmpeg()

        runner.stop.assert_called_once()
        assert watchdog.replay_runner is None

    def test_stop_all_stops_and_joins_capture_thread(self):
        watchdog = build_watchdog()
        capture_thread = MagicMock()
        capture_thread.is_alive.return_value = True
        watchdog.capture_thread = capture_thread

        detect_process = MagicMock()
        watchdog.ffmpeg_detect_process = detect_process

        with patch("frigate.video.ffmpeg.stop_ffmpeg") as stop_process:
            watchdog.stop_all_ffmpeg()

        capture_thread.request_stop.assert_called_once()
        stop_process.assert_called_once_with(detect_process, watchdog.logger)
        capture_thread.join.assert_called_once_with(timeout=5)
        assert watchdog.capture_thread is None
        assert watchdog.ffmpeg_detect_process is None

    def test_reset_capture_thread_restarts_detect(self):
        watchdog = build_watchdog()
        watchdog.ffmpeg_detect_process = None
        watchdog.capture_thread = None

        with patch.object(watchdog, "start_ffmpeg_detect") as start_detect:
            watchdog.reset_capture_thread(terminate=True)

        start_detect.assert_called_once()


class TestCombinedStopEvent(unittest.TestCase):
    def test_trips_on_either_event(self):
        global_stop = threading.Event()
        local_stop = threading.Event()
        combined = _CombinedStopEvent(global_stop, local_stop)

        assert combined.is_set() is False

        local_stop.set()
        assert combined.is_set() is True

        local_stop.clear()
        global_stop.set()
        assert combined.is_set() is True

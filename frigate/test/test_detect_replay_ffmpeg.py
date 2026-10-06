"""Runs the detect replay ffmpeg command on a real file instead of mocking it.

The command tests next to these only compare strings. These run ffmpeg, so a
filter graph that is accepted but behaves differently (frames resampled,
times rounded, a surface copied between the GPU and the host) fails here.
Everything skips when ffmpeg or, for QSV, an Intel render device is missing.
"""

import os
import queue
import shutil
import subprocess as sp
import tempfile
import threading
import unittest
from unittest.mock import MagicMock

from frigate.config import CameraConfig, FrigateConfig
from frigate.util.cache_frame import CacheSegment
from frigate.video.detect_replay import SHOWINFO_TIME, DetectReplayRunner

WIDTH = 640
HEIGHT = 360
DETECT_FPS = 6
FRAME_SIZE = WIDTH * HEIGHT * 3 // 2
CLIP_SECONDS = 4
SOURCE_FPS = 15
QSV_DEVICE = "/dev/dri/renderD128"


def build_camera(hwaccel: str | None = None) -> CameraConfig:
    ffmpeg: dict = {
        "inputs": [
            {"path": "rtsp://10.0.0.1:554/video", "roles": ["record", "detect"]}
        ]
    }

    if hwaccel:
        ffmpeg["hwaccel_args"] = hwaccel

    config = FrigateConfig(
        **{
            "mqtt": {"host": "mqtt"},
            "cameras": {
                "front_door": {
                    "ffmpeg": ffmpeg,
                    "detect": {
                        "mode": "replay",
                        "width": WIDTH,
                        "height": HEIGHT,
                        "fps": DETECT_FPS,
                    },
                    "record": {"enabled": True},
                }
            },
        }
    )
    return config.cameras["front_door"]


class ReplayFfmpegChecks:
    """Shared checks, mixed into one TestCase per decode path."""

    hwaccel: str | None = None
    camera: CameraConfig
    ffmpeg: str
    clip: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.camera = build_camera(cls.hwaccel)
        cls.ffmpeg = cls.camera.ffmpeg.ffmpeg_path

        if shutil.which(cls.ffmpeg) is None:
            raise unittest.SkipTest(f"ffmpeg is not available at {cls.ffmpeg}")

        tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(tmp.cleanup)
        cls.clip = os.path.join(tmp.name, "front_door@20260101000000+0000.mp4")

        # a keyframe every second like a camera stream, larger than the
        # detect size so the scaler has work to do
        result = sp.run(
            [
                cls.ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                f"testsrc2=size=1280x720:rate={SOURCE_FPS}",
                "-t",
                str(CLIP_SECONDS),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-g",
                str(SOURCE_FPS),
                "-an",
                cls.clip,
            ],
            capture_output=True,
            timeout=60,
            check=False,
        )

        if result.returncode != 0:
            raise unittest.SkipTest(
                "ffmpeg cannot encode the test clip: "
                + result.stderr.decode(errors="replace")[-200:]
            )

    def _run(self, loglevel: str | None = None) -> sp.CompletedProcess:
        cmd = self.camera.get_replay_ffmpeg_cmd(self.clip)
        assert cmd is not None

        if loglevel is not None:
            cmd[cmd.index("-loglevel") + 1] = loglevel

        return sp.run(cmd, capture_output=True, timeout=120, check=False)

    @staticmethod
    def _times(stderr: bytes) -> list[float]:
        times = []

        for line in stderr.decode(errors="replace").splitlines():
            match = SHOWINFO_TIME.search(line) if "showinfo" in line else None

            if match:
                times.append(float(match.group(1)))

        return times

    def test_frames_come_out_at_the_detect_size_in_yuv420p(self):
        result = self._run()

        assert result.returncode == 0, result.stderr.decode(errors="replace")[-300:]
        assert len(result.stdout) % FRAME_SIZE == 0
        # one frame per 1/fps window of a 4 second clip
        assert 20 <= len(result.stdout) // FRAME_SIZE <= 26

    def test_every_kept_frame_reports_its_own_capture_time(self):
        result = self._run()
        frames = len(result.stdout) // FRAME_SIZE
        times = self._times(result.stderr)

        # frames and times are matched by position, so the counts must agree
        assert len(times) == frames
        assert times[0] == 0.0
        assert all(b > a for a, b in zip(times, times[1:]))
        assert times[-1] < CLIP_SECONDS

    def test_times_are_the_source_frame_times_and_not_resampled(self):
        times = self._times(self._run().stderr)

        # an fps filter or the hardware scaler's frame rate would move these
        # onto the 1/6 second grid, or duplicate frames to fill it
        for time in times:
            assert abs(time * SOURCE_FPS - round(time * SOURCE_FPS)) < 0.01

    def test_the_runner_replays_the_clip_with_its_original_timestamps(self):
        segment = CacheSegment(self.clip, 1000.0, 1000.0 + CLIP_SECONDS)
        frame_queue: queue.Queue = queue.Queue()
        frame_manager = MagicMock()
        frame_manager.write.side_effect = lambda name: bytearray(FRAME_SIZE)
        runner = DetectReplayRunner(
            self.camera,
            10,
            frame_queue,
            MagicMock(),
            threading.Event(),
            frame_manager=frame_manager,
            list_segments=lambda *_: [segment],
        )

        runner.trigger(1002.0)
        runner._replay_pending()

        items = []
        while not frame_queue.empty():
            items.append(frame_queue.get())

        times = [item[1] for item in items]
        assert 20 <= len(times) <= 26
        assert times[0] == 1000.0
        assert times == sorted(times)
        assert all(1000.0 <= t < 1000.0 + CLIP_SECONDS for t in times)
        # a whole segment of pre roll covers the start of the clip
        assert all(item[2] for item in items)
        # nothing was captured at the time of replay
        assert max(times) < 2000.0


class TestReplayOnCpu(ReplayFfmpegChecks, unittest.TestCase):
    hwaccel = None


@unittest.skipUnless(os.path.exists(QSV_DEVICE), "no Intel render device")
class TestReplayOnQsv(ReplayFfmpegChecks, unittest.TestCase):
    hwaccel = "preset-intel-qsv-h264"

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        # a render device that is not Intel, or without the media driver,
        # cannot run QSV at all, which says nothing about the replay command
        probe = sp.run(
            [
                cls.ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-hwaccel",
                "qsv",
                "-qsv_device",
                QSV_DEVICE,
                "-hwaccel_output_format",
                "qsv",
                "-c:v",
                "h264_qsv",
                "-i",
                cls.clip,
                "-frames:v",
                "1",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            timeout=60,
            check=False,
        )

        if probe.returncode != 0:
            raise unittest.SkipTest(
                "QSV decode is not usable here: "
                + probe.stderr.decode(errors="replace")[-200:]
            )

    def test_the_graph_stays_on_the_gpu_until_the_download(self):
        result = self._run(loglevel="level+verbose")
        stderr = result.stderr.decode(errors="replace")
        showinfo = [
            line
            for line in stderr.splitlines()
            if "showinfo" in line and "pts_time" in line
        ]

        assert result.returncode == 0, stderr[-300:]
        assert showinfo
        # showinfo runs before vpp_qsv, so the frames it sees are still GPU
        # surfaces, not frames that were downloaded and sent back up
        assert all("fmt:qsv" in line for line in showinfo)
        assert "hwupload" not in stderr.lower()


if __name__ == "__main__":
    unittest.main()

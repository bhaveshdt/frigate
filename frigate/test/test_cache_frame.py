"""Tests for grabbing a still frame from a camera's recording cache."""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from frigate.util.cache_frame import (
    get_latest_cache_frame,
    get_latest_finished_cache_segment,
)

NOW = 1_000_000.0


class TestLatestFinishedCacheSegment(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _touch(self, name: str, age: float) -> str:
        path = os.path.join(self.tmp.name, name)
        open(path, "w").close()
        os.utime(path, (NOW - age, NOW - age))
        return path

    def _latest(self, camera: str = "front_door", **kwargs) -> str | None:
        return get_latest_finished_cache_segment(
            camera, cache_dir=self.tmp.name, now=NOW, **kwargs
        )

    def test_skips_the_segment_still_being_written(self):
        self._touch("front_door@20260101000000-0400.mp4", age=25)
        finished = self._touch("front_door@20260101000010-0400.mp4", age=15)
        self._touch("front_door@20260101000020-0400.mp4", age=5)

        assert self._latest() == finished

    def test_needs_a_finished_segment(self):
        assert self._latest() is None

        self._touch("front_door@20260101000000-0400.mp4", age=5)
        assert self._latest() is None

    def test_ignores_sub_streams_and_other_cameras(self):
        self._touch("front_door@20260101000000-0400.mp4", age=25)
        self._touch("front_door@sub@20260101000010-0400.mp4", age=5)
        self._touch("front_door@sub@20260101000020-0400.mp4", age=2)
        self._touch("front_door_2@20260101000030-0400.mp4", age=1)
        self._touch("front_door_2@20260101000040-0400.mp4", age=0.5)

        # only one main segment exists, and it is still the newest
        assert self._latest() is None

    def test_ignores_stale_segments(self):
        self._touch("front_door@20260101000000-0400.mp4", age=500)
        self._touch("front_door@20260101000010-0400.mp4", age=490)

        assert self._latest() is None
        assert self._latest(max_age=600) is not None


class TestLatestCacheFrame(unittest.TestCase):
    def test_returns_a_bgr_frame(self):
        image = np.zeros((20, 30, 3), np.uint8)
        _, jpg = cv2.imencode(".jpg", image)

        with (
            patch(
                "frigate.util.cache_frame.get_latest_finished_cache_segment",
                return_value="/tmp/cache/front_door@1.mp4",
            ),
            patch(
                "frigate.util.cache_frame.run_ffmpeg_snapshot",
                return_value=(jpg.tobytes(), ""),
            ) as snapshot,
        ):
            frame = get_latest_cache_frame(MagicMock(), "front_door")

        assert frame is not None
        assert frame.shape == (20, 30, 3)
        assert snapshot.call_args.args[1] == "/tmp/cache/front_door@1.mp4"
        assert snapshot.call_args.args[2] == "mjpeg"
        assert snapshot.call_args.kwargs["seek_from_end"] == 0.5

    def test_returns_none_without_a_segment(self):
        with (
            patch(
                "frigate.util.cache_frame.get_latest_finished_cache_segment",
                return_value=None,
            ),
            patch("frigate.util.cache_frame.run_ffmpeg_snapshot") as snapshot,
        ):
            assert get_latest_cache_frame(MagicMock(), "front_door") is None

        snapshot.assert_not_called()

    def test_returns_none_when_ffmpeg_fails(self):
        with (
            patch(
                "frigate.util.cache_frame.get_latest_finished_cache_segment",
                return_value="/tmp/cache/front_door@1.mp4",
            ),
            patch(
                "frigate.util.cache_frame.run_ffmpeg_snapshot",
                return_value=(None, "moov atom not found"),
            ),
        ):
            assert get_latest_cache_frame(MagicMock(), "front_door") is None

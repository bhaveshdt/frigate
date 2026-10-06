"""Tests for what latest.{ext} returns for a camera in detect mode replay.

In replay mode nothing decodes a live detect stream, so the frame the tracker
holds was replayed from the recording cache and carries its original capture
time. By that clock it is always stale, which is why replay cameras are served
the newest finished cache segment instead. The response says so in
X-Frigate-Frame-Source, and it is never marked offline.
"""

import asyncio
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np

from frigate.api.defs.query.media_query_parameters import Extension
from frigate.api.media import latest_frame

FRAME = np.zeros((90, 160, 3), np.uint8)


class TestLatestFrameInReplayMode(unittest.TestCase):
    @staticmethod
    def _request(
        *,
        replay: bool = True,
        enabled: bool = True,
        live_frame=None,
        frame_time: float = 0.0,
    ) -> MagicMock:
        camera = MagicMock()
        camera.enabled = enabled
        camera.detect_replay = replay
        camera.ffmpeg.retry_interval = 10

        processor = MagicMock()
        processor.get_current_frame.return_value = live_frame
        processor.get_current_frame_time.return_value = frame_time

        request = MagicMock()
        request.app.detected_frames_processor = processor
        request.app.frigate_config.cameras = {"front_door": camera}
        request.app.camera_error_image = None
        return request

    @staticmethod
    def _params() -> MagicMock:
        params = MagicMock()
        params.bbox = params.timestamp = params.zones = params.mask = 0
        params.motion = params.paths = params.regions = 0
        params.quality = 70
        params.height = None
        params.store = 0
        return params

    def _get(self, request, cache_frame=None, preview_path=None):
        with (
            patch(
                "frigate.api.media.get_latest_cache_frame", return_value=cache_frame
            ) as cache,
            patch(
                "frigate.api.media.get_most_recent_preview_frame",
                return_value=preview_path,
            ) as preview,
            patch("frigate.api.media.cv2.imread", return_value=FRAME),
        ):
            response = asyncio.run(
                latest_frame(request, "front_door", Extension.jpg, self._params())
            )

        return response, cache, preview

    def test_serves_the_newest_cache_segment_when_nothing_is_live(self):
        response, cache, preview = self._get(self._request(), cache_frame=FRAME)

        assert response.status_code == 200
        assert response.headers["X-Frigate-Frame-Source"] == "recording-cache"
        # a cache frame is current footage, not an offline fallback
        assert "X-Frigate-Offline" not in response.headers
        cache.assert_called_once()
        preview.assert_not_called()

    def test_a_replayed_frame_is_stale_by_capture_time_so_the_cache_wins(self):
        # the tracker holds a frame replayed from an hour ago
        request = self._request(
            live_frame=FRAME, frame_time=datetime.now().timestamp() - 3600
        )

        response, cache, _ = self._get(request, cache_frame=FRAME)

        assert response.headers["X-Frigate-Frame-Source"] == "recording-cache"
        cache.assert_called_once()

    def test_a_fresh_live_frame_is_served_without_touching_the_cache(self):
        now = datetime.now().timestamp()
        request = self._request(live_frame=FRAME, frame_time=now)

        response, cache, preview = self._get(request)

        assert response.status_code == 200
        assert "X-Frigate-Frame-Source" not in response.headers
        cache.assert_not_called()
        preview.assert_not_called()

    def test_falls_back_to_the_preview_when_the_cache_has_no_frame(self):
        response, cache, _ = self._get(
            self._request(), cache_frame=None, preview_path="/tmp/preview.webp"
        )

        assert response.status_code == 200
        cache.assert_called_once()
        assert response.headers["X-Frigate-Offline"] == "true"
        assert "X-Frigate-Frame-Source" not in response.headers

    def test_continuous_mode_never_reads_the_recording_cache(self):
        response, cache, _ = self._get(
            self._request(replay=False), preview_path="/tmp/preview.webp"
        )

        cache.assert_not_called()
        assert response.headers["X-Frigate-Offline"] == "true"
        assert "X-Frigate-Frame-Source" not in response.headers

    def test_a_disabled_camera_does_not_read_the_recording_cache(self):
        _, cache, _ = self._get(
            self._request(enabled=False), preview_path="/tmp/preview.webp"
        )

        cache.assert_not_called()


if __name__ == "__main__":
    unittest.main()

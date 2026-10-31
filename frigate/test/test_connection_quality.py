"""Tests for the per-camera connection quality rating."""

import unittest

from frigate.stats.util import rate_connection


class TestRateConnection(unittest.TestCase):
    def test_steady_stream_is_excellent(self):
        self.assertEqual(rate_connection(5.0, 5, 0, 0), "excellent")

    def test_dropped_fps_is_fair(self):
        self.assertEqual(rate_connection(3.5, 5, 1, 0), "fair")

    def test_no_frames_is_unusable(self):
        self.assertEqual(rate_connection(0.0, 5, 0, 0), "unusable")

    def test_many_reconnects_is_unusable(self):
        self.assertEqual(rate_connection(4.0, 5, 11, 0), "unusable")

    def test_in_between_is_poor(self):
        self.assertEqual(rate_connection(2.0, 5, 4, 0), "poor")

    def test_idle_replay_camera_is_neutral_not_unusable(self):
        self.assertEqual(rate_connection(0.0, 5, 0, 0, detect_replay=True), "replay")

    def test_replay_camera_is_neutral_at_any_fps(self):
        # decoding a segment runs far faster than detect.fps
        self.assertEqual(rate_connection(40.0, 5, 0, 0, detect_replay=True), "replay")


if __name__ == "__main__":
    unittest.main()

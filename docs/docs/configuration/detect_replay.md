---
id: detect_replay
title: Detect Replay Mode
---

By default Frigate decodes a camera's detect stream all the time (`detect.mode: continuous`), even when nothing is happening. For cameras where an external or hardware event, such as the camera's own motion detection, already says when something is going on, `detect.mode: replay` avoids that continuous decoding.

```yaml
cameras:
  front_door:
    ffmpeg:
      hwaccel_args: preset-intel-qsv-h264
      inputs:
        # the main stream records and is also the replay source
        - path: rtsp://camera/main
          roles:
            - record
            - detect
    detect:
      mode: replay
      width: 960
      height: 432
      fps: 6
```

## How it works

- `continuous` is the default. It is Frigate's normal behavior and nothing changes for configs without `mode`.
- In `replay` no process decodes a stream for detection. Recording keeps running and the live view is untouched.
- The trigger is the existing detect toggle: when detection is turned on (for example `frigate/<camera>/detect/set` with `ON` from Home Assistant, driven by the camera's motion event), Frigate decodes the recorded main stream footage and runs it through the normal detector, tracker, events, review items, snapshots and thumbnails.
- The time of the event is the time Frigate received the toggle, not the time a background check noticed it. Every toggle is handled in order, so an `ON` followed quickly by an `OFF` is still replayed.
- Replay starts with the recording segment that covers the point one segment length (10 seconds by default) before that event. Frames older than that point are still replayed but with detection off, so objects are only detected from one segment before the event onward.
- Frames keep the time they were captured at. The time comes from the footage itself: the timestamp of each frame inside its segment, added to when that segment started. The segment start is found the same way Frigate finds it for recordings (the time the file was finished, minus its duration). Frames are never stamped with the time of replay.
- Replay selects frames, it never invents them. At most one frame is replayed per `1 / detect.fps` window, and no frame is duplicated to fill a gap. A camera that delivers frames unevenly can therefore give slightly fewer frames than `detect.fps` suggests.
- Replay is not a delayed start of the detect stream. It analyzes footage that was already recorded.
- When detection is turned off, Frigate keeps replaying a short tail of footage so tracked objects can end and motion can report off.
- Overlapping or repeated triggers share one replay per camera. Footage is replayed once.

## Timing

A recording segment can only be decoded once ffmpeg has finished writing it, so detection trails the event by up to one segment (10 seconds by default) plus decode time. Events and recordings still use the correct times.

## Live frame

The `latest.jpg` frame of a replay camera is the newest finished frame from the recording cache, up to one segment old, with the response header `X-Frigate-Frame-Source: recording-cache`. It is not marked offline. Overlay options such as `bbox` and `motion` have nothing to draw on it, because no detection is running on that frame. If the cache has no frame, the most recent preview frame is used and the response is marked offline, as for any camera that is not delivering frames.

## Hardware acceleration

Replay uses the same hardware acceleration settings as the detect role. With `preset-intel-qsv-h264` the H.264 footage is decoded on the GPU and resized to `detect.width` and `detect.height` on the GPU before it is downloaded for the detector. Frames that are not replayed are dropped before they are resized or downloaded. Without hardware acceleration the footage is decoded and scaled on the CPU.

## Requirements and notes

- Recording must be enabled for the camera, and an input must have the `record` role. This is checked when the config loads and again before a camera is switched to replay at runtime. A switch that would break either rule is refused and logged as an error.
- The input with the `detect` role should be the same input that records. A separate detect-only input is not started in replay mode.
- Recording retention based on events (alerts and detections) works as usual because replayed frames drive it.
- Continuous and motion based retention (`record.continuous.days`, `record.motion.days` and the same under `record.sub` when the sub stream records) must be `0`. Frigate only keeps footage for those once it has processed frames from after that footage, and in replay mode nothing is processed between events, so the footage would be silently discarded from the cache. The combination is rejected when the config loads instead.
- While nothing is replayed the cache only holds the most recent segments, so the pre-event footage available to a trigger is bounded by that.
- A segment that cannot be decoded is skipped and logged as a warning with the error from ffmpeg.
- `detect.on_demand` from earlier builds is removed. Use `mode: replay` instead.

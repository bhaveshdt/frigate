"""Wall clock timing of recording cache segments, shared by recording and replay."""

# cache file names have whole second resolution, so a segment's real start is
# up to a second after the start in its name
SEGMENT_NAME_TOLERANCE_S = 1.0


def measure_segment_start(
    name_start: float, mtime: float | None, duration: float
) -> float | None:
    """Measure when a segment's first frame was captured from its cache file.

    ffmpeg closes a cache file when it rolls to the next segment, so the mtime
    is the wall clock at the end of the footage and the probed duration says
    how long ago it began. A result outside the truncation window of the file
    name means the media is shorter than its wall span (a stalled stream, an
    early close), and None is returned so the caller falls back to the floored
    name, which is the safer start.

    The recording maintainer resolves the start it stores for a recording from
    this same measurement, and replay uses it so both agree on segment times.

    Args:
        name_start: epoch seconds parsed from the file name
        mtime: modification time of the cache file, or None if unreadable
        duration: probed duration of the segment in seconds

    Returns:
        Epoch seconds of the first frame, or None when it cannot be measured
    """
    if mtime is None:
        return None

    candidate = mtime - duration

    if 0 <= candidate - name_start < SEGMENT_NAME_TOLERANCE_S:
        return candidate

    return None

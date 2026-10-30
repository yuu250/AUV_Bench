"""Slicing geometry for tall page screenshots (Task 1 protocol).

A page is sent as a scaled-down overview (navigation aid only, only when it
needs more than one slice) followed by fixed-height horizontal slices from
top to bottom. The last slice is bottom-aligned so it always ends at the
page bottom; pages needing more than MAX_SLICES slices are excluded.
"""

DEFAULT_SLICE_H = 1800
DEFAULT_OVERLAP = 180
DEFAULT_OVERVIEW_H = 1500
DEFAULT_MAX_SLICES = 12


def slice_offsets(height: int, slice_h: int = DEFAULT_SLICE_H,
                  overlap: int = DEFAULT_OVERLAP) -> list[int]:
    """Top y-coordinate of every slice; the last slice ends at the page bottom."""
    if height <= slice_h:
        return [0]
    offs = list(range(0, height - slice_h + 1, slice_h - overlap))
    last = height - slice_h
    if offs[-1] != last:
        offs.append(last)
    return offs


def n_slices(height: int, slice_h: int = DEFAULT_SLICE_H,
             overlap: int = DEFAULT_OVERLAP) -> int:
    return len(slice_offsets(height, slice_h, overlap))

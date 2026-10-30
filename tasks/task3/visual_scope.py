#!/usr/bin/env python3
"""Spatial-compliance utilities for the Task 2 -> Task 3 evaluator.

Task 2 locations are expressed on fixed 1800-pixel screenshot canvases.  This
module converts those locations to full-page CSS coordinates and measures how
much the rendered Task 3 output changes outside the model's declared regions.

The pixel score is intentionally tolerant in three ways:

* rendered images are reduced to approximately one sample per CSS pixel, with
  an additional fixed analysis-pixel cap for pathological long pages;
* predicted boxes receive a small, scale-aware halo;
* low-amplitude RGB changes are ignored; and
* tiny connected components (for example antialiasing noise) are removed.

All effective thresholds, the actual analysis resolution, and a conservative
16-bytes-per-grid-cell working-set estimate are returned so a formal run is
auditable.  Source PNG decoder memory is format/library dependent and is not
part of that estimate, but source images are loaded and reduced one at a time.
The score is a behavioural scope check, not a replacement for the lesion,
collateral, or content-preservation scorers.

A fixed render that matches the pre-degradation baseline within the same
0.01%-of-page pixel budget cannot contain any visual over-edit, so
``apply_baseline_restoration_fast_path`` additionally passes the scope check
on a direct per-pixel RGB comparison of the fixed and baseline full-page
renders, regardless of the region verdict.  The fast path can only flip the
combined verdict False -> True; the region-only verdict stays recorded under
``visual_scope_pass_region_check``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

from PIL import Image, ImageChops

try:  # Keep coordinate helpers importable in lightweight environments.
    import numpy as np
    from scipy import ndimage
except ImportError as error:  # pragma: no cover - exercised only without eval deps
    np = None
    ndimage = None
    _ARRAY_IMPORT_ERROR = error
else:
    _ARRAY_IMPORT_ERROR = None


Box = tuple[float, float, float, float]  # x1, y1, x2, y2 in full-page CSS px
ImageInput = Image.Image | str | Path


@dataclass(frozen=True)
class VisualScopeConfig:
    """Frozen defaults for the rendered scope-compliance check."""

    slice_height_px: int = 1800
    slice_overlap_px: int = 180
    min_box_inflation_css: float = 16.0
    box_inflation_fraction: float = 0.10
    channel_delta_threshold: int = 16
    min_component_area_css2: float = 4.0
    max_analysis_pixels: int = 12_000_000
    max_outside_page_fraction: float = 0.0001  # 0.01% of the rendered canvas
    max_outside_changed_fraction: float = 0.02  # 2% of significant changes
    far_distance_css: float = 32.0
    far_component_area_css2: float = 64.0

    def validate(self) -> None:
        if self.slice_height_px <= 0:
            raise ValueError("slice_height_px must be positive")
        if not 0 <= self.slice_overlap_px < self.slice_height_px:
            raise ValueError("slice_overlap_px must be in [0, slice_height_px)")
        if self.min_box_inflation_css < 0 or self.box_inflation_fraction < 0:
            raise ValueError("box inflation thresholds must be non-negative")
        if not 0 <= self.channel_delta_threshold <= 255:
            raise ValueError("channel_delta_threshold must be in [0, 255]")
        if self.min_component_area_css2 <= 0:
            raise ValueError("min_component_area_css2 must be positive")
        if type(self.max_analysis_pixels) is not int or self.max_analysis_pixels <= 0:
            raise ValueError("max_analysis_pixels must be a positive integer")
        for name in ("max_outside_page_fraction", "max_outside_changed_fraction"):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.far_distance_css < 0 or self.far_component_area_css2 < 0:
            raise ValueError("far-component thresholds must be non-negative")


DEFAULT_VISUAL_SCOPE_CONFIG = VisualScopeConfig()


def slice_offsets(image_height_px: int, *, slice_height_px: int = 1800,
                  slice_overlap_px: int = 180) -> list[int]:
    """Return the exact fixed-canvas offsets used by Task 2."""

    if type(image_height_px) is not int or image_height_px <= 0:
        raise ValueError("image_height_px must be a positive integer")
    if type(slice_height_px) is not int or slice_height_px <= 0:
        raise ValueError("slice_height_px must be a positive integer")
    if (type(slice_overlap_px) is not int
            or not 0 <= slice_overlap_px < slice_height_px):
        raise ValueError("slice_overlap_px must be in [0, slice_height_px)")
    if image_height_px <= slice_height_px:
        return [0]
    step = slice_height_px - slice_overlap_px
    offsets = list(range(0, image_height_px - slice_height_px + 1, step))
    last = image_height_px - slice_height_px
    if offsets[-1] != last:
        offsets.append(last)
    return offsets


def _positive_size(size: Sequence[int], name: str) -> tuple[int, int]:
    if (not isinstance(size, Sequence) or isinstance(size, (str, bytes))
            or len(size) != 2):
        raise ValueError(f"{name} must be a (width, height) pair")
    width, height = size
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise ValueError(f"{name} must contain positive integers")
    return width, height


def task2_locations_to_page_css_boxes(
    locations: Sequence[Mapping[str, object]],
    reference_image_size_px: Sequence[int],
    dpr: float,
    *,
    slice_height_px: int = 1800,
    slice_overlap_px: int = 180,
    slice_offsets_px: Sequence[int] | None = None,
) -> list[Box]:
    """Convert Task 2 fixed-slice locations into full-page CSS boxes.

    ``reference_image_size_px`` must be the degraded screenshot size used to
    construct the original Task 2 slices, even when Task 2 selected the clean
    version and Task 3 therefore renders a differently sized page.  Final-slice
    coordinates still use the full fixed canvas height; blank padding is clipped
    only after conversion, matching ``task2_pair.run_trial.score_locations``.
    """

    width_px, height_px = _positive_size(reference_image_size_px,
                                          "reference_image_size_px")
    if not isinstance(dpr, (int, float)) or isinstance(dpr, bool):
        raise ValueError("dpr must be a positive finite number")
    dpr = float(dpr)
    if not math.isfinite(dpr) or dpr <= 0:
        raise ValueError("dpr must be a positive finite number")
    if not isinstance(locations, Sequence) or isinstance(locations, (str, bytes)):
        raise ValueError("locations must be a sequence")

    if slice_offsets_px is None:
        offsets = slice_offsets(height_px, slice_height_px=slice_height_px,
                                slice_overlap_px=slice_overlap_px)
        offsets_source = "derived_from_reference_height"
    else:
        if (
            not isinstance(slice_offsets_px, Sequence)
            or isinstance(slice_offsets_px, (str, bytes))
            or not slice_offsets_px
            or any(type(value) is not int or value < 0 for value in slice_offsets_px)
        ):
            raise ValueError("slice_offsets_px must contain non-negative integers")
        offsets = list(slice_offsets_px)
        if offsets[0] != 0 or any(
            right <= left for left, right in zip(offsets, offsets[1:])
        ):
            raise ValueError(
                "slice_offsets_px must start at zero and be strictly increasing"
            )
        offsets_source = "explicit_frozen_geometry"
    page_width_css = width_px / dpr
    page_height_css = height_px / dpr
    output: list[Box] = []
    for index, location in enumerate(locations):
        if not isinstance(location, Mapping):
            raise ValueError(f"locations[{index}] must be an object")
        slice_number = location.get("slice")
        raw_box = location.get("box")
        if (type(slice_number) is not int
                or not 1 <= slice_number <= len(offsets)):
            raise ValueError(f"locations[{index}].slice is out of range")
        if (not isinstance(raw_box, Sequence) or isinstance(raw_box, (str, bytes))
                or len(raw_box) != 4):
            raise ValueError(f"locations[{index}].box must contain four coordinates")
        coords: list[float] = []
        for value in raw_box:
            if (not isinstance(value, (int, float)) or isinstance(value, bool)
                    or not math.isfinite(float(value)) or not 0 <= float(value) <= 1000):
                raise ValueError(f"locations[{index}].box has an invalid coordinate")
            coords.append(float(value))
        x1, y1, x2, y2 = coords
        if x1 >= x2 or y1 >= y2:
            raise ValueError(f"locations[{index}].box is not a positive-area box")

        y_offset_px = offsets[slice_number - 1]
        converted = (
            x1 / 1000.0 * width_px / dpr,
            (y_offset_px + y1 / 1000.0 * slice_height_px) / dpr,
            x2 / 1000.0 * width_px / dpr,
            (y_offset_px + y2 / 1000.0 * slice_height_px) / dpr,
        )
        clipped = (
            max(0.0, min(page_width_css, converted[0])),
            max(0.0, min(page_height_css, converted[1])),
            max(0.0, min(page_width_css, converted[2])),
            max(0.0, min(page_height_css, converted[3])),
        )
        if clipped[0] < clipped[2] and clipped[1] < clipped[3]:
            output.append(clipped)
    # Keep the local variable purposeful for debuggers without changing the
    # long-established list return type of this public helper.
    _ = offsets_source
    return output


def inflate_css_boxes(
    boxes: Sequence[Sequence[float]],
    page_size_css: Sequence[float],
    *,
    min_inflation_css: float = 16.0,
    inflation_fraction: float = 0.10,
) -> list[Box]:
    """Inflate each box by ``max(minimum, fraction * shorter_side)``."""

    if (not isinstance(page_size_css, Sequence) or isinstance(page_size_css, (str, bytes))
            or len(page_size_css) != 2):
        raise ValueError("page_size_css must be a (width, height) pair")
    page_width, page_height = (float(value) for value in page_size_css)
    if (not math.isfinite(page_width) or not math.isfinite(page_height)
            or page_width <= 0 or page_height <= 0):
        raise ValueError("page_size_css must contain positive finite values")
    if min_inflation_css < 0 or inflation_fraction < 0:
        raise ValueError("inflation thresholds must be non-negative")

    output: list[Box] = []
    for index, raw_box in enumerate(boxes):
        if (not isinstance(raw_box, Sequence) or isinstance(raw_box, (str, bytes))
                or len(raw_box) != 4):
            raise ValueError(f"boxes[{index}] must contain four coordinates")
        box = tuple(float(value) for value in raw_box)
        if not all(math.isfinite(value) for value in box):
            raise ValueError(f"boxes[{index}] contains a non-finite coordinate")
        x1, y1, x2, y2 = box
        if x1 >= x2 or y1 >= y2:
            raise ValueError(f"boxes[{index}] is not a positive-area box")
        margin = max(float(min_inflation_css),
                     min(x2 - x1, y2 - y1) * float(inflation_fraction))
        inflated = (max(0.0, x1 - margin), max(0.0, y1 - margin),
                    min(page_width, x2 + margin), min(page_height, y2 + margin))
        if inflated[0] < inflated[2] and inflated[1] < inflated[3]:
            output.append(inflated)
    return output


def _require_array_dependencies() -> None:
    if np is None or ndimage is None:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "visual scope scoring requires numpy and scipy"
        ) from _ARRAY_IMPORT_ERROR


def _image_size(image: ImageInput) -> tuple[int, int]:
    if isinstance(image, Image.Image):
        return image.size
    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            return opened.size
    raise TypeError("image must be a PIL.Image, path string, or Path")


def _validate_dpr(dpr: float) -> float:
    if not isinstance(dpr, (int, float)) or isinstance(dpr, bool):
        raise ValueError("dpr must be a positive finite number")
    value = float(dpr)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("dpr must be a positive finite number")
    return value


def _analysis_geometry(image_size_px: tuple[int, int], dpr: float,
                       max_analysis_pixels: int) -> dict:
    """Choose an approximately CSS-resolution grid with a fixed memory cap."""

    width_px, height_px = image_size_px
    page_width_css, page_height_css = width_px / dpr, height_px / dpr
    # Never upsample a sub-1 DPR input.  Benchmark pages use DPR 1 or 2, for
    # which this is exactly one grid cell per CSS pixel before the safety cap.
    base_width = min(width_px, max(1, math.ceil(page_width_css)))
    base_height = min(height_px, max(1, math.ceil(page_height_css)))
    base_pixels = base_width * base_height
    scale = min(1.0, math.sqrt(max_analysis_pixels / base_pixels))
    analysis_width = max(1, math.floor(base_width * scale))
    analysis_height = max(1, math.floor(base_height * scale))
    while analysis_width * analysis_height > max_analysis_pixels:
        if analysis_height >= analysis_width and analysis_height > 1:
            analysis_height -= 1
        elif analysis_width > 1:
            analysis_width -= 1
        else:  # pragma: no cover - impossible for a positive limit
            break
    return {
        "image_size_px": image_size_px,
        "page_size_css": (page_width_css, page_height_css),
        "analysis_size_px": (analysis_width, analysis_height),
        "cell_size_css": (page_width_css / analysis_width,
                          page_height_css / analysis_height),
        "downsample_scale_from_physical": (
            analysis_width / width_px, analysis_height / height_px),
    }


def _prepare_rgb(image: ImageInput, target_size: tuple[int, int],
                 canvas_size_px: tuple[int, int]) -> Image.Image:
    """Reduce one image, then top-left pad it on the shared analysis canvas."""

    def prepare(opened: Image.Image) -> Image.Image:
        canvas_width, canvas_height = canvas_size_px
        target_width, target_height = target_size
        content_size = (
            max(1, min(target_width,
                       round(opened.width / canvas_width * target_width))),
            max(1, min(target_height,
                       round(opened.height / canvas_height * target_height))),
        )
        has_alpha = opened.mode in {"RGBA", "LA"} or "transparency" in opened.info
        if has_alpha:
            converted = opened.convert("RGBA")
            if converted.size != content_size:
                converted = converted.resize(content_size, Image.Resampling.BOX)
            content_background = Image.new("RGBA", content_size, "white")
            content = Image.alpha_composite(content_background, converted).convert("RGB")
        else:
            converted = opened if opened.mode == "RGB" else opened.convert("RGB")
            content = (converted.resize(content_size, Image.Resampling.BOX)
                       if converted.size != content_size else converted.copy())
        if content_size == target_size:
            return content
        canvas = Image.new("RGB", target_size, "white")
        canvas.paste(content, (0, 0))
        return canvas

    if isinstance(image, Image.Image):
        return prepare(image)
    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            return prepare(opened)
    raise TypeError("image must be a PIL.Image, path string, or Path")


def _filter_small_components(mask, min_area_css2: float,
                             cell_area_css2: float):
    """Filter components in-place with one reusable int32 label buffer."""

    minimum_cells = max(1, math.ceil(min_area_css2 / cell_area_css2))
    if not mask.any() or minimum_cells <= 1:
        return mask
    labels = np.empty(mask.shape, dtype=np.int32)
    count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8),
                          output=labels)
    if count:
        areas = np.bincount(labels.ravel())
        keep = areas >= minimum_cells
        keep[0] = False
        # ``out=mask`` avoids allocating another page-sized boolean array.
        np.take(keep, labels, out=mask)
    del labels
    return mask


def _significant_diff_with_geometry(
    before_image: ImageInput,
    fixed_image: ImageInput,
    *,
    dpr: float,
    channel_delta_threshold: int,
    min_component_area_css2: float,
    max_analysis_pixels: int,
):
    _require_array_dependencies()
    before_size = _image_size(before_image)
    fixed_size = _image_size(fixed_image)
    canvas_size = (max(before_size[0], fixed_size[0]),
                   max(before_size[1], fixed_size[1]))
    geometry = _analysis_geometry(canvas_size, _validate_dpr(dpr),
                                  max_analysis_pixels)
    geometry["before_image_size_px"] = before_size
    geometry["fixed_image_size_px"] = fixed_size
    target_size = geometry["analysis_size_px"]
    # Work in Pillow uint8 images.  This avoids two full-page int16 RGB arrays
    # and reduces DPR-2 pages to one sample per CSS pixel before NumPy is used.
    before = _prepare_rgb(before_image, target_size, canvas_size)
    fixed = _prepare_rgb(fixed_image, target_size, canvas_size)
    difference = ImageChops.difference(before, fixed)
    red, green, blue = difference.split()
    maximum = ImageChops.lighter(ImageChops.lighter(red, green), blue)
    changed = np.asarray(maximum, dtype=np.uint8) >= channel_delta_threshold
    del before, fixed, difference, red, green, blue, maximum
    cell_width_css, cell_height_css = geometry["cell_size_css"]
    _filter_small_components(changed, min_component_area_css2,
                             cell_width_css * cell_height_css)
    return changed, geometry


def significant_pixel_diff(
    before_image: ImageInput,
    fixed_image: ImageInput,
    *,
    dpr: float = 1.0,
    channel_delta_threshold: int = 16,
    min_component_area_css2: float = 4.0,
    max_analysis_pixels: int = 12_000_000,
):
    """Return a significant-change mask on an auditable CSS-scale grid."""

    _require_array_dependencies()
    if not 0 <= channel_delta_threshold <= 255:
        raise ValueError("channel_delta_threshold must be in [0, 255]")
    if min_component_area_css2 <= 0:
        raise ValueError("min_component_area_css2 must be positive")
    if type(max_analysis_pixels) is not int or max_analysis_pixels <= 0:
        raise ValueError("max_analysis_pixels must be a positive integer")
    return _significant_diff_with_geometry(
        before_image, fixed_image, dpr=dpr,
        channel_delta_threshold=channel_delta_threshold,
        min_component_area_css2=min_component_area_css2,
        max_analysis_pixels=max_analysis_pixels,
    )[0]


def _boxes_mask(boxes: Sequence[Box], analysis_size_px: tuple[int, int],
                page_size_css: tuple[float, float], *, output=None):
    _require_array_dependencies()
    width_px, height_px = analysis_size_px
    page_width_css, page_height_css = page_size_css
    mask = (np.zeros((height_px, width_px), dtype=bool) if output is None else output)
    if mask.shape != (height_px, width_px) or mask.dtype != np.bool_:
        raise ValueError("output mask has an incompatible shape or dtype")
    mask.fill(False)
    scale_x, scale_y = width_px / page_width_css, height_px / page_height_css
    for x1, y1, x2, y2 in boxes:
        left = max(0, min(width_px, int(math.floor(x1 * scale_x))))
        top = max(0, min(height_px, int(math.floor(y1 * scale_y))))
        right = max(0, min(width_px, int(math.ceil(x2 * scale_x))))
        bottom = max(0, min(height_px, int(math.ceil(y2 * scale_y))))
        if left < right and top < bottom:
            mask[top:bottom, left:right] = True
    return mask


def _json_boxes(boxes: Sequence[Box]) -> list[list[float]]:
    return [[round(value, 4) for value in box] for box in boxes]


def _permitted_xywh_to_xyxy(
    regions: object, page_size_css: Sequence[float]
) -> list[Box]:
    if not isinstance(regions, Sequence) or isinstance(regions, (str, bytes)):
        raise ValueError("permitted_regions_css must be an array")
    page_width, page_height = (float(value) for value in page_size_css)
    boxes: list[Box] = []
    for index, region in enumerate(regions):
        if (
            not isinstance(region, Sequence)
            or isinstance(region, (str, bytes))
            or len(region) != 4
        ):
            raise ValueError(f"permitted_regions_css[{index}] must be CSS xywh")
        values = [float(value) for value in region]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"permitted_regions_css[{index}] contains non-finite values")
        x, y, width, height = values
        if (
            x < 0
            or y < 0
            or width <= 0
            or height <= 0
            or x + width > page_width
            or y + height > page_height
        ):
            raise ValueError(f"permitted_regions_css[{index}] is outside the degraded page")
        boxes.append((x, y, x + width, y + height))
    return boxes


def _score_visual_scope_xyxy(
    before_image: ImageInput,
    fixed_image: ImageInput,
    boxes: Sequence[Box],
    *,
    dpr: float,
    scope_source: str,
    config: VisualScopeConfig | None = None,
) -> dict:
    """Score rendered changes against validated page-CSS xyxy regions."""

    _require_array_dependencies()
    config = config or DEFAULT_VISUAL_SCOPE_CONFIG
    config.validate()
    dpr = _validate_dpr(dpr)
    changed, geometry = _significant_diff_with_geometry(
        before_image, fixed_image, dpr=dpr,
        channel_delta_threshold=config.channel_delta_threshold,
        min_component_area_css2=config.min_component_area_css2,
        max_analysis_pixels=config.max_analysis_pixels,
    )
    scoring_size_px = geometry["image_size_px"]
    analysis_size_px = geometry["analysis_size_px"]
    page_size_css = geometry["page_size_css"]
    inflated = inflate_css_boxes(
        boxes, page_size_css,
        min_inflation_css=config.min_box_inflation_css,
        inflation_fraction=config.box_inflation_fraction,
    )
    raw_scope = _boxes_mask(boxes, analysis_size_px, page_size_css)
    tolerant_scope = _boxes_mask(inflated, analysis_size_px, page_size_css)
    work = np.empty_like(changed)

    page_pixels = int(changed.size)
    changed_pixels = int(np.count_nonzero(changed))
    np.logical_and(changed, raw_scope, out=work)
    inside_raw_pixels = int(np.count_nonzero(work))
    np.logical_and(changed, tolerant_scope, out=work)
    inside_tolerant_pixels = int(np.count_nonzero(work))
    nearby_pixels = inside_tolerant_pixels - inside_raw_pixels
    outside_pixels = changed_pixels - inside_tolerant_pixels
    page_allowance = math.ceil(page_pixels * config.max_outside_page_fraction)
    change_allowance = math.ceil(changed_pixels * config.max_outside_changed_fraction)
    # Both quantities are documented as upper bounds, so both must hold.
    # Taking ``max`` here would silently turn them into an OR rule and allow a
    # patch to exceed one of the advertised safety limits.
    allowed_outside_pixels = min(page_allowance, change_allowance)

    # A fixed CSS expansion is equivalent to a conservative rectangular
    # distance gate and avoids a page-sized float64 distance-transform array.
    far_scope_boxes = inflate_css_boxes(
        boxes, page_size_css,
        min_inflation_css=max(config.min_box_inflation_css,
                              config.far_distance_css),
        inflation_fraction=config.box_inflation_fraction,
    )
    _boxes_mask(far_scope_boxes, analysis_size_px, page_size_css,
                output=raw_scope)
    np.logical_not(raw_scope, out=work)
    np.logical_and(work, changed, out=work)
    far_labels = np.empty(changed.shape, dtype=np.int32)
    far_count = ndimage.label(
        work, structure=np.ones((3, 3), dtype=np.uint8), output=far_labels)
    far_components = []
    if far_count:
        areas = np.bincount(far_labels.ravel())
        cell_width_css, cell_height_css = geometry["cell_size_css"]
        cell_area_css2 = cell_width_css * cell_height_css
        objects = ndimage.find_objects(far_labels)
        qualifying = [label_id for label_id in range(1, far_count + 1)
                      if areas[label_id] * cell_area_css2
                      > config.far_component_area_css2]
        for label_id in qualifying:
            y_slice, x_slice = objects[label_id - 1]
            area_px = int(areas[label_id])
            area_css2 = area_px * cell_area_css2
            if area_css2 > config.far_component_area_css2:
                far_components.append({
                    "bbox_analysis_px": [x_slice.start, y_slice.start,
                                         x_slice.stop, y_slice.stop],
                    "bbox_css": [
                        round(x_slice.start * cell_width_css, 4),
                        round(y_slice.start * cell_height_css, 4),
                        round(x_slice.stop * cell_width_css, 4),
                        round(y_slice.stop * cell_height_css, 4),
                    ],
                    "area_analysis_pixels": area_px,
                    "area_css2": round(area_css2, 4),
                })
    del far_labels, work, raw_scope, tolerant_scope

    within_allowance = outside_pixels <= allowed_outside_pixels
    no_large_far_component = not far_components
    scope_pass = within_allowance and no_large_far_component
    return {
        "visual_scope_pass": bool(scope_pass),
        "visual_scope_reason": (
            "pass" if scope_pass else
            "outside_pixel_budget_and_far_component" if not within_allowance and far_components else
            "outside_pixel_budget" if not within_allowance else
            "far_component"
        ),
        "scope_source": scope_source,
        "scope_box_format": "page_css_xyxy",
        "permitted_regions_css": _json_boxes(boxes),
        "inflated_permitted_regions_css": _json_boxes(inflated),
        "before_image_size_px": list(geometry["before_image_size_px"]),
        "fixed_image_size_px": list(geometry["fixed_image_size_px"]),
        "scoring_image_size_px": list(scoring_size_px),
        "analysis_image_size_px": list(analysis_size_px),
        "analysis_cell_size_css": [round(value, 6)
                                   for value in geometry["cell_size_css"]],
        "analysis_downsample_scale_from_physical": [round(value, 8)
                                                     for value in geometry[
                                                         "downsample_scale_from_physical"]],
        "estimated_analysis_peak_bytes": page_pixels * 16,
        "scope_counts": {
            "count_unit": "analysis_grid_pixel_approximately_css_pixel",
            "page_pixels": page_pixels,
            "changed_pixels": changed_pixels,
            "inside_permitted_pixels": inside_raw_pixels,
            "nearby_tolerated_pixels": nearby_pixels,
            "outside_tolerant_scope_pixels": outside_pixels,
            "allowed_outside_pixels": allowed_outside_pixels,
            "outside_page_pixel_allowance": page_allowance,
            "outside_changed_pixel_allowance": change_allowance,
            "outside_allowance_policy": "both_limits_must_hold_minimum_budget",
            "outside_page_fraction": round(outside_pixels / page_pixels, 8),
            "outside_changed_fraction": round(
                outside_pixels / changed_pixels if changed_pixels else 0.0, 8),
        },
        "far_outside_components": far_components,
        "scope_thresholds": asdict(config),
    }


def frozen_gt_permitted_regions(gt_row: object) -> list[list[float]]:
    if not isinstance(gt_row, Mapping):
        raise ValueError("scoring GT must be an object")
    defects = gt_row.get("defects")
    if not isinstance(defects, list) or not defects:
        raise ValueError("scoring GT must contain defects")
    regions: list[list[float]] = []
    for defect_index, defect in enumerate(defects):
        if not isinstance(defect, Mapping):
            raise ValueError(f"defects[{defect_index}] must be an object")
        boxes = defect.get("degraded_bboxes_css")
        if not isinstance(boxes, list) or not boxes:
            raise ValueError(
                f"defects[{defect_index}].degraded_bboxes_css must be non-empty"
            )
        # 位移类缺陷的正确修复会把内容画回 baseline（"家"）位置，必须并集放行
        baseline_boxes = defect.get("baseline_bboxes_css")
        if baseline_boxes is None:
            baseline_boxes = []
        if not isinstance(baseline_boxes, list):
            raise ValueError(
                f"defects[{defect_index}].baseline_bboxes_css must be a list"
            )
        for key, group in (("degraded_bboxes_css", boxes),
                           ("baseline_bboxes_css", baseline_boxes)):
            for box_index, box in enumerate(group):
                if (
                    not isinstance(box, Sequence)
                    or isinstance(box, (str, bytes))
                    or len(box) != 4
                ):
                    raise ValueError(
                        f"defects[{defect_index}].{key}[{box_index}] "
                        "must be CSS xywh"
                    )
                values = [float(value) for value in box]
                if not all(math.isfinite(value) for value in values):
                    raise ValueError(
                        "frozen GT permitted region contains non-finite values"
                    )
                regions.append(values)
    return regions


def score_visual_scope(
    before_image: ImageInput,
    fixed_image: ImageInput,
    permitted_regions_css: object,
    *,
    dpr: float,
    config: VisualScopeConfig | None = None,
) -> dict:
    """Score changes against frozen-GT degraded page regions in CSS xywh."""

    dpr = _validate_dpr(dpr)
    before_width, before_height = _image_size(before_image)
    page_size_css = (before_width / dpr, before_height / dpr)
    validation_error = None
    try:
        boxes = _permitted_xywh_to_xyxy(permitted_regions_css, page_size_css)
        if not boxes:
            validation_error = "permitted_regions_css is empty"
    except (TypeError, ValueError) as error:
        boxes = []
        validation_error = str(error)
    result = _score_visual_scope_xyxy(
        before_image,
        fixed_image,
        boxes,
        dpr=dpr,
        scope_source="frozen_gt_degraded_bboxes_css",
        config=config,
    )
    result["permitted_region_input_format"] = "page_css_xywh"
    result["permitted_region_validation_error"] = validation_error
    if validation_error is not None:
        result["visual_scope_pass"] = False
        result["visual_scope_reason"] = "invalid_or_missing_permitted_regions"
    return result


def score_visual_scope_from_task2(
    before_image: ImageInput,
    fixed_image: ImageInput,
    task2_locations: Sequence[Mapping[str, object]],
    *,
    dpr: float,
    reference_image_size_px: Sequence[int] | None = None,
    task2_slice_offsets_px: Sequence[int] | None = None,
    config: VisualScopeConfig | None = None,
) -> dict:
    """Preserve the historical Task2-location scope scorer for legacy runners."""

    reference_size = (
        _positive_size(reference_image_size_px, "reference_image_size_px")
        if reference_image_size_px is not None
        else _image_size(before_image)
    )
    boxes = task2_locations_to_page_css_boxes(
        task2_locations,
        reference_size,
        dpr,
        slice_height_px=(config or DEFAULT_VISUAL_SCOPE_CONFIG).slice_height_px,
        slice_overlap_px=(config or DEFAULT_VISUAL_SCOPE_CONFIG).slice_overlap_px,
        slice_offsets_px=task2_slice_offsets_px,
    )
    result = _score_visual_scope_xyxy(
        before_image,
        fixed_image,
        boxes,
        dpr=dpr,
        scope_source="task2_predicted_locations_legacy",
        config=config,
    )
    result["reference_image_size_px"] = list(reference_size)
    result["task2_slice_offsets_px"] = (
        list(task2_slice_offsets_px)
        if task2_slice_offsets_px is not None
        else slice_offsets(
            reference_size[1],
            slice_height_px=(config or DEFAULT_VISUAL_SCOPE_CONFIG).slice_height_px,
            slice_overlap_px=(config or DEFAULT_VISUAL_SCOPE_CONFIG).slice_overlap_px,
        )
    )
    result["task2_slice_offsets_source"] = (
        "explicit_frozen_geometry"
        if task2_slice_offsets_px is not None
        else "derived_from_reference_height"
    )
    result["predicted_boxes_css"] = result["permitted_regions_css"]
    result["inflated_boxes_css"] = result["inflated_permitted_regions_css"]
    result["scope_counts"]["inside_predicted_pixels"] = result["scope_counts"][
        "inside_permitted_pixels"
    ]
    return result


BASELINE_FAST_PATH_CHANNEL_DELTA_THRESHOLD = 16
BASELINE_FAST_PATH_BUDGET_RATIO = 0.0001
BASELINE_FAST_PATH_BAND_ROWS = 512


def _load_rgb(image: ImageInput) -> Image.Image:
    """Return a fully loaded RGB image detached from any file handle."""

    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            return opened.convert("RGB")
    raise TypeError("image must be a PIL.Image, path string, or Path")


def baseline_restoration_diff(
    fixed_image: ImageInput,
    baseline_image: ImageInput,
) -> dict:
    """Compare the fixed render against the baseline render pixel by pixel.

    A pixel counts as different when the maximum absolute per-channel RGB
    delta exceeds 16; canvas area present in only one of the two images
    counts as different in full.  The fast path passes when the
    different-pixel count is within ``ceil(baseline_pixels * 0.0001)``, the
    same 0.01%-of-page budget the region-based scorer uses.  Rows are
    processed in fixed-height bands so full-page renders do not materialize
    as whole-page integer arrays at once.
    """

    _require_array_dependencies()
    fixed = _load_rgb(fixed_image)
    baseline = _load_rgb(baseline_image)
    overlap_width = min(fixed.width, baseline.width)
    overlap_height = min(fixed.height, baseline.height)
    union_pixels = (
        max(fixed.width, baseline.width) * max(fixed.height, baseline.height)
    )
    diff_pixels = union_pixels - overlap_width * overlap_height
    for y0 in range(0, overlap_height, BASELINE_FAST_PATH_BAND_ROWS):
        y1 = min(y0 + BASELINE_FAST_PATH_BAND_ROWS, overlap_height)
        fixed_band = np.asarray(
            fixed.crop((0, y0, overlap_width, y1)), dtype=np.int16
        )
        baseline_band = np.asarray(
            baseline.crop((0, y0, overlap_width, y1)), dtype=np.int16
        )
        diff_pixels += int(
            (
                np.abs(fixed_band - baseline_band).max(axis=2)
                > BASELINE_FAST_PATH_CHANNEL_DELTA_THRESHOLD
            ).sum()
        )
        del fixed_band, baseline_band
    page_pixels = baseline.width * baseline.height
    budget = math.ceil(page_pixels * BASELINE_FAST_PATH_BUDGET_RATIO)
    return {
        "baseline_fast_path_pass": bool(diff_pixels <= budget),
        "baseline_diff_pixels": diff_pixels,
        "baseline_page_pixels": page_pixels,
        "baseline_diff_budget": budget,
        "channel_delta_threshold": BASELINE_FAST_PATH_CHANNEL_DELTA_THRESHOLD,
        "budget_ratio": BASELINE_FAST_PATH_BUDGET_RATIO,
        "fixed_image_size_px": [fixed.width, fixed.height],
        "baseline_image_size_px": [baseline.width, baseline.height],
    }


def apply_baseline_restoration_fast_path(
    scope_result: Mapping[str, object],
    fixed_image: ImageInput,
    baseline_image: ImageInput,
) -> dict:
    """OR the baseline-restoration fast path into a region-check result.

    A fixed render matching the baseline within the pixel budget cannot
    contain a visual over-edit, so the scope check passes regardless of the
    degraded-bbox region verdict.  The fast path can only flip the combined
    verdict False -> True; the region-only verdict stays auditable under
    ``visual_scope_pass_region_check``.
    """

    result = dict(scope_result)
    fast_path = baseline_restoration_diff(fixed_image, baseline_image)
    result["baseline_restoration"] = fast_path
    result["visual_scope_pass_region_check"] = scope_result.get(
        "visual_scope_pass"
    )
    if fast_path["baseline_fast_path_pass"] and not result["visual_scope_pass"]:
        result["visual_scope_pass"] = True
        result["visual_scope_reason"] = "baseline_restoration_fast_path"
    return result


__all__ = [
    "BASELINE_FAST_PATH_BAND_ROWS",
    "BASELINE_FAST_PATH_BUDGET_RATIO",
    "BASELINE_FAST_PATH_CHANNEL_DELTA_THRESHOLD",
    "Box",
    "DEFAULT_VISUAL_SCOPE_CONFIG",
    "VisualScopeConfig",
    "apply_baseline_restoration_fast_path",
    "baseline_restoration_diff",
    "frozen_gt_permitted_regions",
    "inflate_css_boxes",
    "score_visual_scope",
    "score_visual_scope_from_task2",
    "significant_pixel_diff",
    "slice_offsets",
    "task2_locations_to_page_css_boxes",
]

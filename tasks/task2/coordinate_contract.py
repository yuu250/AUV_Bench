"""Versioned declared-coordinate contract for the Task 2 v8 evaluator.

Responses declare their coordinate space and box format; valid boxes are
converted to canonical normalized-1000 xyxy without guessing or repair.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from tasks.task2.metrics import parse_dimensions, parse_version


CONTRACT_VERSION = "task2-declared-coordinate-contract-v1"
RESPONSE_SCHEMA_VERSION = "task2-joint-json-schema-declared-coordinate-v1"
RESPONSE_SCHEMA_NAME = "task2_joint_diagnosis_declared_coordinate_v1"
COORDINATE_SPACES = ("normalized_1000", "slice_pixels")
BOX_FORMATS = ("xyxy", "xywh")


def _strict_json_object(text: str) -> dict | None:
    if not isinstance(text, str) or not text.strip():
        return None

    def reject_duplicate_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(text.strip(), object_pairs_hook=reject_duplicate_keys)
    except (json.JSONDecodeError, TypeError, ValueError, OverflowError, RecursionError):
        return None
    return value if isinstance(value, dict) else None


def response_format(n_slices: int, canvas_width: int, slice_height: int,
                    labels: Sequence[str]) -> dict:
    """Build the provider-expressible portion of the candidate contract."""
    if type(n_slices) is not int or n_slices <= 0:
        raise ValueError("n_slices must be a positive integer")
    if type(canvas_width) is not int or canvas_width <= 0:
        raise ValueError("canvas_width must be a positive integer")
    if type(slice_height) is not int or slice_height <= 0:
        raise ValueError("slice_height must be a positive integer")
    if isinstance(labels, (str, bytes)):
        raise ValueError("dimension labels must be a sequence of labels")
    labels = list(labels)
    if (not labels or any(not isinstance(label, str) for label in labels)
            or len(labels) != len(set(labels))):
        raise ValueError("dimension labels must be non-empty and unique")
    raw_coordinate_max = max(1000, canvas_width, slice_height)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": RESPONSE_SCHEMA_NAME,
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "degraded_version": {"type": "integer", "enum": [1, 2]},
                    "dimensions": {
                        "type": "array",
                        "items": {"type": "string", "enum": labels},
                        "maxItems": len(labels),
                    },
                    "coordinate_space": {
                        "type": "string",
                        "enum": list(COORDINATE_SPACES),
                    },
                    "box_format": {
                        "type": "string",
                        "enum": list(BOX_FORMATS),
                    },
                    "locations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "slice": {
                                    "type": "integer",
                                    "enum": list(range(1, n_slices + 1)),
                                },
                                "box": {
                                    "type": "array",
                                    "items": {
                                        "type": "integer",
                                        "minimum": 0,
                                        "maximum": raw_coordinate_max,
                                    },
                                    "minItems": 4,
                                    "maxItems": 4,
                                },
                            },
                            "required": ["slice", "box"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": [
                    "degraded_version",
                    "dimensions",
                    "coordinate_space",
                    "box_format",
                    "locations",
                ],
                "additionalProperties": False,
            },
        },
    }


def _dimensions_match_schema(values, en_to_internal: Mapping[str, str]) -> bool:
    return bool(
        isinstance(values, list)
        and len(values) <= len(en_to_internal)
        and all(isinstance(value, str) and value in en_to_internal for value in values)
    )


def _locations_match_schema(values, n_slices: int, raw_coordinate_max: int) -> bool:
    if not isinstance(values, list):
        return False
    for item in values:
        if not isinstance(item, dict) or set(item) != {"slice", "box"}:
            return False
        slice_number = item["slice"]
        box = item["box"]
        if type(slice_number) is not int or not 1 <= slice_number <= n_slices:
            return False
        if not isinstance(box, list) or len(box) != 4:
            return False
        if any(type(value) is not int or not 0 <= value <= raw_coordinate_max
               for value in box):
            return False
    return True


def convert_box(box: Sequence[int], coordinate_space: str, box_format: str,
                canvas_width: int, slice_height: int) -> list[float] | None:
    """Convert one declared box to normalized-1000 xyxy, or reject it."""
    if (coordinate_space not in COORDINATE_SPACES or box_format not in BOX_FORMATS
            or type(canvas_width) is not int or canvas_width <= 0
            or type(slice_height) is not int or slice_height <= 0
            or not isinstance(box, (list, tuple)) or len(box) != 4
            or any(type(value) is not int for value in box)):
        return None

    a, b, c, d = box
    if box_format == "xyxy":
        x1, y1, x2, y2 = a, b, c, d
    else:
        x1, y1, width, height = a, b, c, d
        if width <= 0 or height <= 0:
            return None
        x2, y2 = x1 + width, y1 + height

    if x1 < 0 or y1 < 0 or x1 >= x2 or y1 >= y2:
        return None
    if coordinate_space == "normalized_1000":
        if x2 > 1000 or y2 > 1000:
            return None
        return [float(x1), float(y1), float(x2), float(y2)]

    if x2 > canvas_width or y2 > slice_height:
        return None
    return [
        1000.0 * x1 / canvas_width,
        1000.0 * y1 / slice_height,
        1000.0 * x2 / canvas_width,
        1000.0 * y2 / slice_height,
    ]


def parse_declared_diagnosis(text: str, en_to_internal: Mapping[str, str],
                             n_slices: int, canvas_width: int,
                             slice_height: int) -> dict:
    """Parse and canonicalize one declared-coordinate joint diagnosis."""
    if type(n_slices) is not int or n_slices <= 0:
        raise ValueError("n_slices must be a positive integer")
    if type(canvas_width) is not int or canvas_width <= 0:
        raise ValueError("canvas_width must be a positive integer")
    if type(slice_height) is not int or slice_height <= 0:
        raise ValueError("slice_height must be a positive integer")
    empty = {
        "object_valid": False,
        "schema_valid": False,
        "dimensions_schema_valid": False,
        "locations_schema_valid": False,
        "coordinate_declaration_valid": False,
        "declared_coordinates_valid": False,
        "version": None,
        "dimensions": None,
        "locations": None,
        "source_locations": None,
        "coordinate_space": None,
        "box_format": None,
    }
    obj = _strict_json_object(text)
    required = {
        "degraded_version", "dimensions", "coordinate_space", "box_format",
        "locations",
    }
    if obj is None or set(obj) != required:
        return empty

    version = parse_version(json.dumps(
        {"degraded_version": obj["degraded_version"]}, ensure_ascii=False))
    dimensions_schema_valid = _dimensions_match_schema(
        obj["dimensions"], en_to_internal)
    dimensions = parse_dimensions(json.dumps(
        {"dimensions": obj["dimensions"]}, ensure_ascii=False), en_to_internal)
    coordinate_space = obj["coordinate_space"]
    box_format = obj["box_format"]
    coordinate_declaration_valid = bool(
        isinstance(coordinate_space, str)
        and coordinate_space in COORDINATE_SPACES
        and isinstance(box_format, str)
        and box_format in BOX_FORMATS
    )
    raw_coordinate_max = max(1000, canvas_width, slice_height)
    locations_schema_valid = _locations_match_schema(
        obj["locations"], n_slices, raw_coordinate_max)
    source_locations = (
        [{"slice": item["slice"], "box": list(item["box"])}
         for item in obj["locations"]]
        if locations_schema_valid else None
    )

    converted_locations = None
    declared_coordinates_valid = False
    if coordinate_declaration_valid and locations_schema_valid:
        converted_locations = []
        for item in obj["locations"]:
            converted = convert_box(
                item["box"], coordinate_space, box_format,
                canvas_width, slice_height,
            )
            if converted is None:
                converted_locations = None
                break
            converted_locations.append((item["slice"], converted))
        declared_coordinates_valid = converted_locations is not None

    return {
        "object_valid": True,
        "schema_valid": bool(
            version is not None
            and dimensions_schema_valid
            and coordinate_declaration_valid
            and locations_schema_valid
        ),
        "dimensions_schema_valid": dimensions_schema_valid,
        "locations_schema_valid": locations_schema_valid,
        "coordinate_declaration_valid": coordinate_declaration_valid,
        "declared_coordinates_valid": declared_coordinates_valid,
        "version": version,
        "dimensions": dimensions,
        "locations": converted_locations,
        "source_locations": source_locations,
        "coordinate_space": (
            coordinate_space if coordinate_declaration_valid else None),
        "box_format": box_format if coordinate_declaration_valid else None,
    }

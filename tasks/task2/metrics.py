"""Task 2 的纯解析与计分函数。

本模块不依赖图片/API，便于对正式指标做单元测试。所有 parser 都采用严格协议：
整段回复必须是唯一 JSON 对象；格式错误由调用方按失败计分，而不是猜测模型意图。
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence


Box = tuple[float, float, float, float]  # x1, y1, x2, y2


def _strict_json_object(text: str) -> dict | None:
    if not isinstance(text, str) or not text.strip():
        return None

    def reject_duplicate_keys(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError(f"duplicate key: {key}")
            obj[key] = value
        return obj

    try:
        obj = json.loads(text.strip(), object_pairs_hook=reject_duplicate_keys)
    except (json.JSONDecodeError, TypeError, ValueError, OverflowError, RecursionError):
        return None
    return obj if isinstance(obj, dict) else None


def parse_version(text: str) -> int | None:
    """解析 {"degraded_version": 1|2}；额外字段或类型不符均失败。"""
    obj = _strict_json_object(text)
    if obj is None or set(obj) != {"degraded_version"}:
        return None
    value = obj["degraded_version"]
    if type(value) is not int or value not in (1, 2):  # bool 是 int 子类，必须显式排除
        return None
    return value


def parse_dimensions(text: str, en_to_internal: Mapping[str, str]) -> set[str] | None:
    """解析 {"dimensions": [...]}；任一未知/重复标签使整份回答失败。"""
    obj = _strict_json_object(text)
    if obj is None or set(obj) != {"dimensions"}:
        return None
    values = obj["dimensions"]
    if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
        return None
    if len(values) != len(set(values)) or any(v not in en_to_internal for v in values):
        return None
    return {en_to_internal[v] for v in values}


def _dimensions_match_json_schema(values, en_to_internal: Mapping[str, str]) -> bool:
    """Validate the dimension constraints accepted by DashScope's schema subset."""
    return bool(
        isinstance(values, list)
        and len(values) <= len(en_to_internal)
        and all(isinstance(value, str) and value in en_to_internal for value in values)
    )


def _locations_match_json_schema(values, n_slices: int) -> bool:
    """Validate the part expressible by the frozen Task 2 JSON Schema."""
    if not isinstance(values, list):
        return False
    for item in values:
        if not isinstance(item, dict) or set(item) != {"slice", "box"}:
            return False
        slice_no = item["slice"]
        box = item["box"]
        if type(slice_no) is not int or not 1 <= slice_no <= n_slices:
            return False
        if not isinstance(box, list) or len(box) != 4:
            return False
        if any(type(value) is not int or not 0 <= value <= 1000 for value in box):
            return False
    return True


def parse_locations(text: str, n_slices: int) -> list[tuple[int, list[float]]] | None:
    """解析严格定位 JSON；任何坏条目都使整份回答失败，空 locations 合法。"""
    obj = _strict_json_object(text)
    if obj is None or set(obj) != {"locations"}:
        return None
    values = obj["locations"]
    if not _locations_match_json_schema(values, n_slices):
        return None

    out: list[tuple[int, list[float]]] = []
    for item in values:
        slice_no = item["slice"]
        box = item["box"]
        coords = [float(value) for value in box]
        x1, y1, x2, y2 = coords
        if x1 >= x2 or y1 >= y2:
            return None
        out.append((slice_no, coords))
    return out


def parse_joint_diagnosis(text: str, en_to_internal: Mapping[str, str],
                          n_slices: int) -> dict:
    """Parse the single-call Task 2 contract with field-level validity.

    The outer object must contain exactly the three required keys. Once that
    contract is satisfied, each value is validated independently so a malformed
    location list does not erase an otherwise valid version decision. This
    preserves the old three-question evaluator's per-stage failure accounting
    while removing its GT-gated follow-up calls.
    """
    empty = {
        "object_valid": False,
        "schema_valid": False,
        "dimensions_schema_valid": False,
        "locations_schema_valid": False,
        "version": None,
        "dimensions": None,
        "locations": None,
    }
    obj = _strict_json_object(text)
    required = {"degraded_version", "dimensions", "locations"}
    if obj is None or set(obj) != required:
        return empty

    # Reuse the frozen single-field validators so label, coordinate, duplicate,
    # and bool-vs-int policies cannot drift between staged and joint protocols.
    version = parse_version(json.dumps(
        {"degraded_version": obj["degraded_version"]}, ensure_ascii=False))
    dimensions = parse_dimensions(json.dumps(
        {"dimensions": obj["dimensions"]}, ensure_ascii=False), en_to_internal)
    dimensions_schema_valid = _dimensions_match_json_schema(
        obj["dimensions"], en_to_internal)
    locations_schema_valid = _locations_match_json_schema(
        obj["locations"], n_slices)
    locations = parse_locations(json.dumps(
        {"locations": obj["locations"]}, ensure_ascii=False), n_slices)
    return {
        "object_valid": True,
        # Duplicate dimensions and x1<x2/y1<y2 are local semantic constraints;
        # all other shape/type/range constraints are enforced by the provider.
        "schema_valid": bool(
            version is not None
            and dimensions_schema_valid
            and locations_schema_valid
        ),
        "dimensions_schema_valid": dimensions_schema_valid,
        "locations_schema_valid": locations_schema_valid,
        "version": version,
        "dimensions": dimensions,
        "locations": locations,
    }


def box_iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def merge_gt_regions(regions: Sequence[tuple[Box, Sequence[str]]],
                     overlap_threshold: float = 0.9) -> list[Box]:
    """把同一视觉区域上的多条注入规则合为一个 GT 区域。

    完全相同的框无条件去重；非完全相同的框只在 XPath 相同/互为祖先且高度
    重叠时合并。不能仅凭框相交就合并，因为父容器内可能有多个独立受害元素。
    """
    unique: dict[Box, set[str]] = {}
    for box, paths in regions:
        unique.setdefault(tuple(box), set()).update(path for path in paths if path)
    items = [(box, paths) for box, paths in unique.items()]
    parent = list(range(len(items)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def related(left: set[str], right: set[str]) -> bool:
        return any(a == b or a.startswith(b + "/") or b.startswith(a + "/")
                   for a in left for b in right)

    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if (related(items[i][1], items[j][1])
                    and box_iou(items[i][0], items[j][0]) >= overlap_threshold):
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri

    groups: dict[int, list[Box]] = {}
    for i, (box, _) in enumerate(items):
        groups.setdefault(find(i), []).append(box)
    return [(min(box[0] for box in boxes), min(box[1] for box in boxes),
             max(box[2] for box in boxes), max(box[3] for box in boxes))
            for boxes in groups.values()]


def clip_box(box: Box, rect: Box) -> Box | None:
    clipped = (max(box[0], rect[0]), max(box[1], rect[1]),
               min(box[2], rect[2]), min(box[3], rect[3]))
    return clipped if clipped[0] < clipped[2] and clipped[1] < clipped[3] else None


def clipped_iou_matrix(predictions: Sequence[tuple[Box, Box]],
                       gt_boxes: Sequence[Box]) -> list[list[float]]:
    """生成 IoU 矩阵。

    每个 prediction 是 (预测框, 该切片在整页中的可见矩形)。GT 先裁到该切片，
    因而跨切片的大元素不会因为不可见部分而被无端降低 IoU。
    """
    matrix: list[list[float]] = []
    for pred_box, slice_rect in predictions:
        row = []
        for gt_box in gt_boxes:
            visible_gt = clip_box(gt_box, slice_rect)
            row.append(box_iou(pred_box, visible_gt) if visible_gt else 0.0)
        matrix.append(row)
    return matrix


def max_visible_slice_numbers(gt_boxes: Sequence[Box],
                              slice_rects: Sequence[Box]) -> list[set[int]]:
    """每个 GT 只允许选择可见面积最大的切片；并列最大切片均允许。"""
    result = []
    for gt_box in gt_boxes:
        areas = []
        for rect in slice_rects:
            visible = clip_box(gt_box, rect)
            areas.append(((visible[2] - visible[0]) * (visible[3] - visible[1])
                          if visible else 0.0))
        largest = max(areas, default=0.0)
        allowed = {index + 1 for index, area in enumerate(areas)
                   if largest > 0 and math.isclose(area, largest, rel_tol=1e-9, abs_tol=1e-6)}
        result.append(allowed)
    return result


def maximum_match_count(scores: Sequence[Sequence[float]], n_gt: int,
                        threshold: float) -> int:
    """阈值图上的最大基数一对一匹配，避免重复框或一个框命中多个 GT。"""
    if not scores or n_gt <= 0:
        return 0
    adjacency = [sorted((j for j, score in enumerate(row[:n_gt]) if score >= threshold),
                        key=lambda j: row[j], reverse=True)
                 for row in scores]
    order = sorted(range(len(adjacency)),
                   key=lambda i: max((scores[i][j] for j in adjacency[i]), default=0.0),
                   reverse=True)
    gt_to_pred: dict[int, int] = {}

    def augment(pred_idx: int, seen: set[int]) -> bool:
        for gt_idx in adjacency[pred_idx]:
            if gt_idx in seen:
                continue
            seen.add(gt_idx)
            previous = gt_to_pred.get(gt_idx)
            if previous is None or augment(previous, seen):
                gt_to_pred[gt_idx] = pred_idx
                return True
        return False

    for pred_idx in order:
        augment(pred_idx, set())
    return len(gt_to_pred)


def localization_metrics(scores: Sequence[Sequence[float]], n_pred: int, n_gt: int,
                         threshold: float) -> dict[str, float | int]:
    tp = maximum_match_count(scores, n_gt, threshold)
    fp, fn = n_pred - tp, n_gt - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1}


def label_macro_metrics(records: Sequence[Mapping], dimensions: Sequence[str]) -> dict:
    """逐标签跨样本计算 TPR/FPR/J，再对标签等权宏平均。"""
    per_dim = {}
    for dim in dimensions:
        tp = fp = fn = tn = 0
        for record in records:
            gt = dim in set(record.get("gt_dims", []))
            pred = dim in set(record.get("attr_pred", []))
            if gt and pred:
                tp += 1
            elif gt:
                fn += 1
            elif pred:
                fp += 1
            else:
                tn += 1
        tpr = tp / (tp + fn) if tp + fn else None
        fpr = fp / (fp + tn) if fp + tn else None
        precision = tp / (tp + fp) if tp + fp else 0.0
        f1 = (2 * precision * tpr / (precision + tpr)
              if tpr is not None and precision + tpr else 0.0)
        j = tpr - fpr if tpr is not None and fpr is not None else None
        per_dim[dim] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                        "tpr": tpr, "fpr": fpr, "j": j,
                        "precision": precision, "f1": f1}

    valid_j = [m["j"] for m in per_dim.values() if m["j"] is not None]
    valid_f1 = [m["f1"] for m in per_dim.values() if m["tpr"] is not None]
    return {"per_dimension": per_dim,
            "n_valid_j_dimensions": len(valid_j),
            "macro_j": sum(valid_j) / len(valid_j) if valid_j else None,
            "macro_f1": sum(valid_f1) / len(valid_f1) if valid_f1 else None}

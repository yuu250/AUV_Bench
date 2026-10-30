#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compat_rules.py — M0b 兼容性扫描: 美学准则的正向断言（v2 · 严格对齐准则 PDF）

v2 相对 v1 的核心变更
---------------------
1. 引入**规范强度双轨**：PDF 措辞为"至少/不应/遵循一致/限制"→ hard；
   措辞为"为最佳/不宜/建议/原则上/警告/残差penalty/KL散度"→ soft。
   只有 hard 违规才可用于 mask 维度；soft 仅作元数据（"不够优"不等于"有缺陷"）。
2. 按 PDF 明确条文修正 5 处不符：
   - 行高：PDF 第7页承认 `1.5×字号`（国际无障碍标准）与 `字号+8` 两个标准并存 → 任一成立即 pass
   - 图片比例：PDF 第10页"排除通栏、全屏 banner 等" → 加豁免
   - UI 对比度：PDF 第1页认可分割线使用低对比中性色 → 排除分割线/细线
   - 8px 网格：PDF 第4页"元素**尺寸**和间距" → 补 width/height；改为残差 penalty
   - 表格对齐：PDF 第10页给的是坐标公式 x_left=C / x_right=W-C → 按坐标判定
3. 米勒定律检测对象修正：PDF"同类**内容区块**"指承载信息/分隔内容的容器，
   非列表项 → 限定为容器级 peer group；阈值按 PDF 自述区间（>5 soft，>9 hard）
4. 解除两对规则共线：CONSIST_CONTAINER_* 限定容器级 peer group，
   SHADOW_PEER_UNIFORM 覆盖全部 peer group
5. SHADOW_LEVEL_MAP 由 na 改为部分实现（PDF 第13-15页给了完整组件→层级映射表
   与精确 box-shadow 参数，可按标签/类名部分机器化）

返回格式
--------
每规则 → {n_app, n_hard, n_soft, ratio, ratio_soft, extra?}
  ratio      = n_hard / n_app   （硬违规率，用于 eligible 判定）
  ratio_soft = n_soft / n_app   （软偏离率，仅元数据）
  n_viol / ratio 保留以兼容既有下游

仍为 na 的规则：LAYOUT_FITTS_EDGE / LAYOUT_GUTENBERG（PDF 技术实现栏"待补充"
且依赖语义角色标注）；PIC_CLARITY 默认 pass（原图即原图）
"""
import json
import math
import re
import statistics
from collections import defaultdict, Counter

# ---------------- 颜色工具 ----------------

def parse_rgba(s):
    """'rgba(r,g,b,a)' -> (r,g,b,a) 归一化; 解析失败返回 None"""
    m = re.match(r"rgba?\(([^)]+)\)", s or "")
    if not m:
        return None
    parts = [p.strip() for p in m.group(1).split(",")]
    try:
        r, g, b = (float(parts[0]), float(parts[1]), float(parts[2]))
        a = float(parts[3]) if len(parts) > 3 else 1.0
    except (ValueError, IndexError):
        return None
    return (r / 255.0, g / 255.0, b / 255.0, a)


def parse_rgb(s):
    """仅取不透明色; alpha 过低视为无色"""
    c = parse_rgba(s)
    if not c or c[3] < 0.05:
        return None
    return c[:3]


def luminance(rgb):
    def f(c):
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = rgb
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)


def contrast(fg, bg):
    """WCAG 2.0 对比度比值。PDF 第2页阈值(4.5:1/3:1)即 WCAG 口径，
    故不采用 PDF 技术实现栏并列提到的 APCA 公式（其标注为待补充且与阈值不配套）。"""
    l1, l2 = luminance(fg), luminance(bg)
    if l1 < l2:
        l1, l2 = l2, l1
    return (l1 + 0.05) / (l2 + 0.05)


def hue(rgb):
    r, g, b = rgb
    mx, mn = max(rgb), min(rgb)
    if mx == mn:
        return None
    d = mx - mn
    if mx == r:
        h = (g - b) / d % 6
    elif mx == g:
        h = (b - r) / d + 2
    else:
        h = (r - g) / d + 4
    return h * 60


def saturation(rgb):
    mx, mn = max(rgb), min(rgb)
    return 0.0 if mx <= 0 else (mx - mn) / mx


# ---------------- CIEDE2000 色差（颜色类规则的唯一合格度量，计划书 §6.2） ----------------

def srgb_to_lab(rgb):
    """(r,g,b) ∈ [0,1] -> CIE Lab (D65)"""
    def f(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (f(c) for c in rgb)
    # sRGB -> XYZ (D65)
    x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
    # D65 白点
    xn, yn, zn = 0.95047, 1.0, 1.08883

    def g2(t):
        return t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116
    fx, fy, fz = g2(x / xn), g2(y / yn), g2(z / zn)
    return (116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz))


def delta_e_2000(rgb1, rgb2):
    """CIEDE2000 色差。JND≈1.0；计划书颜色规则可见性阈值 ΔE≥2.0"""
    from math import atan2, cos, degrees, exp, radians, sin, sqrt
    L1, a1, b1 = srgb_to_lab(rgb1)
    L2, a2, b2 = srgb_to_lab(rgb2)
    C1, C2 = sqrt(a1 * a1 + b1 * b1), sqrt(a2 * a2 + b2 * b2)
    Cb = (C1 + C2) / 2
    G = 0.5 * (1 - sqrt(Cb ** 7 / (Cb ** 7 + 25 ** 7))) if Cb > 0 else 0
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = sqrt(a1p * a1p + b1 * b1), sqrt(a2p * a2p + b2 * b2)

    def hp(ap, bp):
        if ap == 0 and bp == 0:
            return 0.0
        return degrees(atan2(bp, ap)) % 360
    h1p, h2p = hp(a1p, b1), hp(a2p, b2)
    dLp = L2 - L1
    dCp = C2p - C1p
    if C1p * C2p == 0:
        dhp = 0.0
    elif abs(h2p - h1p) <= 180:
        dhp = h2p - h1p
    else:
        dhp = h2p - h1p - 360 if h2p - h1p > 180 else h2p - h1p + 360
    dHp = 2 * sqrt(C1p * C2p) * sin(radians(dhp / 2))
    Lbp = (L1 + L2) / 2
    Cbp = (C1p + C2p) / 2
    if C1p * C2p == 0:
        hbp = h1p + h2p
    elif abs(h1p - h2p) <= 180:
        hbp = (h1p + h2p) / 2
    else:
        hbp = (h1p + h2p + 360) / 2 if h1p + h2p < 360 else (h1p + h2p - 360) / 2
    T = (1 - 0.17 * cos(radians(hbp - 30)) + 0.24 * cos(radians(2 * hbp))
         + 0.32 * cos(radians(3 * hbp + 6)) - 0.20 * cos(radians(4 * hbp - 63)))
    dTheta = 30 * __import__("math").exp(-(((hbp - 275) / 25) ** 2))
    RC = 2 * sqrt(Cbp ** 7 / (Cbp ** 7 + 25 ** 7)) if Cbp > 0 else 0
    SL = 1 + 0.015 * (Lbp - 50) ** 2 / sqrt(20 + (Lbp - 50) ** 2)
    SC = 1 + 0.045 * Cbp
    SH = 1 + 0.030 * Cbp * T
    RT = -sin(radians(2 * dTheta)) * RC
    return sqrt((dLp / SL) ** 2 + (dCp / SC) ** 2 + (dHp / SH) ** 2
                + RT * (dCp / SC) * (dHp / SH))


def px(v):
    if v is None:
        return None
    m = re.match(r"^(-?[\d.]+)px$", str(v).strip())
    return float(m.group(1)) if m else None


def weight_of(v):
    if v in ("normal", None):
        return 400
    if v == "bold":
        return 700
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return 400


# ---------------- 元素角色判定（PDF 术语落地） ----------------

INLINE_TAGS = {"span", "a", "em", "strong", "b", "i", "small", "code", "label",
               "br", "img", "svg", "path", "use", "sup", "sub", "abbr", "cite"}
# PDF 第16页"区块容器：在页面中起到承载信息，分隔内容的作用"
CONTAINER_TAGS = {"div", "section", "article", "aside", "main", "header",
                  "footer", "nav", "ul", "ol", "li", "form", "fieldset",
                  "table", "figure", "dl"}
# PDF 第13-14页阴影量表列出的组件
COMPONENT_TAGS = {"button", "input", "select", "textarea", "a", "label",
                  "progress", "td", "th", "table"}
COMPONENT_CLS = re.compile(r"(^|[-_ ])(btn|button|card|chip|tag|badge|pill)([-_ ]|$)", re.I)


def is_thin_line(e):
    """分割线 / 细线：PDF 第1页明确认可分割线使用低对比中性色"""
    w, h = e["bbox"][2], e["bbox"][3]
    return e["tag"] == "hr" or h <= 3 or w <= 3


def is_container_like(e):
    """容器级元素：视觉上构成'区块容器'（有背景/边框/阴影/圆角且有实体尺寸）"""
    if e["tag"] in INLINE_TAGS:
        return False
    d = e["style"].get("display", "") or ""
    if d.startswith("inline") and d != "inline-block":
        return False
    w, h = e["bbox"][2], e["bbox"][3]
    if w < 80 or h < 40:
        return False
    bg = parse_rgba(e["style"].get("backgroundColor"))
    has_bg = bool(bg and bg[3] > 0.02)
    bw = px(e["style"].get("borderTopWidth")) or 0
    has_shadow = (e["style"].get("boxShadow") or "none") != "none"
    radius = px(e["style"].get("borderRadius")) or 0
    return has_bg or bw >= 0.5 or has_shadow or radius > 0


def is_component_like(e):
    return e["tag"] in COMPONENT_TAGS or bool(COMPONENT_CLS.search(e["cls"] or ""))


class Page:
    def __init__(self, ei: dict):
        self.id = ei["id"]
        self.viewport = ei["viewport"]
        self.overflowX = ei["overflowX"]
        self.els = ei["elements"]
        self.groups = ei["peer_groups"]
        self.by_xpath = {e["xpath"]: e for e in self.els}
        self._parent = {}
        for e in self.els:
            self._parent[e["xpath"]] = (
                e["xpath"].rsplit("/", 1)[0] if "/" in e["xpath"][1:] else None)

    def parent_of(self, xpath):
        return self._parent.get(xpath)

    def eff_bg(self, e):
        """向上逐层合成背景色（含半透明 alpha 混合）；到根仍无不透明背景则以白垫底"""
        layers = []
        xp = e["xpath"]
        while xp:
            cur = self.by_xpath.get(xp)
            if cur:
                c = parse_rgba(cur["style"].get("backgroundColor"))
                if c and c[3] > 0.01:
                    layers.append(c)
                    if c[3] >= 0.999:
                        break
            xp = self.parent_of(xp)
        base = (1.0, 1.0, 1.0)
        for c in reversed(layers):
            a = c[3]
            base = tuple(c[i] * a + base[i] * (1 - a) for i in range(3))
        return base

    def group_members(self, g):
        return [self.by_xpath[x] for x in g["members"] if x in self.by_xpath]

    def container_groups(self):
        """仅保留成员为容器级的 peer group —— 对应 PDF'区块容器一致性原则'"""
        out = []
        for g in self.groups:
            ms = self.group_members(g)
            if len(ms) >= 3 and sum(1 for m in ms if is_container_like(m)) >= len(ms) * 0.8:
                out.append((g, ms))
        return out

    def component_groups(self):
        """仅保留成员为组件的 peer group —— 对应 PDF'组件一致性原则'"""
        out = []
        for g in self.groups:
            ms = self.group_members(g)
            if len(ms) >= 3 and sum(1 for m in ms if is_component_like(m)) >= len(ms) * 0.8:
                out.append((g, ms))
        return out

    def text_groups(self):
        """成员≥80%带文本的 peer group —— 文本类一致性规则（字号/字重/字体族）
        不需要元素是容器/组件，只需有文本（计划书 §3.6：该口径使 seed 覆盖
        58.5%→74.0%）。注入与断言同源：bank 选靶与 r_text_peer_* 都用此方法。"""
        out = []
        for g in self.groups:
            ms = self.group_members(g)
            if len(ms) >= 3 and sum(1 for m in ms if m["text"]) >= len(ms) * 0.8:
                out.append((g, ms))
        return out


def text_els(pg: Page):
    return [e for e in pg.els if e["text"]]


# ================= 色彩 Color =================

def r_color_contrast_text(pg):
    """PDF 第2页：普通文本≥4.5:1，大文本(≥24px 或 ≥18.66px bold)≥3:1 —— WCAG AA「至少」= hard"""
    app = hard = 0
    for e in text_els(pg):
        fg = parse_rgb(e["style"].get("color"))
        if not fg:
            continue
        fs = px(e["style"].get("fontSize")) or 16
        if fs < 6:  # 装饰性极小字，不构成阅读内容
            continue
        w = weight_of(e["style"].get("fontWeight"))
        large = fs >= 24 or (fs >= 18.66 and w >= 700)
        th = 3.0 if large else 4.5
        app += 1
        if contrast(fg, pg.eff_bg(e)) < th:
            hard += 1
    return app, hard, 0


def r_color_contrast_ui(pg):
    """PDF 第2页：图形和 UI 组件与背景对比度≥3:1 = hard。
    但 PDF 第1页明确"中性色…在背景、边框、**分割线**等场景中也非常常见"，
    故排除分割线/细线；且限定为 UI 组件（非任意带边框的盒子）。"""
    app = hard = 0
    for e in pg.els:
        if is_thin_line(e) or not is_component_like(e):
            continue
        bw = px(e["style"].get("borderTopWidth"))
        if not bw or bw < 0.5:
            continue
        bc = parse_rgb(e["style"].get("borderTopColor"))
        if not bc:
            continue
        app += 1
        if contrast(bc, pg.eff_bg(e)) < 3.0:
            hard += 1
    return app, hard, 0


# PDF 第2页：绿色"常用于环保品牌或**成功状态标识**"；红色"适合促销按钮或错误提示"
# → 限定为「状态标识」：短文本且整体即状态词，排除按钮/链接与句中出现
SUCCESS_WORDS = re.compile(r"^(成功|完成|已完成|已通过|通过|已支付|已发货|生效中|正常|success|done|completed?|active|passed)$", re.I)
ERROR_WORDS = re.compile(r"^(失败|错误|异常|已拒绝|驳回|超时|已停用|error|failed?|invalid|rejected)$", re.I)


def r_color_semantic(pg):
    """PDF 技术实现栏为「待补充」，本条属我方操作化。
    状态标识文本（成功/失败词）的色相必须落在对应语义带：
    ok=绿相[90,170]，err=红相[≥330 或 ≤25]。带外即违规 = hard：
    状态色语义是全球通用惯例（红=错/绿=成功），任何带外配色都会误导用户
    对状态的判读，可自动断言（G1/G4 口径）。bad = 违规元素 xpath。"""
    app = hard = 0
    bad = []
    for e in text_els(pg):
        t = e["text"].strip()
        if len(t) > 8 or e["tag"] in ("button", "a", "input"):
            continue  # PDF 指"状态标识"，非操作按钮/链接
        kind = "ok" if SUCCESS_WORDS.match(t) else ("err" if ERROR_WORDS.match(t) else None)
        if not kind:
            continue
        fg = parse_rgb(e["style"].get("color"))
        bg = parse_rgb(e["style"].get("backgroundColor"))
        rgb = fg or bg
        if not rgb or saturation(rgb) < 0.15:
            continue  # 中性灰无语义色，PDF 未要求
        h = hue(rgb)
        if h is None:
            continue
        app += 1
        ok_band = 90 <= h <= 170 if kind == "ok" else (h >= 330 or h <= 25)
        if not ok_band:
            hard += 1
            bad.append(e["xpath"])
    return app, hard, 0, {"bad_xpaths": bad}


def r_color_peer_accent(pg):
    """同级组件强调色一致（我方新增，源自 PDF 组件一致性原则）。
    仅统计**强调色**（有饱和度的色），中性色/透明差异不计 —— 否则斑马纹表格会假阳性。
    bad = 偏离组内主导强调色的成员（G1 xpath 口径用）。"""
    app = hard = 0
    bad = []
    for g, ms in pg.component_groups():
        mem_acc = []
        for m in ms:
            acc = None
            for prop in ("backgroundColor", "color"):
                c = parse_rgba(m["style"].get(prop))
                if c and c[3] > 0.05 and saturation(c[:3]) >= 0.25:
                    acc = tuple(round(x, 2) for x in c[:3])
                    break
            mem_acc.append((m, acc))
        accents = {a for _, a in mem_acc if a}
        if not accents:
            continue
        app += 1
        if len(accents) >= 2:
            hard += 1
            cnt = Counter(a for _, a in mem_acc if a)
            mode = cnt.most_common(1)[0][0]
            bad += [m["xpath"] for m, a in mem_acc if a and a != mode]
    return app, hard, 0, {"bad_xpaths": bad}


# ================= 间距 Spacing =================

def _grid_residual(v):
    """PDF 第4页：penalty = Σ(x_i % 8)，额外包含 4、12。返回该数值对网格的残差(px)"""
    a = abs(v)
    if a < 0.5:
        return 0.0
    if abs(a - 4) < 0.5 or abs(a - 12) < 0.5:
        return 0.0
    r = a % 8
    return round(min(r, 8 - r), 2)


SPACING_PROPS = ["marginTop", "marginBottom", "marginLeft", "marginRight",
                 "paddingTop", "paddingBottom", "paddingLeft", "paddingRight",
                 "rowGap", "columnGap"]


def r_spacing_grid_8(pg):
    """PDF 第4页技术实现给的是**残差求和 penalty**（连续量）而非二值判定 → soft。
    PDF 原文为"所有元素**尺寸和间距**"，故补 width/height（仅对容器级元素，
    文本行盒宽高由内容决定，非设计决策）。"""
    app = soft = 0
    total_res = 0.0
    for e in pg.els:
        for p in SPACING_PROPS:
            v = px(e["style"].get(p))
            if v is None or abs(v) < 0.5:
                continue  # 0 值不构成测点，避免分母虚高稀释残差
            app += 1
            res = _grid_residual(v)
            total_res += res
            if res > 0.5:
                soft += 1
        if is_container_like(e):
            for v in (e["bbox"][2], e["bbox"][3]):
                if v < 0.5:
                    continue
                app += 1
                res = _grid_residual(v)
                total_res += res
                if res > 0.5:
                    soft += 1
    extra = {"penalty": round(total_res, 1),
             "mean_res": round(total_res / app, 3) if app else None}
    return app, 0, soft, extra


def _rows_of(ms, axis):
    """把成员按行(横排)/列(竖排)聚簇，避免 flex-wrap 换行被当成间距不均"""
    if axis == 0:
        ms = sorted(ms, key=lambda m: (round(m["bbox"][1] / 8), m["bbox"][0]))
        key = lambda m: round(m["bbox"][1] / 8)
    else:
        ms = sorted(ms, key=lambda m: (round(m["bbox"][0] / 8), m["bbox"][1]))
        key = lambda m: round(m["bbox"][0] / 8)
    rows = defaultdict(list)
    for m in ms:
        rows[key(m)].append(m)
    return [v for v in rows.values() if len(v) >= 3]


def _gap_rows_scan(ms, axis):
    """行内间距一致性判据（SPACING_PEER_UNIFORM / CONSIST_CONTAINER_GAP 共用）。
    返回 (app, hard, bad_xpaths)：bad = 与中位间距偏差 >2px 的间隙两侧元素。"""
    app = hard = 0
    bad = []
    for row in _rows_of(ms, axis):
        pairs = list(zip(row, row[1:]))
        gaps = []
        for a, b in pairs:
            if axis == 0:
                gaps.append(b["bbox"][0] - (a["bbox"][0] + a["bbox"][2]))
            else:
                gaps.append(b["bbox"][1] - (a["bbox"][1] + a["bbox"][3]))
        pos = [x for x in gaps if x > -2]
        if len(pos) < 2:
            continue
        app += 1
        if max(pos) - min(pos) > 2:
            hard += 1
            med = statistics.median(pos)
            for (a, b), g in zip(pairs, gaps):
                if g > -2 and abs(g - med) > 2:
                    bad.extend([a["xpath"], b["xpath"]])
    return app, hard, bad


def r_spacing_peer_uniform(pg):
    """PDF 只在第16页对**区块容器**要求"互相间距保持一致"（该部分由
    CONSIST_CONTAINER_GAP 承担）。把同一要求推广到任意同级组
    （含文本项/内联元素）属我方扩展（计划书标 P+），PDF 无此条文 →
    basis=ours。升 hard：Bank 需对其出题，断言闸依赖 n_hard 增量口径。"""
    app = hard = 0
    bad = []
    for g in pg.groups:
        ms = pg.group_members(g)
        if len(ms) < 3:
            continue
        xs = [m["bbox"][0] for m in ms]
        ys = [m["bbox"][1] for m in ms]
        axis = 0 if (max(xs) - min(xs)) >= (max(ys) - min(ys)) else 1
        a, h, bx = _gap_rows_scan(ms, axis)
        app += a
        hard += h
        bad.extend(bx)
    return app, hard, 0, {"bad_xpaths": bad}


def _top_containers(pg):
    """页面主干上的大区块容器（PDF"区块容器"）"""
    cands = []
    for e in pg.els:
        depth = e["xpath"].count("/")
        if depth <= 3 and e["tag"] in ("section", "main", "div", "footer", "header") \
                and e["bbox"][3] >= 150 and e["bbox"][2] >= pg.viewport[0] * 0.5:
            cands.append(e)
    cands.sort(key=lambda e: e["bbox"][1])
    top = []
    for e in cands:
        inside = any(e["bbox"][1] >= o["bbox"][1] and
                     e["bbox"][1] + e["bbox"][3] <= o["bbox"][1] + o["bbox"][3] + 1
                     and e["xpath"] != o["xpath"] for o in top)
        if not inside:
            top.append(e)
    return top


def r_spacing_container_rhythm(pg):
    """PDF 第16页"区块容器…互相间距保持一致" = hard"""
    top = _top_containers(pg)
    if len(top) < 3:
        return 0, 0, 0
    gaps = [b["bbox"][1] - (a["bbox"][1] + a["bbox"][3]) for a, b in zip(top, top[1:])]
    pos = [g for g in gaps if g > -2]
    if len(pos) < 2:
        return 0, 0, 0
    mean = sum(pos) / len(pos)
    sd = math.sqrt(sum((g - mean) ** 2 for g in pos) / len(pos))
    bad = (mean > 0 and sd / mean > 0.35) or (mean == 0 and sd > 2)
    return 1, 1 if bad else 0, 0


# ================= 布局 Layout =================

def r_layout_alignment(pg):
    """PDF 第10页技术实现：x_left = C（左对齐）/ x_right = W - C（右对齐）恒定。
    PDF 该判据挂在"文本对齐"（表格列表）一行，横向顶边对齐 PDF 未给条文
    → 竖排左/右边缘恒定为 hard（PDF 明文）；横排顶边对齐限定容器级组且记 soft
    （文本项因字号/行高不同导致顶边差异属正常排版，非缺陷）。"""
    app = hard = soft = 0
    bad = []
    for g in pg.groups:
        ms = pg.group_members(g)
        if len(ms) < 3:
            continue
        xs = [m["bbox"][0] for m in ms]
        ys = [m["bbox"][1] for m in ms]
        rights = [m["bbox"][0] + m["bbox"][2] for m in ms]
        spread_x, spread_y = max(xs) - min(xs), max(ys) - min(ys)
        if spread_y > spread_x * 2:
            # 竖排：左边缘或右边缘恒定即视为对齐（PDF 明文判据）
            app += 1
            if min(spread_x, max(rights) - min(rights)) > 2:
                hard += 1
                # bad = 左右边缘同时偏离中位的成员（单个横移元素两者皆偏）
                mx, mr = statistics.median(xs), statistics.median(rights)
                for m in ms:
                    x0, _, w0, _ = m["bbox"]
                    if abs(x0 - mx) > 2 and abs(x0 + w0 - mr) > 2:
                        bad.append(m["xpath"])
        elif spread_x > spread_y * 2:
            # 横排顶边对齐：仅对容器级组，且属我方扩展 → soft
            if sum(1 for m in ms if is_container_like(m)) < len(ms) * 0.8:
                continue
            app += 1
            if spread_y > 2:
                soft += 1
    return app, hard, soft, {"bad_xpaths": bad}


def r_layout_miller_chunk(pg):
    """PDF 第5页："同类**内容区块**分布不应超过 5 个"。
    v2 修正检测对象：PDF 第16页界定"区块容器"为承载信息/分隔内容的容器，
    非列表项/链接/图标 → 仅统计容器级 peer group。
    阈值按 PDF 自述区间：>5 为"不应"(soft)，>9 为"极易出错"(hard)。"""
    app = hard = soft = 0
    for g, ms in pg.container_groups():
        app += 1
        n = len(ms)
        if n > 9:
            hard += 1
        elif n > 5:
            soft += 1
    return app, hard, soft


def r_layout_overflow(pg):
    """无溢出。计划书定为 regression-only，不作退化注入项。"""
    return 1, 1 if pg.overflowX else 0, 0


# ---------------- 古腾堡图表（容器级操作化子规则） ----------------
# PDF P5-6 原文："把最重要信息放在左上角…把最重要的操作放在右下角"。
# 整页 Z 动线需语义角色（已放弃），故操作化为两条容器级子规则，
# 角色用视觉重量锚定（primary=实底彩色/深色按钮；heading=最大字号文本），
# 无需 VLM。见 survey_gutenberg.py 普查与 demo_gutenberg.py 样例。

def _gut_is_button(e):
    if e["tag"] == "button":
        return True
    cls = (e.get("cls") or "")
    return e["tag"] in ("a", "div", "span") and \
        ("btn" in cls.split() or "button" in cls.split())


def _gut_is_primary(e):
    c = parse_rgba(e["style"].get("backgroundColor"))
    if not c or c[3] < 0.9:
        return False
    r, g, b = c[:3]
    return (max(r, g, b) - min(r, g, b)) > 40 or \
        (0.2126 * r + 0.7152 * g + 0.0722 * b) < 90


def gut_action_groups(pg):
    """同父、同行 ≥2 个带文本按钮 = 动作组（表单底栏/弹窗动作条）"""
    byp = defaultdict(list)
    for e in pg.els:
        x, y, w, h = e["bbox"]
        if w > 1 and h > 1 and _gut_is_button(e) and (e.get("text") or "").strip():
            byp[e["xpath"].rsplit("/", 1)[0]].append(e)
    out = []
    for ms in byp.values():
        rows = []
        for e in sorted(ms, key=lambda e: e["bbox"][0]):
            cy = e["bbox"][1] + e["bbox"][3] / 2
            for r in rows:
                if abs(r[0] - cy) < max(e["bbox"][3], r[1]) * 0.6:
                    r[2].append(e)
                    r[0] = (r[0] + cy) / 2
                    break
            else:
                rows.append([cy, e["bbox"][3], [e]])
        for _, _, row in rows:
            if len(row) >= 2:
                out.append(sorted(row, key=lambda e: e["bbox"][0]))
    return out


def gut_card_heads(pg):
    """(card, y 序文本列, heading 下标)：card=实底+圆角/阴影容器；
    heading=字号 ≥ 正文众数+4 且 ≥16 的最大字号文本"""
    out = []
    for card in pg.els:
        c = parse_rgba(card["style"].get("backgroundColor"))
        if not c or c[3] < 0.9:
            continue
        r = px(card["style"].get("borderRadius")) or 0
        sh = card["style"].get("boxShadow") not in (None, "none")
        x, y, w, h = card["bbox"]
        if not ((r > 0 or sh) and w >= 240 and h >= 140):
            continue
        texts = [e for e in pg.els
                 if (e.get("text") or "").strip()
                 and e["tag"] not in ("button", "a")
                 and e["xpath"] != card["xpath"]
                 and e["bbox"][0] >= x - 2 and e["bbox"][1] >= y - 2
                 and e["bbox"][0] + e["bbox"][2] <= x + w + 2
                 and e["bbox"][1] + e["bbox"][3] <= y + h + 2]
        if len(texts) < 2:
            continue
        texts.sort(key=lambda e: (e["bbox"][1], e["bbox"][0]))
        fss = [px(e["style"].get("fontSize")) or 0 for e in texts]
        mode = Counter(fss).most_common(1)[0][0]
        hi = max(range(len(texts)), key=lambda i: fss[i])
        if fss[hi] < max(mode + 4, 16):
            continue
        out.append((card, texts, hi))
    return out


def r_layout_gutenberg_action(pg):
    """古腾堡-操作：动作组里唯一实底主按钮应在阅读终点（最右）。"""
    app = hard = 0
    bad = []
    for row in gut_action_groups(pg):
        pris = [m for m in row if _gut_is_primary(m)]
        if len(pris) != 1:
            continue
        app += 1
        if pris[0] is not row[-1]:
            hard += 1
            bad.append(pris[0]["xpath"])
    return app, hard, 0, {"bad_xpaths": bad}


def r_layout_gutenberg_info(pg):
    """古腾堡-信息：卡片 heading 应为 y 序第一个文本（阅读起点）。"""
    app = hard = 0
    bad = []
    for _card, texts, hi in gut_card_heads(pg):
        app += 1
        if hi != 0:
            hard += 1
            bad.append(texts[hi]["xpath"])
    return app, hard, 0, {"bad_xpaths": bad}


# ---------------- 费茨定律（容器级操作化子规则） ----------------
# PDF P4 原则"目标越大且距离越近便越快到达；屏幕边缘是无限大目标"，
# P6 规则"桌面端中（鼠标）高频元素建议距离边缘 5%-12%，优先利用角落"
# （技术实现方法=待补充）。容器级操作化：动作组**整组**应贴容器右缘带：
#   gr = (card.right - row.right) / card.width ≤ 0.12 = 干净
# 锚点是整组而非主按钮（PDF 例子移动整个动作条）；角落是"优先"非"必须"，
# 故干净态只看右缘带，gb 仅记角落 subtype。选靶防误判洞见 survey_fitts.py。

GR_MAX = 0.12  # 右缘距离比干净态上限


def _fitts_parent_xpath(xp: str) -> str:
    return xp.rsplit("/", 1)[0] or "/"


def fitts_containers(pg):
    """实底 + 尺寸门槛 + 视觉可辨面板。
    圆角/阴影直接可辨；纯对比度路径额外要求与父元素底色不同
    （否则同色紧包裹 wrapper 会伪造"贴角"几何）"""
    body = pg.by_xpath.get("/html[1]/body[1]") or {}
    body_bg = parse_rgba(body.get("style", {}).get("backgroundColor")) \
        or (255, 255, 255, 1)
    out = []
    for c in pg.els:
        col = parse_rgba(c["style"].get("backgroundColor"))
        if not col or col[3] < 0.9:
            continue
        r = px(c["style"].get("borderRadius")) or 0
        sh = c["style"].get("boxShadow") not in (None, "none")
        x, y, w, h = c["bbox"]
        if w < 240 or h < 140:
            continue
        if r > 0 or sh:
            out.append(c)
            continue
        if col[:3] == body_bg[:3]:
            continue
        par = pg.by_xpath.get(_fitts_parent_xpath(c["xpath"])) or {}
        pcol = parse_rgba(par.get("style", {}).get("backgroundColor"))
        if pcol and pcol[3] >= 0.9 and pcol[:3] == col[:3]:
            continue  # 与父同色 → 视觉不可辨，不算面板
        out.append(c)
    return out


def fitts_action_rows(pg):
    """有容器的动作组几何明细（flex-wrap 折行已合并、表格重复容器/
    横跨度组已排除）。干净判定（gr≤GR_MAX）留给调用方。"""
    cards = fitts_containers(pg)
    rows = gut_action_groups(pg)
    # flex-wrap 折行合并：同父 + 水平重叠的相邻视觉行 = 一个动作组
    merged, used = [], [False] * len(rows)
    for i, r in enumerate(rows):
        if used[i]:
            continue
        used[i] = True
        cur = list(r)
        pi = _fitts_parent_xpath(r[0]["xpath"])
        x1 = max(e["bbox"][0] + e["bbox"][2] for e in r)
        for j in range(i + 1, len(rows)):
            if used[j] or _fitts_parent_xpath(rows[j][0]["xpath"]) != pi:
                continue
            jx0 = min(e["bbox"][0] for e in rows[j])
            if jx0 < x1:  # 水平重叠 → 折行
                used[j] = True
                cur += list(rows[j])
                x1 = max(x1, max(e["bbox"][0] + e["bbox"][2] for e in rows[j]))
        merged.append(cur)
    # 按最小包含容器聚合；排除重复组容器（表格/列表行）
    by_card = defaultdict(list)
    geo = {}
    for row in merged:
        x0 = min(e["bbox"][0] for e in row)
        y0 = min(e["bbox"][1] for e in row)
        x1 = max(e["bbox"][0] + e["bbox"][2] for e in row)
        y1 = max(e["bbox"][1] + e["bbox"][3] for e in row)
        cand = [c for c in cards
                if c["bbox"][0] <= x0 - 2 and c["bbox"][1] <= y0 - 2
                and c["bbox"][0] + c["bbox"][2] >= x1 + 2
                and c["bbox"][1] + c["bbox"][3] >= y1 + 2]
        if not cand:
            continue
        card = min(cand, key=lambda c: c["bbox"][2] * c["bbox"][3])
        cw, ch = card["bbox"][2], card["bbox"][3]
        geo[id(row)] = dict(row=row, card=card, x0=x0, x1=x1,
                            gr=(card["bbox"][0] + cw - x1) / cw,
                            gb=(card["bbox"][1] + ch - y1) / ch)
        by_card[card["xpath"]].append(id(row))
    repeat_cards = {cx for cx, ids in by_card.items() if len(ids) >= 3}

    out = []
    for g in geo.values():
        if g["card"]["xpath"] in repeat_cards:
            continue
        if (g["x1"] - g["x0"]) / g["card"]["bbox"][2] > 0.7:
            continue  # 整行横跨的组无"角落/边缘"语义
        gb = g["gb"]
        out.append({
            "card": g["card"]["xpath"], "card_bbox": g["card"]["bbox"],
            "n_btn": len(g["row"]),
            "gr": round(g["gr"], 3), "gb": round(gb, 3),
            "corner": "BR" if gb <= 0.30 else "TR" if gb >= 0.70 else "R",
            "row_x0": g["x0"], "row_x1": g["x1"],
            "row_xpaths": [e["xpath"] for e in g["row"]],
            "texts": [e.get("text", "")[:12] for e in g["row"]]})
    return out


def r_layout_fitts_edge(pg):
    """费茨边缘带：高频动作组整组右缘距离比 > 0.12 = 违规。"""
    app = hard = 0
    bad = []
    for g in fitts_action_rows(pg):
        app += 1
        if g["gr"] > GR_MAX:
            hard += 1
            bad += [xp for xp in g["row_xpaths"]]
    return app, hard, 0, {"bad_xpaths": bad}


# ================= 文本 Text =================

def r_text_lineheight_formula(pg):
    """PDF 第7页的推导链条是：
      ① 国际无障碍标准 line-height = font-size × 1.5；
      ② 但固定 1.5 倍在大字号时行高过大、信息割裂，故"基于**呼吸空间一致**，
         让不同字号之间的间距都保持相同"，得出「行高 = 字号 + n」；
      ③ n 取 8 是因为其在 14px/16px 常用字号处与 1.5 倍重合。
    可见 PDF 的**准则本体是"呼吸空间(lh-fs)页内一致"**，n=8 是推荐取值。
    故 hard 判据 = 既不满足两个标准、又偏离本页主导呼吸空间；
       soft 判据 = 呼吸空间一致但 n≠8（不够优，非缺陷）。
    v1 只认 n=8 且容差 2px，导致 95%+ 假违规。"""
    items = []
    for e in text_els(pg):
        lh = px(e["style"].get("lineHeight"))
        fs = px(e["style"].get("fontSize"))
        if lh is None or not fs or fs <= 0:
            continue  # 'normal' 由 UA 决定，PDF 未规范
        items.append((lh, fs))
    if not items:
        return 0, 0, 0
    # 本页主导呼吸空间：(lh - fs) 的众数（按 2px 分箱）
    bins = defaultdict(int)
    for lh, fs in items:
        bins[round((lh - fs) / 2) * 2] += 1
    dominant = max(bins.items(), key=lambda kv: kv[1])[0]
    app = hard = soft = 0
    for lh, fs in items:
        app += 1
        ok_plus8 = abs(lh - fs - 8) <= 2
        ok_ratio = abs(lh - 1.5 * fs) <= max(1.0, 0.1 * fs)
        ok_breath = abs((lh - fs) - dominant) <= 2  # 呼吸空间与本页一致
        if not (ok_plus8 or ok_ratio or ok_breath):
            hard += 1
        elif not ok_plus8:
            soft += 1
    return app, hard, soft, {"dominant_breath": dominant}


def _size_key(v):
    """字号归一到 0.5px 栅格，避免 15.008/15.0 被当成两个字阶"""
    return None if not v else round(v * 2) / 2


def r_text_scale_count(pg):
    """PDF 第7页"字阶**不宜**超过 5 种"。Bank 口径：超 5 阶 = hard
    （字阶数是纯计数判据，可自动断言）。bad = 字号为"孤阶"（全页仅 1 个
    元素使用）的元素——注入产生的 off-scale 字号必然孤立，G1 xpath 口径用。"""
    els = [e for e in text_els(pg)]
    keys = [_size_key(px(e["style"].get("fontSize"))) for e in els]
    sizes = Counter(k for k in keys if k)
    if len(sizes) <= 5:
        return 1, 0, 0, {"n_scales": len(sizes)}
    bad = [e["xpath"] for e, k in zip(els, keys) if k and sizes[k] == 1]
    return 1, 1, 0, {"n_scales": len(sizes), "bad_xpaths": bad}


def r_text_peer_fontsize(pg):
    """PDF 第17页"同类组件确保参数一致" = hard
    v2.1：分组口径从 component_groups 切为 text_groups（与 bank 选靶同源，
    计划书 §3.6/§2.3）。"""
    app = hard = 0
    bad = []
    for g, ms in pg.text_groups():
        sizes = {_size_key(px(m["style"].get("fontSize"))) for m in ms if m["text"]}
        sizes.discard(None)
        if not sizes:
            continue
        app += 1
        if len(sizes) >= 2:
            hard += 1
            bad += [m["xpath"] for m in ms if m["text"]]
    return app, hard, 0, {"bad_xpaths": bad}


OPTIMAL_WEIGHTS = {400, 500, 600}


def r_text_weight_set(pg):
    """PDF 第8页"字重以 400、500、600 区分**为最佳**" → soft（v1 误作硬集合，
    导致 700 bold 标题这一普遍且正当的用法被判违规，88.6% 假违规）"""
    app = soft = 0
    dist = defaultdict(int)
    for e in text_els(pg):
        w = weight_of(e["style"].get("fontWeight"))
        app += 1
        dist[w] += 1
        if w not in OPTIMAL_WEIGHTS:
            soft += 1
    return app, 0, soft, {"weights": dict(sorted(dist.items()))}


def r_text_weight_count(pg):
    """PDF 第8页"单应用字重类型**不应**超过 3 种" = hard"""
    ws = {weight_of(e["style"].get("fontWeight")) for e in text_els(pg)}
    return 1, 1 if len(ws) > 3 else 0, 0, {"n_weights": len(ws)}


# 归一化字体族：剥离通用族与系统别名，排除图标字体
GENERIC_FAMILIES = {"sans-serif", "serif", "monospace", "cursive", "fantasy",
                    "system-ui", "ui-sans-serif", "ui-serif", "ui-monospace",
                    "ui-rounded", "-apple-system", "blinkmacsystemfont",
                    "inherit", "initial", "unset", "emoji", "math", "fangsong"}
ICON_FAMILY = re.compile(r"(icon|fontawesome|material|glyph)", re.I)


def _norm_family(fam):
    for part in (fam or "").split(","):
        f = part.strip().strip('"').strip("'").lower()
        if not f or f in GENERIC_FAMILIES or ICON_FAMILY.search(f):
            continue
        return f
    return None


def r_text_font_family_count(pg):
    """PDF 第6页"**原则上**同一页面…不超过中英文各 1 款"。
    Bank 口径升级 hard：字体栈是品牌视觉规范最直接的外显，第 3 款字体出现 =
    混入另一套规范来源（与 P17 组件一致性同源），可自动断言。
    bad = 少数款（按使用量 top2 之外）元素 xpath。"""
    els = [e for e in text_els(pg)]
    keys = [_norm_family(e["style"].get("fontFamily")) for e in els]
    cnt = Counter(k for k in keys if k)
    if len(cnt) <= 2:
        return 1, 0, 0, {"families": sorted(cnt)[:8], "n_families": len(cnt)}
    top2 = {f for f, _ in cnt.most_common(2)}
    bad = [e["xpath"] for e, k in zip(els, keys) if k and k not in top2]
    return 1, 1, 0, {"families": sorted(cnt)[:8], "n_families": len(cnt),
                     "bad_xpaths": bad}


CJK_RE = re.compile(r"[\u3000-\u9fff\uff00-\uffef]")


def r_text_line_length(pg):
    """PDF 第9页"单行文本数**不宜**超过 85 个字符"，技术实现"超限触发**警告**" → soft。
    v2 修正字宽系数：v1 固定 0.55（英文窄字）会把中文页面字符数高估约 1.8 倍。"""
    app = soft = 0
    for e in text_els(pg):
        t = e["text"]
        if len(t) < 40:
            continue
        fs = px(e["style"].get("fontSize")) or 16
        w = e["bbox"][2]
        if w <= 0 or fs <= 0:
            continue
        cjk_ratio = len(CJK_RE.findall(t)) / len(t)
        per_char = fs * (cjk_ratio * 1.0 + (1 - cjk_ratio) * 0.52)
        app += 1
        if w / per_char > 85:
            soft += 1
    return app, 0, soft


NUM_RE = re.compile(r"^[+-]?[¥$€£]?\s*[\d,.]+\s*[%‰]?[a-zA-Z]{0,3}$")


def r_text_table_align(pg):
    """PDF 第9页"表格列表中，文本遵循左对齐，数字阿拉伯文等遵循右对齐" = hard。
    双判据：① CSS 对齐口径——数据格(td)对齐方向违反列型（文本列非左/数字列非右），
    th 表头豁免（居中表头是常见设计惯例）；② 坐标口径（PDF 第10页技术实现）——
    CSS 全合规但同列格子边缘不齐 = 结构错位。bad = 违规格（G1 xpath 口径用）。"""
    cells = [e for e in pg.els if e["tag"] in ("td", "th") and e["text"]]
    cols = defaultdict(list)
    for e in cells:
        m = re.search(r"/t[dh]\[(\d+)\]$", e["xpath"])
        if not m:
            continue
        tr = e["xpath"].rsplit("/", 1)[0]
        tbl = tr.rsplit("/", 1)[0]
        cols[(tbl, int(m.group(1)))].append(e)
    app = hard = 0
    bad = []
    for cs in cols.values():
        if len(cs) < 3:
            continue
        n_num = sum(1 for c in cs if NUM_RE.match(c["text"].strip()))
        is_num_col = n_num >= len(cs) * 0.7
        app += 1
        ok_set = {"right", "end"} if is_num_col else {"left", "start", "justify"}
        bad_css = [c for c in cs if c["tag"] == "td"
                     and c["style"].get("textAlign", "start") not in ok_set]
        if bad_css:
            hard += 1
            bad += [c["xpath"] for c in bad_css]
            continue
        aligns = {c["style"].get("textAlign", "start") for c in cs}
        if is_num_col:
            if aligns <= {"right", "end"}:
                continue
            rights = [c["bbox"][0] + c["bbox"][2] for c in cs]
            if max(rights) - min(rights) > 2:
                hard += 1
                mode_r = Counter(round(r, 1) for r in rights).most_common(1)[0][0]
                bad += [c["xpath"] for c in cs
                        if abs((c["bbox"][0] + c["bbox"][2]) - mode_r) > 2]
        else:
            if aligns <= {"left", "start", "justify"}:
                continue
            lefts = [c["bbox"][0] for c in cs]
            if max(lefts) - min(lefts) > 2:
                hard += 1
                mode_l = Counter(round(l, 1) for l in lefts).most_common(1)[0][0]
                bad += [c["xpath"] for c in cs if abs(c["bbox"][0] - mode_l) > 2]
    return app, hard, 0, {"bad_xpaths": bad}


# ================= 图像 Picture =================

# 题库 v2：PIC_ASPECT_RATIO 已整条下线（人工目检压扁后视觉仍合理，构念不成立），
# STANDARD_RATIOS 仅保留供 compat_case_detail.py 等 v1 档案脚本 import 兼容。
STANDARD_RATIOS = (1.0, 4 / 3, 16 / 9, 3 / 4, 9 / 16)


# ================= 阴影 Shadow =================

def shadow_level(bs):
    """按 PDF 第15页 fusion 量表反推层级：
    L0 无投影 / L1 blur 3px / L2 blur 4px / L3 blur 30px"""
    if not bs or bs == "none":
        return 0
    nums = [abs(float(x)) for x in re.findall(r"(-?[\d.]+)px", bs)]
    if not nums:
        return 0
    m = max(nums)
    if m <= 0.5:
        return 0
    if m <= 3.5:
        return 1
    if m <= 6.0:
        return 2
    return 3


# PDF 第13-14页组件→层级映射表中可由标签/类名判定的部分
L0_TAGS = {"button", "input", "select", "textarea", "table", "td", "th",
           "a", "progress", "label", "hr"}
L1_TAGS = {"nav"}
CLS_L2 = re.compile(r"(dropdown|drawer|popover|submenu)", re.I)
CLS_L3 = re.compile(r"(modal|dialog|toast|notification|popup)", re.I)


def _expected_level(e):
    cls = e["cls"] or ""
    if e["tag"] == "dialog" or CLS_L3.search(cls):
        return 3
    if CLS_L2.search(cls):
        return 2
    if e["tag"] in L1_TAGS:
        return 1
    if e["tag"] in L0_TAGS:
        return 0
    return None


def r_shadow_level_map(pg):
    """PDF 第13-15页给了完整组件→层级映射与精确参数（按钮/导航=L0-L1、
    下拉/抽屉=L2、弹窗/提示=L3），映射表是明确规范 → 跨级即 hard
    （Bank 口径：v2 曾把跨 1 级记 soft，升级 hard 理由同 COLOR_SEMANTIC——
    组件阴影层级是 Design System 令牌，跨级=用错令牌，可自动断言）。
    v2 先按标签/类名可判定的子集落地（组件识别到位后再扩）。
    bad = 跨级元素 xpath。"""
    app = hard = 0
    bad = []
    for e in pg.els:
        exp = _expected_level(e)
        if exp is None:
            continue
        app += 1
        d = abs(shadow_level(e["style"].get("boxShadow")) - exp)
        if d >= 1:
            hard += 1
            bad.append(e["xpath"])
    return app, hard, 0, {"bad_xpaths": bad}


def r_shadow_peer_uniform(pg):
    """同级元素阴影一致（覆盖全部 peer group）= hard"""
    app = hard = 0
    bad = []
    for g in pg.groups:
        ms = pg.group_members(g)
        if len(ms) < 3:
            continue
        levels = {shadow_level(m["style"].get("boxShadow")) for m in ms}
        app += 1
        if len(levels) >= 2:
            hard += 1
            bad += [m["xpath"] for m in ms]
    return app, hard, 0, {"bad_xpaths": bad}


# ================= 一致性 Consistency =================

def r_consist_container_radius(pg):
    """PDF 第16页"区块容器需遵循圆角度数…保持一致" = hard。
    v2 限定容器级 peer group（v1 用任意 peer group，检测对象与 PDF 不符）。"""
    app = hard = 0
    bad = []
    for g, ms in pg.container_groups():
        rs = {m["style"].get("borderRadius", "0px") for m in ms}
        app += 1
        if len(rs) >= 2:
            hard += 1
            bad += [m["xpath"] for m in ms]
    return app, hard, 0, {"bad_xpaths": bad}


def r_consist_container_shadow(pg):
    """PDF 第16页"区块容器需遵循…阴影…保持一致" = hard。
    v2 限定容器级 peer group —— 与 SHADOW_PEER_UNIFORM 解除共线。"""
    app = hard = 0
    for g, ms in pg.container_groups():
        levels = {shadow_level(m["style"].get("boxShadow")) for m in ms}
        app += 1
        if len(levels) >= 2:
            hard += 1
    return app, hard, 0


def r_consist_container_gap(pg):
    """PDF 第16页"区块容器需遵循…互相间距保持一致" = hard。
    v2 改为容器级 peer group 的组内间距一致 —— 与 SPACING_CONTAINER_RHYTHM
    （页面主干区块的垂直节奏）解除共线。"""
    app = hard = 0
    bad = []
    for g, ms in pg.container_groups():
        xs = [m["bbox"][0] for m in ms]
        ys = [m["bbox"][1] for m in ms]
        axis = 0 if (max(xs) - min(xs)) >= (max(ys) - min(ys)) else 1
        a, h, bx = _gap_rows_scan(ms, axis)
        app += a
        hard += h
        bad.extend(bx)
    return app, hard, 0, {"bad_xpaths": bad}


def r_consist_component_param(pg):
    """PDF 第17页"同类组件确保参数一致" = hard。v2 限定组件级 peer group。"""
    app = hard = 0
    for g, ms in pg.component_groups():
        sigs = set()
        for m in ms:
            s = m["style"]
            sigs.add((s.get("paddingTop"), s.get("paddingLeft"),
                      _size_key(px(s.get("fontSize"))),
                      s.get("borderRadius"),
                      (s.get("backgroundColor") or "")[:20]))
        app += 1
        if len(sigs) >= 2:
            hard += 1
    return app, hard, 0


# ---------------- 规则注册表 ----------------
# kind : abs=绝对阈值 / rel=组内一致性 / na=后置 / pass=默认合规
# sev  : hard=PDF 措辞为"至少/不应/遵循一致/限制"，可用于 mask 维度
#        soft=PDF 措辞为"为最佳/不宜/建议/原则上/警告/残差/KL"，仅元数据
# basis: pdf=PDF 明文给出判据 / pdf* =PDF 标"待补充"由我方操作化 / ours=我方新增
RULES = {
    "COLOR_CONTRAST_TEXT":      ("abs", "hard", "pdf",  r_color_contrast_text),
    "COLOR_CONTRAST_UI":        ("abs", "hard", "pdf",  r_color_contrast_ui),
    "COLOR_PROPORTION":         ("abs", "soft", "pdf",  None),   # 像素统计, compat_scan 补
    "COLOR_SEMANTIC":           ("abs", "hard", "pdf*", r_color_semantic),
    "COLOR_PEER_ACCENT":        ("rel", "hard", "ours", r_color_peer_accent),
    "SPACING_GRID_8":           ("abs", "soft", "pdf",  r_spacing_grid_8),
    "SPACING_PEER_UNIFORM":     ("rel", "hard", "ours", r_spacing_peer_uniform),
    "SPACING_CONTAINER_RHYTHM": ("rel", "hard", "pdf",  r_spacing_container_rhythm),
    "LAYOUT_ALIGNMENT":         ("rel", "hard", "pdf",  r_layout_alignment),
    "LAYOUT_FITTS_EDGE":        ("na",  "soft", "pdf*", None),   # 待补充 + 需语义角色
    "LAYOUT_MILLER_CHUNK":      ("abs", "soft", "pdf*", r_layout_miller_chunk),
    "LAYOUT_GUTENBERG":         ("na",  "soft", "pdf*", None),   # 待补充 + 需语义角色
    "LAYOUT_OVERFLOW":          ("abs", "hard", "ours", r_layout_overflow),  # regression-only
    "TEXT_LINEHEIGHT_FORMULA":  ("abs", "hard", "pdf",  r_text_lineheight_formula),
    "TEXT_SCALE_COUNT":         ("abs", "hard", "pdf",  r_text_scale_count),
    "TEXT_PEER_FONTSIZE":       ("rel", "hard", "pdf",  r_text_peer_fontsize),
    "TEXT_WEIGHT_SET":          ("abs", "soft", "pdf",  r_text_weight_set),
    "TEXT_WEIGHT_COUNT":        ("abs", "hard", "pdf",  r_text_weight_count),
    "TEXT_PEER_WEIGHT":         ("rel", "hard", "pdf",  None),   # = TEXT_PEER_FONTSIZE 同族, 下方单独定义
    "TEXT_FONT_FAMILY_COUNT":   ("abs", "hard", "pdf",  r_text_font_family_count),
    "TEXT_LINE_LENGTH":         ("abs", "soft", "pdf",  r_text_line_length),
    "TEXT_TABLE_ALIGN":         ("abs", "hard", "pdf",  r_text_table_align),
    "PIC_CLARITY":              ("pass", "hard", "pdf", None),   # 原图即原图
    "SHADOW_LEVEL_MAP":         ("abs", "hard", "pdf",  r_shadow_level_map),
    "SHADOW_PEER_UNIFORM":      ("rel", "hard", "pdf",  r_shadow_peer_uniform),
    "CONSIST_CONTAINER_RADIUS": ("rel", "hard", "pdf",  r_consist_container_radius),
    "CONSIST_CONTAINER_SHADOW": ("rel", "hard", "pdf",  r_consist_container_shadow),
    "CONSIST_CONTAINER_GAP":    ("rel", "hard", "pdf",  r_consist_container_gap),
    "CONSIST_COMPONENT_PARAM":  ("rel", "hard", "pdf",  r_consist_component_param),
}

# regression-only：不作为退化注入项（计划书 §2.3 LAYOUT_OVERFLOW）
REGRESSION_ONLY = {"LAYOUT_OVERFLOW"}

# COLOR_PROPORTION 的 KL 阈值（PDF 未给阈值，需按语料分位数定标）
COLOR_PROPORTION_KL_MAX = 0.35


def r_text_peer_weight(pg):
    """PDF 第17页"同类组件确保参数一致"（字重）= hard
    v2.1：分组口径从 component_groups 切为 text_groups（与 bank 选靶同源）。"""
    app = hard = 0
    bad = []
    for g, ms in pg.text_groups():
        ws = {weight_of(m["style"].get("fontWeight")) for m in ms if m["text"]}
        if not ws:
            continue
        app += 1
        if len(ws) >= 2:
            hard += 1
            bad += [m["xpath"] for m in ms if m["text"]]
    return app, hard, 0, {"bad_xpaths": bad}


RULES["TEXT_PEER_WEIGHT"] = ("rel", "hard", "pdf", r_text_peer_weight)

# 古腾堡容器级子规则（父项 LAYOUT_GUTENBERG 仍保留 na 占位，代表整页 Z 动线）
RULES["LAYOUT_GUTENBERG_ACTION"] = ("rel", "hard", "pdf", r_layout_gutenberg_action)
RULES["LAYOUT_GUTENBERG_INFO"] = ("rel", "hard", "pdf", r_layout_gutenberg_info)
# 费茨容器级子规则（父项 LAYOUT_FITTS_EDGE 原 na 占位，2026-08 容器级收编）
RULES["LAYOUT_FITTS_EDGE"] = ("rel", "hard", "pdf*", r_layout_fitts_edge)


def scan_seed(ei: dict) -> dict:
    """对一条 seed 跑全部规则。
    返回 {rule_id: {n_app, n_hard, n_soft, ratio, ratio_soft, sev, basis, extra?}}"""
    pg = Page(ei)
    out = {}
    for rid, (kind, sev, basis, fn) in RULES.items():
        if kind == "na":
            out[rid] = {"na": "need_semantic", "sev": sev, "basis": basis}
            continue
        if kind == "pass":
            out[rid] = {"na": "default_pass", "sev": sev, "basis": basis}
            continue
        if fn is None:
            continue  # COLOR_PROPORTION 由像素模块补
        try:
            res = fn(pg)
        except Exception as e:  # noqa: BLE001
            out[rid] = {"na": f"error:{type(e).__name__}", "sev": sev, "basis": basis}
            continue
        app, hard, soft = res[0], res[1], res[2]
        rec = {"n_app": app, "n_hard": hard, "n_soft": soft,
               "n_viol": hard,  # 兼容既有下游
               "ratio": round(hard / app, 4) if app else None,
               "ratio_soft": round(soft / app, 4) if app else None,
               "sev": sev, "basis": basis}
        if len(res) > 3 and res[3]:
            rec["extra"] = res[3]
        if rid in REGRESSION_ONLY:
            rec["regression_only"] = True
        out[rid] = rec
    return out


def eligible_dims(scan: dict, dim_of: dict) -> tuple:
    """按 v2 口径拆分维度：只有 **hard 违规** 才 mask 维度；
    soft 偏离不 mask（"不够优"不等于"有缺陷"），仅作元数据。
    regression_only 规则不参与 eligible。"""
    hard_viol, seen = set(), set()
    for rid, v in scan.items():
        if rid not in dim_of or v.get("regression_only") or "na" in v:
            continue
        if v.get("ratio") is None:
            continue
        seen.add(dim_of[rid])
        if v["sev"] == "hard" and v["ratio"] > 0:
            hard_viol.add(dim_of[rid])
    return sorted(seen - hard_viol), sorted(hard_viol)


if __name__ == "__main__":
    import sys
    d = json.load(open(sys.argv[1]))
    res = scan_seed(d)
    for rid, r in res.items():
        print(f"{rid:28s} {r}")

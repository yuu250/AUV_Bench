"""Task 1 prompts and scoring dimensions.

The rubric texts and the prompt template below are part of the experimental
definition. They are kept verbatim; do not reword.
"""

DIMENSIONS = [
    ("color", "色彩", "页面配色是否与页面整体协调，颜色的使用是否具有一致性，文字和背景的对比是否清晰"),
    ("typography", "字体", "字体、字号、字重、行高以及文字层级是否清晰、合理、易读"),
    ("graphics", "图形图像", "图片、图标、插画等视觉素材本身是否清晰、完整、比例合理、质量良好"),
    ("layout", "排版布局", "页面结构、信息层级、对齐、间距、分组以及空间利用是否合理"),
    ("component_consistency", "组件规范一致性", "相同功能或相同层级的组件是否遵循统一的设计规则"),
    ("visual_style_consistency", "视觉风格一致性", "圆角、阴影、边框、线条、图标风格等视觉语言是否统一"),
    ("copy_quality", "文案内容", "页面文字是否清楚、自然、容易理解，并符合当前页面的使用场景"),
    ("image_text_fit", "图文适配度", "图片、图标等视觉元素与其对应的文字、功能和上下文语义是否匹配"),
]

DIM_KEYS = [d[0] for d in DIMENSIONS]

# ─────────────────────────── Prompt ───────────────────────────

RUBRIC_TEXT = """\
## 评分标准

### 总体评分尺度（1-5 分）

| 分数 | 含义 |
|------|------|
| 5 | 优秀：没有明显可感知问题，整体达到较成熟的产品设计水平 |
| 4 | 良好：存在少量轻微问题，但基本不影响整体体验 |
| 3 | 一般：存在若干明显问题，需要优化，但页面仍可正常理解和使用 |
| 2 | 较差：存在多个较严重问题，已经明显影响视觉体验或信息理解 |
| 1 | 很差：存在系统性或致命问题，严重影响页面正常呈现或理解 |

### 色彩

> 页面配色是否与页面整体协调，颜色的使用是否具有一致性，文字和背景的对比是否清晰

- 5：配色成熟协调，主色、辅助色、背景色、强调色关系清晰；颜色使用具有一致规律；文字与背景对比清楚，无明显突兀、冲突或过度饱和问题。
- 4：整体配色协调，有明确的颜色体系，仅存在少量轻微问题，如个别颜色略突兀、局部对比稍弱或强调色使用略多，但不影响整体观感和阅读。
- 3：整体基本可接受，但存在若干明显问题，例如部分区域颜色不协调、颜色层级不够明确、局部文字与背景对比不足，页面仍可正常阅读和使用。
- 2：配色存在较严重问题，例如多个区域颜色冲突、颜色使用缺乏规律、高饱和颜色滥用或明显影响文字辨识，整体视觉质量较差。
- 1：配色严重失控，大面积颜色冲突或对比度异常，造成大量内容难以辨认，明显影响页面正常理解和使用。

### 字体

> 字体、字号、字重、行高以及文字层级是否清晰、合理、易读

- 5：字体体系完整统一；标题、正文、说明文字等层级明确；字号、字重、行高搭配合理，阅读舒适，无明显异常文字样式。
- 4：整体字体体系清晰，仅存在少量轻微问题，例如个别字号、字重或行高不够理想，但不影响阅读和信息层级。
- 3：基本可正常阅读，但存在若干明显字体问题，例如标题与正文区分不足、字号层级不稳定、字重使用不一致、局部文字过密或过松。
- 2：字体体系明显混乱，多个文本层级难以区分，字号、字重、行高等设置明显不合理，对阅读产生较大影响。
- 1：大量文字难以阅读，字体大小、样式或排布严重异常，文本层级基本失效。

### 图形图像

> 图片、图标、插画等视觉素材本身是否清晰、完整、比例合理、质量良好

- 5：图片清晰完整、比例正确、裁切自然；图标和其他图形精致、易识别，无拉伸、模糊、破损、缺失等问题。
- 4：整体素材质量良好，仅存在少量轻微问题，例如个别图片裁切不够理想、局部图标精细度略低，但不会明显影响视觉效果。
- 3：素材基本可用，但存在若干可感知问题，例如部分图片略模糊、裁切欠佳、图标质量一般或部分图形辨识度较低。
- 2：多个重要视觉素材存在明显模糊、拉伸、异常裁切、低质量或缺失问题，已经明显降低页面视觉质量。
- 1：大量核心图片或图形损坏、缺失、无法辨认或严重失真，明显破坏页面正常呈现。

### 排版布局

> 页面结构、信息层级、对齐、间距、分组以及空间利用是否合理

- 5：页面整体结构清晰；主次关系明确；元素对齐准确；间距规律稳定；相关内容合理聚合，不同信息之间有明确区隔；无明显拥挤、异常留白、错位、遮挡或溢出。
- 4：整体布局合理成熟，仅有少量轻微的间距、对齐或尺寸比例问题，不影响阅读路径和信息结构。
- 3：主体结构基本合理，但存在若干明显问题，例如部分对齐不统一、间距不稳定、局部过密或过松、信息层级不够突出，需要一定人工调整。
- 2：页面存在较严重的布局问题，例如多个区域明显错位、拥挤、异常留白、层级混乱或空间分配失衡，明显影响信息获取。
- 1：页面布局严重失效，大量元素重叠、溢出、错位或结构混乱，难以正常理解页面内容。

### 组件规范一致性

> 相同功能或相同层级的组件是否遵循统一的设计规则

- 5：按钮、卡片、输入框、标签、导航等同类组件具有明确且统一的规范；相同功能对应稳定的尺寸、结构和状态表达，没有明显例外。
- 4：整体组件体系统一，仅有个别组件在尺寸、结构或样式上存在轻微偏差，不影响整体一致性感知。
- 3：可以识别出基本组件规范，但存在若干明显不一致，例如同级按钮尺寸不同、相同类型卡片结构存在差异、相同状态使用不同表达方式。
- 2：大量同类组件缺乏统一规则，相同功能或层级经常采用不同的设计方式，页面设计系统感较弱。
- 1：基本不存在统一组件规范，同类组件之间差异显著，页面呈现明显的拼接感。

### 视觉风格一致性

> 圆角、阴影、边框、线条、图标风格等视觉语言是否统一

- 5：页面具有稳定、统一的视觉语言，圆角、阴影、描边、线条粗细、装饰方式等具有明确规律，各区域风格高度一致。
- 4：整体视觉风格统一，仅存在少量局部细节偏差，例如个别圆角、阴影或边框样式略有不同，但不会破坏整体风格。
- 3：页面具有基本统一的视觉方向，但能够明显观察到若干风格不一致，例如不同区域使用不同圆角、阴影、边框或装饰方式。
- 2：不同区域视觉语言差异明显，圆角、阴影、描边、图标风格等缺乏统一规则，整体视觉较为割裂。
- 1：页面缺乏统一视觉风格，各区域像来自完全不同的设计体系，整体风格严重混乱。

### 文案内容

> 页面文字是否清楚、自然、容易理解，并符合当前页面的使用场景

- 5：文案准确、自然、简洁，标题、正文、按钮、提示语等表达清楚且符合使用场景，无明显歧义、语病、无意义文本或信息缺失。
- 4：整体文案清楚合理，仅存在少量措辞不自然、略显冗余或表达不够精准的问题，不影响主要信息理解。
- 3：核心内容能够理解，但存在若干明显问题，例如表达生硬、部分信息含糊、内容重复、文本组织一般或少量不符合场景的表达。
- 2：多个重要文案表达不清、明显不符合页面场景、存在语义错误或信息缺失，已经影响用户理解。
- 1：大量核心文字乱码、无意义、明显错误或无法理解，页面无法正常传递主要信息。

### 图文适配度

> 图片、图标等视觉元素与其对应的文字、功能和上下文语义是否匹配

- 5：图片、图标和文字高度匹配，视觉元素能够准确表达对应内容或功能，并有效辅助用户理解信息。
- 4：整体图文关系合理，仅存在少量语义表达不够精准的视觉元素，但不会造成明显误解。
- 3：大部分图文关系合理，但存在若干图片相关性一般、图标语义不够直观或需要结合上下文才能理解的问题。
- 2：多个重要图片或图标与对应文字、功能明显不匹配，容易造成理解偏差或错误预期。
- 1：图文关系大面积错误、无关或产生错误语义，关键视觉元素严重误导用户理解。
"""

RUBRIC_TEXT_EN = """\
## Scoring Rubric

### Overall Scale (1-5)

| Score | Meaning |
|------|---------|
| 5 | Excellent: no perceptible issues; overall quality matches mature product design |
| 4 | Good: a few minor issues that barely affect the overall experience |
| 3 | Fair: several noticeable issues that need improvement, but the page remains understandable and usable |
| 2 | Poor: multiple serious issues that clearly hurt the visual experience or information comprehension |
| 1 | Very poor: systemic or fatal problems that severely break the page's presentation or comprehension |

### Color

> Whether the palette harmonizes with the page as a whole, colors are used consistently, and text-background contrast is clear

- 5: Mature, harmonious palette with clear roles for primary, secondary, background, and accent colors; colors follow a consistent logic; text-background contrast is clear; no jarring, clashing, or oversaturated usage.
- 4: Overall harmonious palette with a clear color system; only a few minor issues, e.g., an occasionally jarring color, slightly weak local contrast, or slight overuse of accent colors; overall look and readability unaffected.
- 3: Generally acceptable, but with several noticeable problems, e.g., some areas with uncoordinated colors, unclear color hierarchy, or locally insufficient text-background contrast; the page remains readable and usable.
- 2: Serious palette problems, e.g., clashing colors in multiple areas, no discernible color logic, abuse of highly saturated colors, or text clearly hard to distinguish; overall visual quality is poor.
- 1: Palette completely out of control: large-scale color clashes or abnormal contrast make large amounts of content hard to recognize and clearly impede understanding and use.

### Typography

> Whether fonts, sizes, weights, line heights, and the text hierarchy are clear, reasonable, and readable

- 5: Complete and consistent type system; clear hierarchy among headings, body text, and captions; well-matched sizes, weights, and line heights; comfortable to read; no abnormal text styling.
- 4: Overall clear type system with only a few minor issues, e.g., occasional suboptimal size, weight, or line height choices; reading and information hierarchy unaffected.
- 3: Basically readable, but with several noticeable type problems, e.g., insufficient heading-body distinction, unstable size hierarchy, inconsistent weight usage, or locally cramped/loose text.
- 2: Clearly chaotic type system: multiple text levels hard to distinguish; sizes, weights, and line heights clearly unreasonable; reading is significantly affected.
- 1: Large amounts of text are hard to read; font sizes, styles, or arrangement are severely abnormal; the text hierarchy is effectively broken.

### Graphics & Imagery

> Whether images, icons, illustrations, and other visual assets are clear, intact, well-proportioned, and of good quality

- 5: Images are sharp, intact, correctly proportioned, and naturally cropped; icons and other graphics are refined and recognizable; no stretching, blurring, damage, or missing assets.
- 4: Overall good asset quality with only a few minor issues, e.g., occasionally imperfect cropping or slightly less refined icons; visual impact barely affected.
- 3: Assets are basically usable, but with several perceptible problems, e.g., some slightly blurry images, suboptimal cropping, mediocre icon quality, or low-recognizability graphics.
- 2: Multiple important visual assets are clearly blurry, stretched, abnormally cropped, low-quality, or missing, noticeably degrading the page's visual quality.
- 1: Large numbers of core images or graphics are broken, missing, unrecognizable, or severely distorted, clearly disrupting the page's presentation.

### Layout

> Whether the page structure, information hierarchy, alignment, spacing, grouping, and space utilization are reasonable

- 5: Clear overall structure with well-defined priorities; accurate element alignment; regular, stable spacing; related content properly grouped with clear separation between sections; no crowding, abnormal whitespace, misalignment, occlusion, or overflow.
- 4: Overall mature, reasonable layout with only a few minor spacing, alignment, or proportion issues; reading flow and information structure unaffected.
- 3: Main structure basically sound, but with several noticeable problems, e.g., inconsistent alignment, unstable spacing, locally cramped or loose areas, or weak information hierarchy; needs manual adjustment.
- 2: Serious layout problems, e.g., multiple areas clearly misaligned, crowded, oddly spaced, hierarchically confused, or unbalanced in space allocation; information access is clearly impeded.
- 1: Layout completely broken: large numbers of overlapping, overflowing, or misplaced elements; the page content is hard to comprehend.

### Component Consistency

> Whether components of the same function or hierarchy follow unified design rules

- 5: Buttons, cards, inputs, tags, navigation, and other component types follow clear, unified rules; the same function maps to stable sizes, structures, and state expressions, with no visible exceptions.
- 4: Overall unified component system with only occasional slight deviations in size, structure, or style; the perception of consistency is unaffected.
- 3: A basic component system is recognizable, but with several obvious inconsistencies, e.g., same-level buttons in different sizes, same-type cards with different structures, or the same state expressed in different ways.
- 2: Large numbers of similar components lack unified rules; the same function or hierarchy is frequently designed in different ways; the page feels weakly systematized.
- 1: Essentially no unified component rules; similar components differ drastically; the page feels visibly stitched together.

### Visual Style Consistency

> Whether the visual language — corner radii, shadows, borders, line weights, icon styles — is unified

- 5: Stable, unified visual language across the page; corner radii, shadows, strokes, line weights, and decorative treatments follow clear rules; style is highly consistent across sections.
- 4: Overall unified visual style with only minor local deviations, e.g., occasional differences in corner radius, shadow, or border style, without breaking the whole.
- 3: A basically unified visual direction exists, but several style inconsistencies are clearly visible, e.g., different sections using different corner radii, shadows, borders, or decorations.
- 2: Visual languages differ obviously across sections; corner radii, shadows, strokes, and icon styles lack unified rules; the page feels visually fragmented.
- 1: No unified visual style at all; sections look like they come from entirely different design systems; the overall style is severely chaotic.

### Copy Quality

> Whether the page text is clear, natural, easy to understand, and appropriate for the page's context

- 5: Copy is accurate, natural, and concise; headings, body text, buttons, and hints are clear and context-appropriate; no ambiguity, grammatical errors, meaningless text, or missing information.
- 4: Copy is generally clear and reasonable, with only minor unnatural phrasing, slight redundancy, or imprecise wording; main information is unaffected.
- 3: Core content is understandable, but with several obvious problems, e.g., stiff phrasing, partially vague information, repeated content, mediocre text organization, or occasional context-inappropriate expressions.
- 2: Multiple important texts are unclear, clearly context-inappropriate, semantically wrong, or missing information; user comprehension is already affected.
- 1: Large amounts of core text are garbled, meaningless, clearly wrong, or incomprehensible; the page fails to convey its main information.

### Image-Text Fit

> Whether images, icons, and other visual elements semantically match their associated text, function, and context

- 5: Images, icons, and text are highly matched; visual elements accurately express the corresponding content or function and effectively aid comprehension.
- 4: Image-text relationships are generally reasonable, with only a few visual elements that are semantically imprecise but cause no obvious misunderstanding.
- 3: Most image-text relationships are reasonable, but with several issues, e.g., weakly relevant images, icons whose meaning is not intuitive, or visuals that require context to be understood.
- 2: Multiple important images or icons clearly mismatch their text or function, easily causing misunderstanding or wrong expectations.
- 1: Image-text relationships are largely wrong, irrelevant, or misleading; key visual elements seriously misguide user comprehension.
"""


def build_prompt(lang: str = "en") -> str:
    """构造评分 prompt。lang='en'（主 setting）或 'zh'（对照 setting）。"""
    if lang == "zh":
        dims_json = ",\n".join(
            f'    "{key}": {{"score": <1-5>, "reason": "<一句话理由>"}}'
            for key, _, _ in DIMENSIONS
        )
        return f"""你是一位专业 UI 设计评审专家。请根据以下评分标准，对给定的 UI 页面截图进行多维度美学评分。

{RUBRIC_TEXT}

## 输出要求

请严格按照以下 JSON 格式输出，不要输出任何其他内容：

```json
{{
{dims_json},
    "overall": {{"score": <1-5>, "reason": "<一句话总评>"}}
}}
```

注意：
- 每个维度独立评分，不要因为某一维度差就拉低所有维度
- overall 是对页面的整体美学评价，不是简单的维度平均分
- reason 用中文，一句话即可
"""
    dims_json = ",\n".join(
        f'    "{key}": {{"score": <1-5>, "reason": "<one-sentence justification>"}}'
        for key, _, _ in DIMENSIONS
    )
    return f"""You are a professional UI design reviewer. Rate the given UI page screenshot on multiple aesthetic dimensions according to the rubric below.

{RUBRIC_TEXT_EN}

## Output Requirements

Output ONLY the following JSON, with no other text:

```json
{{
{dims_json},
    "overall": {{"score": <1-5>, "reason": "<one-sentence overall comment>"}}
}}
```

Notes:
- Score each dimension independently; do not let one weak dimension drag down the others.
- "overall" is a holistic aesthetic judgment, not a simple average of the dimension scores.
- Keep each reason to a single sentence, in English.
"""

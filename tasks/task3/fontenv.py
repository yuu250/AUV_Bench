"""Pinned open-licensed font set: fonts.conf generation and manifest building.

The benchmark pages write their CSS font stacks against generic system family
names (``-apple-system``, ``PingFang SC``, ``Microsoft YaHei``, ``SimSun``,
``Arial``, ``Helvetica`` ...).  This module resolves every such name to the
pinned open font set shipped in the data package, via a hermetic fontconfig
configuration that deliberately excludes host system font directories so no
host font can leak into a render.

The exact file set is pinned in ``REQUIRED_FONTS``; ``build_manifest`` records
each file's byte length so the runtime provenance check
(``render_environment.render_environment_provenance``) can verify the set by
direct comparison.

Measured Blink/fontconfig behaviors the mapping relies on (Chromium 151):

* CSS ``system-ui`` (quoted or not) reaches fontconfig as the generic
  "sans-serif", so the GENERIC mappings control it.
* CSS ``sans-serif``/``serif`` keywords are pre-resolved by Blink to the
  literal families "Arial"/"Times New Roman" and never reach the generic
  rules; the ALIASES table therefore maps those literals.
* Blink also hardcodes "Times New Roman" for the page default font and for
  ui-* keywords, which the same ALIASES entry covers.
* ``mode="assign"`` replaces a family in place, preserving CSS stack
  priority; ``mode="append" binding="weak"`` mirrors the fallback cascade
  (latin primary wins for covered glyphs, CJK companion covers the rest).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# family -> sorted font files shipped in the data package.  All files are
# open-licensed (SIL OFL or Apache-2.0) and pinned byte-for-byte.
REQUIRED_FONTS: dict[str, list[str]] = {
    "Noto Sans SC": [
        "NotoSansSC-Bold.otf",
        "NotoSansSC-Medium.otf",
        "NotoSansSC-Regular.otf",
    ],
    "Noto Serif SC": [
        "NotoSerifSC-Bold.otf",
        "NotoSerifSC-Regular.otf",
    ],
    "Liberation Sans": [
        "LiberationSans-Bold.ttf",
        "LiberationSans-BoldItalic.ttf",
        "LiberationSans-Italic.ttf",
        "LiberationSans-Regular.ttf",
    ],
    "Liberation Serif": [
        "LiberationSerif-Bold.ttf",
        "LiberationSerif-BoldItalic.ttf",
        "LiberationSerif-Italic.ttf",
        "LiberationSerif-Regular.ttf",
    ],
    "Liberation Mono": [
        "LiberationMono-Bold.ttf",
        "LiberationMono-BoldItalic.ttf",
        "LiberationMono-Italic.ttf",
        "LiberationMono-Regular.ttf",
    ],
    "DejaVu Sans": ["DejaVuSans-Bold.ttf", "DejaVuSans.ttf"],
    "DejaVu Serif": ["DejaVuSerif-Bold.ttf", "DejaVuSerif.ttf"],
    "DejaVu Sans Mono": ["DejaVuSansMono-Bold.ttf", "DejaVuSansMono.ttf"],
    "Inter": ["InterVariable.ttf"],
    "Roboto": ["Roboto[wdth,wght].ttf"],
}

SANS_UI = "Noto Sans SC"
SANS_LATIN = "Liberation Sans"
SERIF_LATIN = "Liberation Serif"
SERIF_CJK = "Noto Serif SC"
MONO = "Liberation Mono"

# fontconfig generics (Chromium passes CSS system-ui through "sans-serif").
GENERIC = {
    "sans-serif": SANS_UI,
    "sans": SANS_UI,
    "serif": SERIF_LATIN,
    "mono": MONO,
    "monospace": MONO,
}

# CSS family name -> pinned open family.
ALIASES = {
    # platform UI stacks
    "-apple-system": SANS_UI,
    "BlinkMacSystemFont": SANS_UI,
    "system-ui": SANS_UI,
    # CJK sans
    "PingFang SC": SANS_UI,
    "PingFangSC-Regular": SANS_UI,
    "PingFangSC-Medium": SANS_UI,
    "PingFangSC-Semibold": SANS_UI,
    "苹方-简": SANS_UI,
    "微软雅黑": SANS_UI,
    "Microsoft YaHei": SANS_UI,
    "黑体": SANS_UI,
    "SimHei": SANS_UI,
    "STHeiti": SANS_UI,
    "华文黑体": SANS_UI,
    # CJK serif
    "simsun": SERIF_CJK,
    "SimSun": SERIF_CJK,
    "宋体": SERIF_CJK,
    "Songti SC": SERIF_CJK,
    "STSong": SERIF_CJK,
    "华文宋体": SERIF_CJK,
    "楷体": SERIF_CJK,
    "KaiTi": SERIF_CJK,
    "Kaiti SC": SERIF_CJK,
    "仿宋": SERIF_CJK,
    "FangSong": SERIF_CJK,
    # latin sans
    "Arial": SANS_LATIN,
    "Helvetica": SANS_LATIN,
    "Helvetica Neue": SANS_LATIN,
    "freesans": SANS_LATIN,
    "ReithSans": SANS_LATIN,
    "Nacelle": SANS_LATIN,
    "Verdana": "DejaVu Sans",
    "Geneva": "DejaVu Sans",
    "Tahoma": "DejaVu Sans",
    "Segoe UI": "DejaVu Sans",
    "Lucida Grande": "DejaVu Sans",
    "lucida": "DejaVu Sans",
    # latin serif
    "Times New Roman": SERIF_LATIN,
    "Times": SERIF_LATIN,
    "ReithSerif": SERIF_LATIN,
    "Georgia": "DejaVu Serif",
    # monospace
    "Courier New": MONO,
    "Courier": MONO,
    "Menlo": MONO,
    # shipped under their own names
    "Roboto": "Roboto",
    "Inter": "Inter",
}

# Latin families get a weak CJK companion, mirroring the platform fallback
# cascade: the primary family still wins for every glyph it covers.
CJK_SANS_AFTER = [SANS_LATIN, "DejaVu Sans", "Roboto", "Inter", MONO]
CJK_SERIF_AFTER = [SERIF_LATIN, "DejaVu Serif"]

# Last-resort cascade for families nothing resolves to.
DEFAULT_FAMILY = SERIF_LATIN
DEFAULT_CJK_FALLBACK = SERIF_CJK

MANIFEST_PURPOSE = "task3 hermetic render font environment"
MANIFEST_SOURCE = "pinned open-licensed fonts (SIL OFL / Apache-2.0)"


def _substitute(name: str, target: str) -> str:
    return (
        "  <match target=\"pattern\">\n"
        f"    <test name=\"family\" compare=\"eq\">"
        f"<string>{name}</string></test>\n"
        f"    <edit name=\"family\" mode=\"assign\" "
        f"binding=\"same\"><string>{target}</string></edit>\n"
        "  </match>\n"
    )


def _append_fallback(after: str, fallback: str) -> str:
    return (
        "  <match target=\"pattern\">\n"
        f"    <test name=\"family\" compare=\"eq\">"
        f"<string>{after}</string></test>\n"
        f"    <edit name=\"family\" mode=\"append\" "
        f"binding=\"weak\"><string>{fallback}</string></edit>\n"
        "  </match>\n"
    )


def build_fonts_conf(render_fonts_dir: str | Path) -> str:
    """Return the hermetic fonts.conf content for the pinned font set."""

    render_fonts_dir = Path(render_fonts_dir)
    fonts_dir = render_fonts_dir / "fonts"
    cache_dir = render_fonts_dir / "cache"
    families = set(REQUIRED_FONTS)

    parts = [
        '<?xml version="1.0"?>',
        '<!DOCTYPE fontconfig SYSTEM "fonts.dtd">',
        "<fontconfig>",
        f"  <dir>{fonts_dir.as_posix()}</dir>",
        f"  <cachedir>{cache_dir.as_posix()}</cachedir>",
        "",
        "  <!-- CSS font stack names -> pinned open families -->",
    ]
    for css_name, target in {**ALIASES, **GENERIC}.items():
        if target in families:
            parts.append(_substitute(css_name, target))

    parts.append("  <!-- CJK companion cascade: sans -->")
    for fam in CJK_SANS_AFTER:
        if fam in families:
            parts.append(_append_fallback(fam, SANS_UI))
    parts.append("  <!-- CJK companion cascade: serif -->")
    for fam in CJK_SERIF_AFTER:
        if fam in families:
            parts.append(_append_fallback(fam, SERIF_CJK))

    parts.append(
        "  <!-- unresolvable families -> default cascade, last resort -->\n"
        "  <match target=\"pattern\">\n"
        f"    <edit name=\"family\" mode=\"append\" "
        f"binding=\"weak\"><string>{DEFAULT_FAMILY}</string></edit>\n"
        f"    <edit name=\"family\" mode=\"append\" "
        f"binding=\"weak\"><string>{DEFAULT_CJK_FALLBACK}</string></edit>\n"
        "  </match>\n"
    )
    # Deterministic rasterization: no hinting (hinting shifts advance widths
    # and therefore line wrap), grayscale antialiasing, no subpixel.
    parts.append(
        "  <!-- pinned rendering: no hinting, grayscale AA -->\n"
        "  <match target=\"font\">\n"
        "    <edit name=\"antialias\" mode=\"assign\"><bool>true</bool></edit>\n"
        "    <edit name=\"hinting\" mode=\"assign\"><bool>false</bool></edit>\n"
        "    <edit name=\"hintstyle\" mode=\"assign\">"
        "<const>hintnone</const></edit>\n"
        "    <edit name=\"rgba\" mode=\"assign\"><const>none</const></edit>\n"
        "    <edit name=\"autohint\" mode=\"assign\"><bool>false</bool></edit>\n"
        "    <edit name=\"lcdfilter\" mode=\"assign\">"
        "<const>lcdnone</const></edit>\n"
        "  </match>\n"
    )
    parts.append("</fontconfig>")
    return "\n".join(parts) + "\n"


def build_manifest(fonts_dir: str | Path) -> dict:
    """Build the font manifest from the pinned file set on disk.

    Fails closed on any missing or extra file: the manifest describes exactly
    ``REQUIRED_FONTS`` and nothing else.
    """

    fonts_dir = Path(fonts_dir)
    expected = {name for names in REQUIRED_FONTS.values() for name in names}
    actual = {
        path.name
        for path in fonts_dir.iterdir()
        if path.suffix.lower() in {".ttf", ".ttc", ".otf", ".otc", ".dfont"}
    }
    if actual != expected:
        raise ValueError(
            "font directory does not match the pinned set: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    files = [
        {"file": name, "bytes": (fonts_dir / name).stat().st_size}
        for name in sorted(expected)
    ]
    return {
        "purpose": MANIFEST_PURPOSE,
        "source": MANIFEST_SOURCE,
        "n_fonts": len(files),
        "families": {family: list(names) for family, names in REQUIRED_FONTS.items()},
        "files": files,
    }


def setup_font_environment(render_fonts_dir: str | Path) -> dict:
    """Write MANIFEST.json + fonts.conf into ``render_fonts_dir``.

    ``render_fonts_dir/fonts`` must already contain exactly the pinned set
    (it ships that way in the data package).  Returns the manifest.
    """

    render_fonts_dir = Path(render_fonts_dir)
    fonts_dir = render_fonts_dir / "fonts"
    if not fonts_dir.is_dir():
        raise ValueError(f"font directory is missing: {fonts_dir}")
    manifest = build_manifest(fonts_dir)
    (render_fonts_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    conf = build_fonts_conf(render_fonts_dir)
    conf_path = render_fonts_dir / "fonts.conf"
    conf_path.write_text(conf, encoding="utf-8")
    cache_dir = render_fonts_dir / "cache"
    cache_dir.mkdir(exist_ok=True)
    return manifest


def render_environ(render_fonts_dir: str | Path) -> dict[str, str]:
    """Process environment for renders: full environ + pinned FONTCONFIG_FILE."""

    conf = (Path(render_fonts_dir) / "fonts.conf").resolve()
    return {**os.environ, "FONTCONFIG_FILE": conf.as_posix()}


DEFAULT_RENDER_FONTS_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "data" / "task3" / "render_fonts"
)
RENDER_FONTS_ENV_VAR = "TASK3_RENDER_FONTS_DIR"


def resolve_render_fonts_dir() -> Path:
    override = os.environ.get(RENDER_FONTS_ENV_VAR)
    if override:
        return Path(override).resolve()
    return DEFAULT_RENDER_FONTS_DIR


def configure_render_environment(render_fonts_dir: str | Path | None = None) -> dict:
    """Regenerate the font manifest/conf and export FONTCONFIG_FILE.

    Idempotent: the manifest and fonts.conf are deterministic functions of
    the shipped ``fonts/`` directory.  Must run before any browser launch so
    Chromium's fontconfig picks up the hermetic font set.
    """

    root = Path(render_fonts_dir) if render_fonts_dir else resolve_render_fonts_dir()
    manifest = setup_font_environment(root)
    os.environ.update(render_environ(root))
    return manifest


__all__ = [
    "DEFAULT_RENDER_FONTS_DIR",
    "RENDER_FONTS_ENV_VAR",
    "REQUIRED_FONTS",
    "build_fonts_conf",
    "build_manifest",
    "configure_render_environment",
    "render_environ",
    "resolve_render_fonts_dir",
    "setup_font_environment",
]

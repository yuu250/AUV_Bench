#!/usr/bin/env python3
"""DOM element-index collection for the no-JS grading render.

``COLLECT_JS`` is injected into a page rendered with JavaScript disabled (the
injected script itself is exempt from ``java_script_enabled=False``) and
collects, for every visible element, its xpath, tag, class, id, bounding box,
key computed styles, own text, and image natural sizes.  Page-level scroll
geometry is collected alongside.

``detect_peer_groups`` then buckets elements by (parent xpath, tag, sorted
class signature) and keeps buckets of >= 3 members whose width and height
spread stays within a max/min ratio of 1.5.
"""

from __future__ import annotations

STYLE_PROPS = [
    "fontSize", "fontWeight", "fontFamily", "lineHeight", "color",
    "backgroundColor", "marginTop", "marginRight", "marginBottom", "marginLeft",
    "paddingTop", "paddingRight", "paddingBottom", "paddingLeft",
    "rowGap", "columnGap", "borderRadius", "boxShadow",
    "borderTopWidth", "borderTopColor", "textAlign", "display", "position",
]

COLLECT_JS = """
(props) => {
  function xpathOf(el) {
    const parts = [];
    let n = el;
    while (n && n.nodeType === 1) {
      let i = 1, s = n.previousElementSibling;
      while (s) { if (s.tagName === n.tagName) i++; s = s.previousElementSibling; }
      parts.unshift(n.tagName.toLowerCase() + '[' + i + ']');
      n = n.parentElement;
    }
    return '/' + parts.join('/');
  }
  const out = [];
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_ELEMENT);
  let node;
  while ((node = walker.nextNode())) {
    const cs = getComputedStyle(node);
    if (cs.display === 'none' || cs.visibility === 'hidden' || parseFloat(cs.opacity) === 0) continue;
    const r = node.getBoundingClientRect();
    if (r.width < 0.5 && r.height < 0.5) continue;
    const st = {};
    for (const p of props) st[p] = cs[p];
    let text = '';
    for (const c of node.childNodes) {
      if (c.nodeType === 3) text += c.textContent;
    }
    text = text.replace(/\\s+/g, ' ').trim().slice(0, 120);
    const e = {
      xpath: xpathOf(node),
      tag: node.tagName.toLowerCase(),
      cls: (node.getAttribute('class') || '').slice(0, 200),
      id: node.id || '',
      bbox: [Math.round(r.x * 10) / 10, Math.round(r.y * 10) / 10,
             Math.round(r.width * 10) / 10, Math.round(r.height * 10) / 10],
      style: st,
      text: text,
    };
    if (node.tagName === 'IMG') {
      e.natural = [node.naturalWidth || 0, node.naturalHeight || 0];
      e.src = (node.getAttribute('src') || '').slice(0, 160);
    }
    if (node.tagName === 'INPUT' || node.tagName === 'TEXTAREA') {
      e.placeholder = (node.getAttribute('placeholder') || '').slice(0, 80);
    }
    out.push(e);
    if (out.length >= 6000) break;
  }
  return {
    scrollW: document.documentElement.scrollWidth,
    scrollH: document.documentElement.scrollHeight,
    clientW: document.documentElement.clientWidth,
    overflowX: document.documentElement.scrollWidth > document.documentElement.clientWidth + 1,
    n_elements: out.length,
    elements: out,
  };
}
"""


def detect_peer_groups(elements):
    """Same parent + same tag/class signature + similar bbox size (max/min <= 1.5)
    + group size >= 3.  Mutates each member element with its ``peer_group`` id."""
    from collections import defaultdict
    buckets = defaultdict(list)
    for i, e in enumerate(elements):
        parent = e["xpath"].rsplit("/", 1)[0]
        sig = " ".join(sorted(e["cls"].split()))
        buckets[(parent, e["tag"], sig)].append(i)
    groups = []
    gid = 0
    for (_, tag, _), idxs in sorted(buckets.items()):
        if len(idxs) < 3:
            continue
        ws = [elements[i]["bbox"][2] for i in idxs if elements[i]["bbox"][2] > 0]
        hs = [elements[i]["bbox"][3] for i in idxs if elements[i]["bbox"][3] > 0]
        if not ws or not hs:
            continue
        if max(ws) / min(ws) > 1.5 or max(hs) / min(hs) > 1.5:
            continue
        members = []
        for i in idxs:
            elements[i]["peer_group"] = gid
            members.append(elements[i]["xpath"])
        groups.append({"gid": gid, "tag": tag, "size": len(idxs), "members": members})
        gid += 1
    return groups

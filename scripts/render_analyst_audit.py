"""Render the analyst methodology Markdown into a self-contained web page.

No runtime Markdown dependency is needed: this intentionally small renderer covers
the headings, tables, lists, quotes, code blocks, and inline formatting used by the
audit. Run it after editing docs/analyst-numbers-audit.md.
"""

from __future__ import annotations

import html
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "analyst-numbers-audit.md"
OUTPUT = ROOT / "src" / "web" / "static" / "analyst-numbers-audit.html"


def inline(text: str) -> str:
    escaped = html.escape(text, quote=False)
    code: list[str] = []

    def stash(match: re.Match[str]) -> str:
        code.append(f"<code>{match.group(1)}</code>")
        return f"\x00CODE{len(code) - 1}\x00"

    escaped = re.sub(r"`([^`]+)`", stash, escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"\[([^]]+)]\(([^)]+)\)", r'<a href="\2">\1</a>', escaped)
    for i, value in enumerate(code):
        escaped = escaped.replace(f"\x00CODE{i}\x00", value)
    return escaped


def cells(line: str) -> list[str]:
    return [part.strip() for part in line.strip().strip("|").split("|")]


def render(markdown: str) -> str:
    lines = markdown.splitlines()
    out: list[str] = []
    paragraph: list[str] = []
    list_kind: str | None = None
    in_code = False
    code_lines: list[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            out.append(f"<p>{inline(' '.join(part.strip() for part in paragraph))}</p>")
            paragraph.clear()

    def close_list() -> None:
        nonlocal list_kind
        if list_kind:
            out.append(f"</{list_kind}>")
            list_kind = None

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```"):
            flush_paragraph()
            close_list()
            if in_code:
                out.append(f"<pre><code>{html.escape(chr(10).join(code_lines))}</code></pre>")
                code_lines.clear()
                in_code = False
            else:
                in_code = True
            i += 1
            continue
        if in_code:
            code_lines.append(line)
            i += 1
            continue

        if not stripped:
            flush_paragraph()
            close_list()
            i += 1
            continue

        if re.fullmatch(r"-{3,}", stripped):
            flush_paragraph()
            close_list()
            out.append("<hr>")
            i += 1
            continue

        heading = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if heading:
            flush_paragraph()
            close_list()
            level = len(heading.group(1))
            title = heading.group(2)
            anchor = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
            out.append(f'<h{level} id="{anchor}">{inline(title)}</h{level}>')
            i += 1
            continue

        if (
            "|" in stripped
            and i + 1 < len(lines)
            and re.fullmatch(r"\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*", lines[i + 1])
        ):
            flush_paragraph()
            close_list()
            headers = cells(stripped)
            i += 2
            rows: list[list[str]] = []
            while i < len(lines) and "|" in lines[i] and lines[i].strip():
                rows.append(cells(lines[i]))
                i += 1
            out.append('<div class="table-wrap"><table><thead><tr>')
            out.extend(f"<th>{inline(value)}</th>" for value in headers)
            out.append("</tr></thead><tbody>")
            for row in rows:
                out.append("<tr>")
                out.extend(f"<td>{inline(value)}</td>" for value in row)
                out.append("</tr>")
            out.append("</tbody></table></div>")
            continue

        quote = re.match(r"^>\s?(.*)$", stripped)
        if quote:
            flush_paragraph()
            close_list()
            quote_lines = [quote.group(1)]
            i += 1
            while i < len(lines) and lines[i].strip().startswith(">"):
                quote_lines.append(re.sub(r"^>\s?", "", lines[i].strip()))
                i += 1
            out.append(f"<blockquote>{inline(' '.join(quote_lines))}</blockquote>")
            continue

        item = re.match(r"^\s*([-*]|\d+\.)\s+(.+)$", line)
        if item:
            flush_paragraph()
            kind = "ol" if item.group(1)[0].isdigit() else "ul"
            if list_kind != kind:
                close_list()
                out.append(f"<{kind}>")
                list_kind = kind
            out.append(f"<li>{inline(item.group(2))}</li>")
            i += 1
            continue

        paragraph.append(stripped)
        i += 1

    flush_paragraph()
    close_list()
    if in_code:
        out.append(f"<pre><code>{html.escape(chr(10).join(code_lines))}</code></pre>")
    return "\n".join(out)


def page(body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="dark">
  <title>AlphaDesk — Analyst numbers and formulas</title>
  <style>
    :root {{ --bg:#0d110f; --panel:#121714; --ink:#e7ebe8; --muted:#9ba39e; --teal:#9ecdc7;
      --line:#303833; --amber:#dfc182; --red:#dc8f91; --code:#171d19; }}
    * {{ box-sizing:border-box; }}
    html {{ scroll-behavior:smooth; }}
    body {{ margin:0; background:var(--bg); color:var(--ink); font:16px/1.72 ui-monospace,SFMono-Regular,
      Menlo,Monaco,Consolas,"Liberation Mono",monospace; }}
    .top {{ position:sticky; top:0; z-index:2; display:flex; align-items:center; gap:18px; padding:12px 22px;
      border-bottom:1px solid var(--line); background:rgba(13,17,15,.94); backdrop-filter:blur(12px); }}
    .brand {{ color:var(--teal); font-weight:800; letter-spacing:.08em; }}
    .top a {{ color:var(--muted); text-decoration:none; font-size:13px; }}
    .top a:hover {{ color:var(--teal); }}
    .top button {{ margin-left:auto; border:1px solid var(--line); border-radius:6px; padding:7px 11px;
      color:var(--ink); background:var(--panel); font:inherit; font-size:12px; cursor:pointer; }}
    main {{ width:min(1040px,calc(100% - 36px)); margin:34px auto 100px; padding:38px 52px 70px;
      border:1px solid var(--line); border-radius:14px; background:var(--panel); box-shadow:0 24px 80px rgba(0,0,0,.24); }}
    h1,h2,h3,h4 {{ color:var(--teal); line-height:1.3; scroll-margin-top:74px; }}
    h1 {{ margin:0 0 8px; font-size:36px; letter-spacing:-.04em; }}
    h2 {{ margin:54px 0 18px; padding-bottom:9px; border-bottom:1px solid var(--line); font-size:25px; }}
    h3 {{ margin:34px 0 12px; font-size:18px; color:var(--amber); }}
    h4 {{ color:var(--ink); }}
    p {{ margin:12px 0; }}
    strong {{ color:#fff; }}
    a {{ color:var(--teal); }}
    hr {{ margin:42px 0; border:0; border-top:1px solid var(--line); }}
    blockquote {{ margin:22px 0; padding:16px 20px; border-left:4px solid var(--amber); border-radius:0 8px 8px 0;
      color:#f3e6ca; background:#1b1a14; font-size:17px; }}
    ul,ol {{ padding-left:25px; }} li {{ margin:6px 0; }}
    code {{ padding:2px 5px; border:1px solid var(--line); border-radius:4px; color:#c9ded9; background:var(--code); }}
    pre {{ overflow:auto; margin:18px 0; padding:17px 19px; border:1px solid var(--line); border-radius:8px;
      background:var(--code); line-height:1.55; }}
    pre code {{ padding:0; border:0; background:transparent; }}
    .table-wrap {{ overflow:auto; margin:20px 0 26px; border:1px solid var(--line); border-radius:9px; }}
    table {{ width:100%; border-collapse:collapse; font-size:14px; line-height:1.5; }}
    th,td {{ padding:11px 13px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }}
    th {{ color:var(--teal); background:#171d19; }} tr:last-child td {{ border-bottom:0; }}
    tbody tr:hover {{ background:#151b17; }}
    @media(max-width:700px) {{ .top {{ padding:10px 14px; }} main {{ width:100%; margin:0; padding:28px 18px 60px;
      border-left:0; border-right:0; border-radius:0; }} h1 {{ font-size:28px; }} h2 {{ font-size:22px; }} }}
    @media print {{ body {{ background:#fff; color:#111; font:11pt/1.5 Georgia,serif; }} .top {{ display:none; }}
      main {{ width:auto; margin:0; padding:0; border:0; box-shadow:none; background:#fff; }} h1,h2,h3 {{ color:#163f3a; }}
      code,pre {{ color:#111; background:#f5f5f5; }} .table-wrap {{ overflow:visible; }} blockquote {{ color:#222; background:#f7f3e9; }} }}
  </style>
</head>
<body>
  <nav class="top"><span class="brand">ALPHADESK</span><a href="/">← back to the desk</a>
    <button onclick="window.print()">print / save pdf</button></nav>
  <main>{body}</main>
</body>
</html>
"""


if __name__ == "__main__":
    OUTPUT.write_text(page(render(SOURCE.read_text())), encoding="utf-8")
    print(f"rendered {SOURCE.relative_to(ROOT)} -> {OUTPUT.relative_to(ROOT)}")

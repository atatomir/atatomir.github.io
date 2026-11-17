"""One-time import of the old HelpNDoc help center into clean article fragments.

Usage: python3 -I _src/import_help.py <dir with the downloaded *.htm files and contents_data.js>

Writes _src/help/toc.json and _src/help/<page>.html. build.py turns those into /help/*.htm
(same URLs as before, so links from the app keep working). Images live in /help/images/.
After the import, edit the fragments in _src/help/ directly.
"""
import json
import re
import struct
import sys
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Comment

SRC = Path(sys.argv[1])
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "_src/help"
IMG = ROOT / "help/images"
OUT.mkdir(exist_ok=True)


def clean_name(s):
    return re.sub(r"\s+", " ", s.replace("&nbsp;", " ")).strip()


# ---------------------------------------------------------------- table of contents
toc, sections, cur = [], {}, None
for line in (SRC / "contents_data.js").read_text(encoding="utf-8-sig").splitlines():
    m = re.match(r"(\w+) = insFld\(foldersTree, gFld\('(.*?)', '(.*?)'\)\)", line)
    if m:
        sec = {"title": clean_name(m.group(2)), "file": m.group(3), "pages": []}
        sections[m.group(1)] = sec
        toc.append(sec)
        continue
    m = re.match(r"docAux = insDoc\((\w+), gLnk\('S', '(.*?)', '(.*?)'\)\)", line)
    if m:
        page = {"title": clean_name(m.group(2)), "file": m.group(3)}
        if m.group(1) == "foldersTree":
            toc.append({**page, "pages": []})
        else:
            sections[m.group(1)]["pages"].append(page)
(OUT / "toc.json").write_text(json.dumps(toc, indent=1))


def png_size(path):
    with open(path, "rb") as f:
        head = f.read(24)
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return struct.unpack(">II", head[16:24])
    return None


KEEP_ATTRS = {"a": {"href"}, "img": {"src", "alt"}, "td": {"colspan"}, "th": {"colspan"}}
UNWRAP = {"g", "span", "font", "tbody", "thead", "u", "center", "div", "big", "small"}


def text_of(el):
    return re.sub(r"\s+", " ", el.get_text(" ")).strip()


def convert(html):
    html = re.sub(r"<br\s*/?>", "<br/>", html, flags=re.I)
    soup = BeautifulSoup(html, "html.parser")
    root = soup.find("div", class_="PageHeader_Content")
    sib = root.find("div", class_="var_Sibling_Page_List")
    if sib:
        prev = sib.previous_sibling
        if isinstance(prev, NavigableString) and "See also" in prev:
            prev.extract()
        sib.decompose()
    for c in root.find_all(string=lambda s: isinstance(s, Comment)):
        c.extract()

    # "Learn more..." toggle panels -> <details>
    for t in root.find_all("table", class_="TogglePanel"):
        head = t.find("td", class_="TogglePanel_Heading")
        body = t.find("td", class_="TogglePanel_Body")
        d = soup.new_tag("details", attrs={"class": "more"})
        s = soup.new_tag("summary")
        s.string = text_of(head).rstrip(". ").rstrip(".") if head else "Learn more"
        d.append(s)
        inner = soup.new_tag("div", attrs={"class": "more-body"})
        for ch in list(body.contents if body else []):
            inner.append(ch.extract())
        d.append(inner)
        t.replace_with(d)

    # tables, innermost first
    for t in reversed(root.find_all("table")):
        if t.find("img", src=re.compile(r"info_32")):
            # info note -> callout
            for im in t.find_all("img", src=re.compile(r"info_32")):
                im.decompose()
            box = soup.new_tag("div", attrs={"class": "callout"})
            for td in t.find_all("td", recursive=True):
                if td.find("td"):
                    continue
                for ch in list(td.contents):
                    box.append(ch.extract())
            t.replace_with(box)
        elif t.find("img"):
            # layout table (screenshot next to text) -> media rows
            wrap = soup.new_tag("div", attrs={"class": "hm"})
            for tr in t.find_all("tr"):
                if tr.find_parent("table") is not t:
                    continue
                for td in tr.find_all(["td", "th"], recursive=False):
                    cell = soup.new_tag("div", attrs={"class": "hm-media" if td.find("img") and not text_of(td) else "hm-text"})
                    for ch in list(td.contents):
                        cell.append(ch.extract())
                    if cell.contents and (text_of(cell) or cell.find("img") or cell.find(["table", "details", "div"])):
                        wrap.append(cell)
            t.replace_with(wrap)
        else:
            t.attrs = {"class": ["props"]}

    # blockquote that only wraps a screenshot -> figure
    for bq in root.find_all("blockquote"):
        if bq.find("img") and not text_of(bq):
            bq.name = "p"
            bq.attrs = {"class": "fig"}

    # strip attributes, unwrap presentational tags
    keep_cls = {"more", "more-body", "callout", "hm", "hm-media", "hm-text", "props", "fig"}
    for el in root.find_all(True):
        cls = [c for c in (el.get("class") or []) if c in keep_cls]
        if el.name in UNWRAP and not cls:
            el.unwrap()
            continue
        allowed = KEEP_ATTRS.get(el.name, set())
        el.attrs = {k: v for k, v in el.attrs.items() if k in allowed}
        if cls:
            el["class"] = cls
        if el.name == "b":
            el.name = "strong"
        if el.name == "i":
            el.name = "em"

    # images: lazy, sized, alt text
    for im in root.find_all("img"):
        src = im.get("src", "")
        sz = png_size(IMG / src.split("/", 1)[-1]) if src.startswith("images/") else None
        if sz:
            im["width"], im["height"] = sz
        im["alt"] = im.get("alt") or ""
        im["loading"] = "lazy"
        im["decoding"] = "async"

    for a in root.find_all("a"):
        h = a.get("href", "")
        if h.startswith("http") and "screentovideo.com/help" not in h:
            a["target"] = "_blank"
            a["rel"] = "noopener"

    out = root.decode_contents(formatter="html5")
    out = out.replace("\xa0", " ").replace("&nbsp;", " ")
    out = re.sub(r"<p>\s*</p>", "", out)
    out = re.sub(r"(<br>\s*){2,}", "<br>", out)
    out = re.sub(r"<br>\s*(</li>|</p>|</div>|<ul>|<ol>|<p)", r"\1", out)
    out = re.sub(r"(<li>|<p>|<div[^>]*>)\s*<br>", r"\1", out)
    out = re.sub(r"\n\s*\n+", "\n", out)
    out = re.sub(r"(\s*<br>\s*)+$", "", out.strip())
    return out.strip() + "\n"


n = 0
for sec in toc:
    for pg in [sec] + sec["pages"]:
        html = (SRC / pg["file"]).read_text(encoding="utf-8-sig")
        (OUT / pg["file"].replace(".htm", ".html")).write_text(convert(html))
        n += 1
print(f"imported {n} pages")

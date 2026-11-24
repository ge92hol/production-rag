import json
import re
import pandas as pd
import shutil
from pathlib import Path

import pymupdf4llm
from bs4 import BeautifulSoup
from langchain_core.documents import Document

from config import RAW_DIR, PROCESSED_DIR


def ensure_directories():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def save_raw_file(file_path):
    source = Path(file_path)
    destination = RAW_DIR / source.name
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    return destination


def make_metadata(file_path, page_number=None, section_heading="", heading_stack=()):
    return {
        "source_document": Path(file_path).name,
        "page_number": page_number,
        "section_heading": section_heading,
        # Internal: headings still open *before* this document's text (e.g. carried
        # over from the previous PDF page) as JSON [[level, title], ...]. Chunking
        # uses it to build full heading paths and then drops it.
        "heading_stack": json.dumps([list(h) for h in heading_stack])
    }


def save_processed_documents(documents, source_name):
    rows = [{
        "page_content": doc.page_content,
        "source_document": doc.metadata["source_document"],
        "page_number": doc.metadata["page_number"],
        "section_heading": doc.metadata["section_heading"],
        "heading_stack": doc.metadata.get("heading_stack", "[]")
    } for doc in documents]
    pd.DataFrame(rows).to_parquet(PROCESSED_DIR / f"{source_name}.parquet", index=False)


def load_processed_document(file_path):
    df = pd.read_parquet(file_path)
    return [
        Document(
            page_content=row["page_content"],
            metadata={
                "source_document": row["source_document"],
                "page_number": None if pd.isna(row["page_number"]) else int(row["page_number"]),
                "section_heading": "" if pd.isna(row["section_heading"]) else row["section_heading"],
                "heading_stack": row.get("heading_stack") or "[]"
            }
        )
        for _, row in df.iterrows()
    ]


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)(?:\s+#+)?\s*$")


def find_headings(text):
    """Return (char_position, level, title) for Markdown headings, ignoring fenced code blocks."""
    headings, in_fence, pos = [], False, 0

    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith(("```", "~~~")):
            in_fence = not in_fence
        elif not in_fence:
            match = HEADING_RE.match(stripped)
            if match:
                title = match.group(2).strip("* ").strip()
                if title:
                    indent = len(line) - len(line.lstrip())
                    headings.append((pos + indent, len(match.group(1)), title))
        pos += len(line)

    return headings


def heading_stack_at(stack, headings, position):
    """Heading hierarchy active at `position`: start from `stack`, apply headings at or before it."""
    stack = [tuple(h) for h in stack]
    for heading_position, level, title in headings:
        if heading_position > position:
            break
        stack = [h for h in stack if h[0] < level] + [(level, title)]
    return stack


def heading_path(stack):
    return " > ".join(title for _, title in stack)


def get_top_heading_level(text):
    levels = [level for _, level, _ in find_headings(text)]
    return min(levels) if levels else None


def split_top_sections(text, file_path, top_level, page_number=None, stack=()):
    """Split at top-level headings. Returns the documents and the heading stack at the end of text."""
    documents = []
    headings = find_headings(text)

    def save_section(start, end):
        raw = text[start:end]
        if raw.strip():
            content_start = start + len(raw) - len(raw.lstrip())
            documents.append(Document(
                page_content=raw.strip(),
                metadata=make_metadata(
                    file_path,
                    page_number,
                    heading_path(heading_stack_at(stack, headings, content_start)),
                    heading_stack_at(stack, headings, content_start - 1)
                )
            ))

    start = 0
    for position, level, _ in headings:
        if level == top_level:
            save_section(start, position)
            start = position

    save_section(start, len(text))
    return documents, heading_stack_at(stack, headings, len(text))


def load_pdf(file_path):
    pages = pymupdf4llm.to_markdown(str(file_path), page_chunks=True)
    all_text = "\n".join(page["text"] for page in pages)
    top_level = get_top_heading_level(all_text)

    documents = []
    stack = []  # heading hierarchy carried across pages; only headings pymupdf4llm detected

    for page in pages:
        text = page["text"].strip()
        if not text:
            continue

        page_number = page["metadata"]["page_number"]
        page_docs, stack = split_top_sections(text, file_path, top_level, page_number, stack)
        documents.extend(page_docs)

    return documents


def load_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        return []

    return [Document(
        page_content=text,
        metadata=make_metadata(file_path)
    )]


def load_markdown(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        return []

    top_level = get_top_heading_level(text)

    if top_level is None:
        return [Document(
            page_content=text,
            metadata=make_metadata(file_path)
        )]

    documents, _ = split_top_sections(text, file_path, top_level)
    return documents


HTML_BLOCK_TAGS = [
    "h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre",
    "blockquote", "tr", "dt", "dd", "figcaption", "caption", "div"
]
HTML_NON_DIV_BLOCKS = [tag for tag in HTML_BLOCK_TAGS if tag != "div"]


def load_html(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    for tag in soup(["script", "style", "noscript", "template"]):
        tag.decompose()

    lines = []

    for element in soup.find_all(HTML_BLOCK_TAGS):
        # Only leaf divs carry their own text; other blocks are emitted once by
        # their outermost block ancestor (avoids duplicating nested li/p text).
        if element.name == "div" and element.find(HTML_BLOCK_TAGS):
            continue
        if element.find_parent(HTML_NON_DIV_BLOCKS):
            continue

        if element.name == "pre":
            code = element.get_text().strip("\n")
            if code.strip():
                lines.append(f"```\n{code}\n```")
                lines.append("")
            continue

        if element.name == "tr":
            cells = [cell.get_text(" ", strip=True) for cell in element.find_all(["td", "th"])]
            text = " | ".join(cell for cell in cells if cell)
        else:
            text = element.get_text(" ", strip=True)

        if not text:
            continue

        if element.name[0] == "h" and element.name[1:].isdigit():
            lines.append(f"{'#' * int(element.name[1])} {text}")
        elif element.name == "li":
            lines.append(f"- {text}")
        else:
            lines.append(text)

        lines.append("")

    text = "\n".join(lines).strip()

    if not text:
        return []

    top_level = get_top_heading_level(text)

    if top_level is None:
        return [Document(
            page_content=text,
            metadata=make_metadata(file_path)
        )]

    documents, _ = split_top_sections(text, file_path, top_level)
    return documents


def load_document(file_path):
    ensure_directories()
    raw_path = save_raw_file(file_path)
    extension = raw_path.suffix.lower()

    if extension == ".pdf":
        documents = load_pdf(raw_path)
    elif extension == ".txt":
        documents = load_text(raw_path)
    elif extension in [".md", ".markdown"]:
        documents = load_markdown(raw_path)
    elif extension in [".html", ".htm"]:
        documents = load_html(raw_path)
    else:
        raise ValueError(f"Unsupported format: {extension}")

    # Use the full file name so guide.md and guide.pdf don't overwrite each other.
    save_processed_documents(documents, raw_path.name)
    return documents


def load_all_processed_documents():
    """Load every processed file for re-indexing without touching the raw sources."""
    ensure_directories()
    return [load_processed_document(path) for path in sorted(PROCESSED_DIR.glob("*.parquet"))]
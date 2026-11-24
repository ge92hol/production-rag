import json
import re
import numpy as np

from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

from config import SEMANTIC_BREAKPOINT_PERCENTILE, MIN_SEMANTIC_CHUNK_SIZE
from ingestion import HEADING_RE, find_headings, heading_path, heading_stack_at


def add_chunk_metadata(chunks, strategy):
    final_chunks = []

    for i, chunk in enumerate(chunks):
        metadata = dict(chunk.metadata)
        metadata["chunk_index"] = i
        metadata["chunking_strategy"] = strategy
        metadata["character_count"] = len(chunk.page_content)

        final_chunks.append(Document(
            page_content=chunk.page_content,
            metadata=metadata
        ))

    return final_chunks


def is_plain_text(document):
    return str(document.metadata.get("source_document", "")).lower().endswith(".txt")


def get_headings(document):
    # Plain text has no reliable heading markup, so never infer headings from it.
    if is_plain_text(document):
        return []
    return find_headings(document.page_content)


def make_chunk(document, headings, content, position):
    """Child chunk whose section_heading is the full heading path active at `position`
    (e.g. "Configuration > Timeouts > Client")."""
    metadata = dict(document.metadata)
    inherited = json.loads(metadata.pop("heading_stack", None) or "[]")
    if position >= 0 and (headings or inherited):
        metadata["section_heading"] = heading_path(heading_stack_at(inherited, headings, position))
    return Document(page_content=content, metadata=metadata)


def split_with(splitter, documents):
    chunks = []
    for document in documents:
        headings = get_headings(document)
        for chunk in splitter.split_documents([document]):
            start = chunk.metadata.pop("start_index", -1)
            chunks.append(make_chunk(document, headings, chunk.page_content, start))
    return chunks


def fixed_chunking(documents, chunk_size, chunk_overlap):
    splitter = CharacterTextSplitter(
        separator="",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True
    )

    return add_chunk_metadata(split_with(splitter, documents), "fixed")


def recursive_chunking(documents, chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(
        separators=[
            r"\n#{1,6}\s+",
            "\n\n",
            "\n",
            r"(?<=[.!?])\s+",
            " ",
            ""
        ],
        is_separator_regex=True,
        keep_separator="start",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True
    )

    return add_chunk_metadata(split_with(splitter, documents), "recursive")


# Unit boundaries: sentence ends, blank lines, and line breaks before headings/list items/table rows.
UNIT_BOUNDARY = re.compile(
    r"(?<=[.!?])\s+|\n\s*\n\s*|\n(?=\s*(?:#{1,6}\s|[-*+]\s|\d+[.)]\s|\||```))"
)


def split_into_sentences(text, max_length):
    """Return (start, end) spans of sentence-like units; units longer than max_length are cut."""
    spans, start = [], 0
    bounds = [(m.start(), m.end()) for m in UNIT_BOUNDARY.finditer(text)] + [(len(text), len(text))]

    for boundary, next_start in bounds:
        piece = text[start:boundary]
        if piece.strip():
            s = start + len(piece) - len(piece.lstrip())
            e = start + len(piece.rstrip())
            for cut in range(s, e, max_length):
                spans.append((cut, min(cut + max_length, e)))
        start = next_start

    return spans


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def trim(text, start, end):
    segment = text[start:end]
    return start + len(segment) - len(segment.lstrip()), start + len(segment.rstrip())


def build_heading_tree(text, headings):
    """Nested sections: each heading spans until the next heading of the same or higher level."""
    root = {"start": 0, "end": len(text), "level": 0, "children": []}
    stack = [root]
    for i, (position, level, _) in enumerate(headings):
        end = next((p for p, l, _ in headings[i + 1:] if l <= level), len(text))
        node = {"start": position, "end": end, "level": level, "children": []}
        while stack[-1]["level"] >= level:
            stack.pop()
        stack[-1]["children"].append(node)
        stack.append(node)
    return root


def partition(text, node, chunk_size):
    """Top-down structural partition of a section into ["whole" | "semantic", start, end] pieces.

    A section that fits is kept whole (descendants included). Otherwise it is split into
    its intro text + direct children, descending only into children that are too large;
    adjacent pieces that fit are packed back together. A leaf (or intro) that is still
    too large becomes a "semantic" piece for the embedding-based splitter.
    """
    start, end = trim(text, node["start"], node["end"])
    if end <= start:
        return []
    if end - start <= chunk_size:
        return [["whole", start, end, False]]
    if not node["children"]:
        return [["semantic", start, end, False]]

    groups = []  # (pieces, packable): only whole siblings/intro of THIS node get packed
    intro_start, intro_end = trim(text, start, node["children"][0]["start"])
    if intro_end > intro_start:
        fits = intro_end - intro_start <= chunk_size
        # intro=True: may later be packed into the first chunk of the first child (see pack_intros)
        groups.append(([["whole" if fits else "semantic", intro_start, intro_end, fits]], fits))
    for child in node["children"]:
        pieces = partition(text, child, chunk_size)
        groups.append((pieces, len(pieces) == 1 and pieces[0][0] == "whole"))

    result, previous_packable = [], False
    for pieces, packable in groups:
        if packable and previous_packable and pieces[0][2] - result[-1][1] <= chunk_size:
            result[-1][2] = pieces[0][2]
            result[-1][3] = False  # now contains sibling content, no longer a bare intro
        else:
            result.extend(pieces)
        previous_packable = packable
    return result


def has_body(text, heading_starts, start, end):
    """True if text[start:end] contains anything besides heading lines."""
    position = start
    for line in text[start:end].splitlines(keepends=True):
        line_start = position + len(line) - len(line.lstrip())
        if line.strip() and line_start not in heading_starts:
            return True
        position += len(line)
    return False


def attach_heading_only(pieces, text, heading_starts, chunk_size):
    """Fold body-less heading pieces (e.g. "## Timeouts" before its first child) into the
    next piece (or the previous one at the end) so no chunk is just a heading."""
    result, pending = [], None
    for kind, start, end, intro in pieces:
        if not has_body(text, heading_starts, start, end):
            pending = start if pending is None else pending
            continue
        if pending is not None:
            start, pending = pending, None
        result.append([kind, start, end, intro])
    if pending is not None:
        if result:
            result[-1][2] = pieces[-1][2]
        else:
            result.append(["whole", pending, pieces[-1][2], False])
    for piece in result:  # a whole piece that grew past chunk_size is split semantically
        if piece[0] == "whole" and piece[2] - piece[1] > chunk_size:
            piece[0], piece[3] = "semantic", False
    return result


def pack_intros(spans, chunk_size):
    """Pack a short parent intro into the chunk right after it (the first chunk of its first
    child's result) when the combined span fits. Right-to-left, so it applies recursively:
    a deeper intro is packed first, then its parent's intro can absorb the result."""
    for i in range(len(spans) - 2, -1, -1):
        start, end, intro = spans[i]
        if intro and spans[i + 1][1] - start <= chunk_size:
            spans[i] = [start, spans[i + 1][1], True]
            del spans[i + 1]
    return spans


def span_path(text, headings, inherited, start, end):
    """Deepest heading path shared by ALL content in text[start:end] (heading lines don't count)."""
    heading_starts = {p for p, _, _ in headings}
    cuts = [start] + [p for p, _, _ in headings if start < p < end]
    paths = [
        heading_stack_at(inherited, headings, a)
        for a, b in zip(cuts, cuts[1:] + [end])
        if has_body(text, heading_starts, a, b)
    ] or [heading_stack_at(inherited, headings, end - 1)]

    common = paths[0]
    for path in paths[1:]:
        n = 0
        while n < min(len(common), len(path)) and common[n] == path[n]:
            n += 1
        common = common[:n]
    return heading_path(common)


def make_span_chunk(document, headings, start, end):
    metadata = dict(document.metadata)
    inherited = json.loads(metadata.pop("heading_stack", None) or "[]")
    if headings or inherited:
        metadata["section_heading"] = span_path(document.page_content, headings, inherited, start, end)
    return Document(page_content=document.page_content[start:end], metadata=metadata)


def semantic_chunking(documents, embedding_model, chunk_size,
                      breakpoint_percentile=SEMANTIC_BREAKPOINT_PERCENTILE,
                      min_size=MIN_SEMANTIC_CHUNK_SIZE):
    """Hierarchical semantic chunking: structure first (heading tree, top-down by size),
    embedding-based breakpoints only inside leaves that are still larger than chunk_size."""
    # Keep the minimum meaningful if a small --chunk-size is passed (no effect at 800/200).
    min_size = min(min_size, chunk_size // 2)
    plans = []
    for document in documents:
        text = document.page_content
        headings = get_headings(document)
        if is_plain_text(document):
            # No reliable structure in TXT: always run the semantic splitter, even when the
            # whole file fits chunk_size (it may still contain several topics).
            start, end = trim(text, 0, len(text))
            pieces = [["semantic", start, end, False]] if end > start else []
        else:
            pieces = partition(text, build_heading_tree(text, headings), chunk_size)
            pieces = attach_heading_only(pieces, text, {p for p, _, _ in headings}, chunk_size)
        for piece in pieces:
            if piece[0] == "semantic":
                offset = piece[1]
                piece.append([(offset + a, offset + b) for a, b in
                              split_into_sentences(text[piece[1]:piece[2]], chunk_size)])
        plans.append((document, headings, pieces))

    # Embed only the units of oversized leaves, in a single batched call.
    texts = [document.page_content[a:b] for document, _, pieces in plans
             for piece in pieces if piece[0] == "semantic" for a, b in piece[4]]
    vectors = np.array(embedding_model.embed_documents(texts)) if texts else np.empty((0, 0))
    if len(vectors):
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    chunks, offset = [], 0
    for document, headings, pieces in plans:
        text = document.page_content
        spans = []  # [start, end, intro] for every final chunk of this document, in order
        for piece in pieces:
            if piece[0] == "whole":
                spans.append(piece[1:4])
                continue

            units = piece[4]
            embeddings = vectors[offset:offset + len(units)]
            offset += len(units)
            for first, last in semantic_breaks(units, embeddings, chunk_size, breakpoint_percentile,
                                               [is_heading_line(text, unit) for unit in units],
                                               min_size):
                spans.append([units[first][0], units[last][1], False])

        for start, end, _ in pack_intros(spans, chunk_size):
            chunks.append(make_span_chunk(document, headings, start, end))

    return add_chunk_metadata(chunks, "semantic")


def is_heading_line(text, unit):
    content = text[unit[0]:unit[1]].strip()
    return "\n" not in content and HEADING_RE.match(content) is not None


def semantic_breaks(units, embeddings, chunk_size, breakpoint_percentile, heading_units,
                    min_size=MIN_SEMANTIC_CHUNK_SIZE):
    """Return (first, last) unit-index ranges for one piece, aiming for
    min_size <= chunk <= chunk_size. """
    length = lambda a, b: units[b][1] - units[a][0]
    n = len(units)

    # distance[i - 1] = topic shift at the gap before unit i, comparing the mean of the
    # 2 units before the gap with the 2 after it (smooths noise, centred on the gap).
    distances = np.array([
        1 - cosine(embeddings[max(0, i - 2):i].mean(axis=0), embeddings[i:i + 2].mean(axis=0))
        for i in range(1, n)
    ])
    # Breakpoint candidates: gaps with at least min_size of text on both sides of the piece.
    # Edge gaps compare fewer units, have inflated distances, and could only produce a
    # sub-minimum chunk, so they don't take part in the percentile.
    candidate = [length(0, i - 1) >= min_size and length(i, n - 1) >= min_size for i in range(1, n)]
    candidate_distances = distances[candidate] if len(distances) else distances
    threshold = np.percentile(candidate_distances, breakpoint_percentile) if len(candidate_distances) else np.inf

    def cuttable(i):  # gap before unit i; never leave a heading line stranded at a chunk end
        return not heading_units[i - 1]

    def segments(cuts):
        bounds = [0] + sorted(cuts) + [n]
        return [(bounds[k], bounds[k + 1] - 1) for k in range(len(bounds) - 1)]

    # 1. Topic boundaries: strong (percentile) gaps, strongest first, accepted only if every
    #    resulting segment keeps at least min_size characters.
    cuts = []
    strong = [i for i in range(1, n) if candidate[i - 1] and distances[i - 1] >= threshold and cuttable(i)]
    for i in sorted(strong, key=lambda i: -distances[i - 1]):
        if all(length(a, b) >= min_size for a, b in segments(cuts + [i])):
            cuts.append(i)

    # 2. Size limit: a segment over chunk_size is cut into the fewest roughly equal parts
    #    (at unit boundaries), instead of filling to chunk_size and leaving a tiny remainder.
    def size_split(a, b):
        total = length(a, b)
        if total <= chunk_size or a == b:
            return [(a, b)]
        k = -(-total // chunk_size)
        parts, first = [], a
        for j in range(1, k):
            target = units[a][0] + j * total / k
            options = [i for i in range(first + 1, b + 1) if cuttable(i)]
            if not options:
                break
            i = min(options, key=lambda i: abs(units[i][0] - target))
            parts.append((first, i - 1))
            first = i
        parts.append((first, b))
        return [p for part in parts for p in (size_split(*part) if part != (a, b) else [part])]

    ranges = [part for a, b in segments(cuts) for part in size_split(a, b)]

    # Merge remaining tiny chunks (e.g. a short tail after a forced size split) into a
    # neighbour within this same piece when the result still fits chunk_size.
    i = 0
    while i < len(ranges):
        a, b = ranges[i]
        if len(ranges) > 1 and length(a, b) < min_size:
            if i > 0 and length(ranges[i - 1][0], b) <= chunk_size:
                ranges[i - 1:i + 1] = [(ranges[i - 1][0], b)]
                i -= 1
                continue
            if i + 1 < len(ranges) and length(a, ranges[i + 1][1]) <= chunk_size:
                ranges[i:i + 2] = [(a, ranges[i + 1][1])]
                continue
        i += 1

    return ranges

def chunk_documents(documents, strategy, embedding_model=None, chunk_size=800, chunk_overlap=150):
    if strategy == "fixed":
        return fixed_chunking(documents, chunk_size, chunk_overlap)

    if strategy == "recursive":
        return recursive_chunking(documents, chunk_size, chunk_overlap)

    if strategy == "semantic":
        if embedding_model is None:
            raise ValueError("Semantic chunking requires an embedding model.")

        return semantic_chunking(
            documents,
            embedding_model,
            chunk_size
        )

    raise ValueError(f"Unknown strategy: {strategy}")

import json
import re
import PyPDF2

from .utils import get_number_of_pages, remove_fields


# ── Helpers ──────────────────────────────────────────────────────────────────

def _parse_pages(pages: str) -> list[int]:
    """Parse a pages string like '5-7', '3,8', or '12' into a sorted list of ints."""
    result = []
    for part in pages.split(','):
        part = part.strip()
        if '-' in part:
            start, end = int(part.split('-', 1)[0].strip()), int(part.split('-', 1)[1].strip())
            if start > end:
                raise ValueError(f"Invalid range '{part}': start must be <= end")
            result.extend(range(start, end + 1))
        else:
            result.append(int(part))
    return sorted(set(result))


def _count_pages(doc_info: dict) -> int:
    """Return total page count for a PDF document."""
    if doc_info.get('page_count'):
        return doc_info['page_count']
    if doc_info.get('pages'):
        return len(doc_info['pages'])
    return get_number_of_pages(doc_info['path'])


def _iter_structure_nodes(nodes: list[dict], depth: int = 0, path: list[str] | None = None):
    path = path or []
    for node in nodes or []:
        title = str(node.get('title') or '')
        node_path = path + ([title] if title else [])
        yield node, depth, node_path
        if node.get('nodes'):
            yield from _iter_structure_nodes(node['nodes'], depth + 1, node_path)


def _node_page_span(node: dict) -> tuple[int | None, int | None]:
    start = node.get('start_index', node.get('start_page', node.get('page')))
    end = node.get('end_index', node.get('end_page', start))
    try:
        start = int(start) if start is not None else None
        end = int(end) if end is not None else start
    except (TypeError, ValueError):
        return None, None
    return start, end


def _compact_structure_node(node: dict, depth: int, path: list[str], include_summary: bool = False, summary_chars: int = 240) -> dict:
    start, end = _node_page_span(node)
    item = {
        'node_id': node.get('node_id'),
        'title': node.get('title', ''),
        'start_page': start,
        'end_page': end,
        'depth': depth,
        'path': ' > '.join(path),
        'child_count': len(node.get('nodes') or []),
    }
    if include_summary and node.get('summary'):
        summary = str(node['summary'])
        item['summary'] = summary[:summary_chars] + ('...' if len(summary) > summary_chars else '')
    return item


_SEARCH_STOPWORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'by', 'for', 'from', 'in', 'is', 'of',
    'on', 'or', 'the', 'to', 'with',
}


def _query_groups(query: str) -> list[dict]:
    groups = []
    for chunk in re.split(r'[|,;\n]+', str(query or '')):
        phrase = chunk.strip().lower()
        terms = [
            term
            for term in re.findall(r'[a-z0-9]+', phrase)
            if term not in _SEARCH_STOPWORDS
        ]
        if terms:
            groups.append({'phrase': phrase, 'terms': terms})
    return groups


def _match_score(text: str, groups: list[dict]) -> int:
    lower_text = str(text or '').lower()
    if not lower_text:
        return 0

    score = 0
    for group in groups:
        phrase = group['phrase']
        terms = group['terms']
        if phrase and phrase in lower_text:
            score += 10 + len(terms)
            continue
        matched_terms = sum(1 for term in terms if term in lower_text)
        if matched_terms:
            score += matched_terms
        if matched_terms == len(terms):
            score += 5 + matched_terms
    return score


def _find_match_index(text: str, groups: list[dict]) -> int:
    lower_text = str(text or '').lower()
    for group in groups:
        phrase = group['phrase']
        if phrase:
            index = lower_text.find(phrase)
            if index >= 0:
                return index
        for term in group['terms']:
            index = lower_text.find(term)
            if index >= 0:
                return index
    return -1


def _snippet(text: str, groups: list[dict], context_chars: int = 240) -> str:
    text = str(text or '')
    if not text:
        return ''
    context_chars = max(40, int(context_chars or 240))
    index = _find_match_index(text, groups)
    if index < 0:
        return text[:context_chars] + ('...' if len(text) > context_chars else '')

    half = context_chars // 2
    start = max(0, index - half)
    end = min(len(text), index + half)
    prefix = '...' if start > 0 else ''
    suffix = '...' if end < len(text) else ''
    return prefix + text[start:end].strip() + suffix


def _coerce_fields(fields) -> list[str]:
    if fields is None:
        return ['title', 'summary']
    if isinstance(fields, str):
        return [field.strip() for field in fields.split(',') if field.strip()]
    return [str(field).strip() for field in fields if str(field).strip()]


def _format_page_range(page_nums: list[int]) -> str:
    if not page_nums:
        return ''
    if len(page_nums) == 1:
        return str(page_nums[0])
    if page_nums == list(range(page_nums[0], page_nums[-1] + 1)):
        return f'{page_nums[0]}-{page_nums[-1]}'
    return ','.join(str(page) for page in page_nums)


def _find_node_by_id(nodes: list[dict], node_id: str) -> dict | None:
    for node, _, _ in _iter_structure_nodes(nodes):
        if str(node.get('node_id')) == str(node_id):
            return node
    return None


def _find_node_by_title_query(nodes: list[dict], title_query: str) -> dict | None:
    groups = _query_groups(title_query)
    best = None
    best_score = 0
    for node, _, _ in _iter_structure_nodes(nodes):
        score = _match_score(node.get('title', ''), groups)
        if score > best_score:
            best = node
            best_score = score
    return best


def _get_pdf_page_content(doc_info: dict, page_nums: list[int]) -> list[dict]:
    """Extract text for specific PDF pages (1-indexed). Prefer cached pages, fallback to PDF."""
    cached_pages = doc_info.get('pages')
    if cached_pages:
        page_map = {p['page']: p['content'] for p in cached_pages}
        return [
            {'page': p, 'content': page_map[p]}
            for p in page_nums if p in page_map
        ]
    path = doc_info['path']
    with open(path, 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)
        total = len(pdf_reader.pages)
        valid_pages = [p for p in page_nums if 1 <= p <= total]
        return [
            {'page': p, 'content': pdf_reader.pages[p - 1].extract_text() or ''}
            for p in valid_pages
        ]


def _get_md_page_content(doc_info: dict, page_nums: list[int]) -> list[dict]:
    """
    For Markdown documents, 'pages' are line numbers.
    Find nodes whose line_num falls within [min(page_nums), max(page_nums)] and return their text.
    """
    min_line, max_line = min(page_nums), max(page_nums)
    results = []
    seen = set()

    def _traverse(nodes):
        for node in nodes:
            ln = node.get('line_num')
            if ln and min_line <= ln <= max_line and ln not in seen:
                seen.add(ln)
                results.append({'page': ln, 'content': node.get('text', '')})
            if node.get('nodes'):
                _traverse(node['nodes'])

    _traverse(doc_info.get('structure', []))
    results.sort(key=lambda x: x['page'])
    return results


# ── Tool functions ────────────────────────────────────────────────────────────

def get_document(documents: dict, doc_id: str) -> str:
    """Return JSON with document metadata: doc_id, doc_name, doc_description, type, status, page_count (PDF) or line_count (Markdown)."""
    doc_info = documents.get(doc_id)
    if not doc_info:
        return json.dumps({'error': f'Document {doc_id} not found'})
    result = {
        'doc_id': doc_id,
        'doc_name': doc_info.get('doc_name', ''),
        'doc_description': doc_info.get('doc_description', ''),
        'type': doc_info.get('type', ''),
        'status': 'completed',
    }
    if doc_info.get('type') == 'pdf':
        result['page_count'] = _count_pages(doc_info)
    else:
        result['line_count'] = doc_info.get('line_count', 0)
    return json.dumps(result)


def get_document_structure(documents: dict, doc_id: str) -> str:
    """Return tree structure JSON with text fields removed (saves tokens)."""
    doc_info = documents.get(doc_id)
    if not doc_info:
        return json.dumps({'error': f'Document {doc_id} not found'})
    structure = doc_info.get('structure', [])
    structure_no_text = remove_fields(structure, fields=['text'])
    return json.dumps(structure_no_text, ensure_ascii=False)


def get_document_outline(
    documents: dict,
    doc_id: str,
    depth: int = 4,
    limit: int = 100,
    include_summary: bool = False,
) -> str:
    """Return a compact outline of PageIndex nodes without dumping the full tree."""
    doc_info = documents.get(doc_id)
    if not doc_info:
        return json.dumps({'error': f'Document {doc_id} not found'})

    depth = max(0, int(depth or 4))
    limit = max(1, int(limit or 100))
    nodes = []
    total = 0
    for node, node_depth, path in _iter_structure_nodes(doc_info.get('structure', [])):
        if node_depth > depth:
            continue
        total += 1
        if len(nodes) < limit:
            nodes.append(_compact_structure_node(node, node_depth, path, include_summary=include_summary))

    return json.dumps(
        {
            'nodes': nodes,
            'total_nodes_at_depth': total,
            'truncated': total > len(nodes),
        },
        ensure_ascii=False,
    )


def search_document_structure(
    documents: dict,
    doc_id: str,
    query: str,
    fields=None,
    limit: int = 10,
    context_chars: int = 240,
) -> str:
    """Search titles/summaries in the PageIndex tree and return compact matching nodes."""
    doc_info = documents.get(doc_id)
    if not doc_info:
        return json.dumps({'error': f'Document {doc_id} not found'})

    groups = _query_groups(query)
    if not groups:
        return json.dumps({'error': 'query must contain at least one searchable term'})

    fields = _coerce_fields(fields)
    limit = max(1, int(limit or 10))
    matches = []
    for node, depth, path in _iter_structure_nodes(doc_info.get('structure', [])):
        matched_fields = []
        best_text = ''
        best_score = 0
        score = 0
        for field in fields:
            text = str(node.get(field) or '')
            field_score = _match_score(text, groups)
            if field_score <= 0:
                continue
            matched_fields.append(field)
            score += field_score
            if field_score > best_score:
                best_text = text
                best_score = field_score

        if score <= 0:
            continue

        item = _compact_structure_node(node, depth, path)
        item['matched_fields'] = matched_fields
        item['score'] = score
        item['snippet'] = _snippet(best_text, groups, context_chars=context_chars)
        matches.append(item)

    matches.sort(key=lambda item: (-item['score'], item['start_page'] or 0, item['title']))
    return json.dumps(
        {
            'query': query,
            'matches': matches[:limit],
            'total_matches': len(matches),
            'truncated': len(matches) > limit,
        },
        ensure_ascii=False,
    )


def get_section_content(
    documents: dict,
    doc_id: str,
    node_id: str | None = None,
    title_query: str | None = None,
    max_pages: int = 5,
) -> str:
    """Return page text for the PageIndex section identified by node_id or title_query."""
    doc_info = documents.get(doc_id)
    if not doc_info:
        return json.dumps({'error': f'Document {doc_id} not found'})
    if not node_id and not title_query:
        return json.dumps({'error': 'Provide node_id or title_query'})

    structure = doc_info.get('structure', [])
    node = _find_node_by_id(structure, node_id) if node_id else None
    if node is None and title_query:
        node = _find_node_by_title_query(structure, title_query)
    if node is None:
        return json.dumps({'error': 'No matching section found'})

    start_page, end_page = _node_page_span(node)
    if start_page is None or end_page is None:
        return json.dumps({'error': 'Matching section does not have page bounds'})
    if start_page > end_page:
        start_page, end_page = end_page, start_page

    max_pages = max(1, int(max_pages or 5))
    all_pages = list(range(start_page, end_page + 1))
    returned_pages = all_pages[:max_pages]
    content = _get_pdf_page_content(doc_info, returned_pages) if doc_info.get('type') == 'pdf' else _get_md_page_content(doc_info, returned_pages)

    return json.dumps(
        {
            'node_id': node.get('node_id'),
            'title': node.get('title', ''),
            'start_page': start_page,
            'end_page': end_page,
            'returned_pages': _format_page_range(returned_pages),
            'full_page_range': _format_page_range(all_pages),
            'truncated': len(returned_pages) < len(all_pages),
            'pages': content,
        },
        ensure_ascii=False,
    )


def search_page_content(
    documents: dict,
    doc_id: str,
    query: str,
    pages: str | None = None,
    limit: int = 10,
    context_chars: int = 240,
) -> str:
    """Search page text and return compact page-level snippets."""
    doc_info = documents.get(doc_id)
    if not doc_info:
        return json.dumps({'error': f'Document {doc_id} not found'})

    groups = _query_groups(query)
    if not groups:
        return json.dumps({'error': 'query must contain at least one searchable term'})

    try:
        if pages:
            page_nums = _parse_pages(pages)
        else:
            page_nums = list(range(1, _count_pages(doc_info) + 1))
    except (ValueError, AttributeError) as e:
        return json.dumps({'error': f'Invalid pages format: {pages!r}. Use "5-7", "3,8", or "12". Error: {e}'})

    try:
        content = _get_pdf_page_content(doc_info, page_nums) if doc_info.get('type') == 'pdf' else _get_md_page_content(doc_info, page_nums)
    except Exception as e:
        return json.dumps({'error': f'Failed to read page content: {e}'})

    limit = max(1, int(limit or 10))
    matches = []
    for page in content:
        text = page.get('content', '')
        score = _match_score(text, groups)
        if score <= 0:
            continue
        matches.append(
            {
                'page': page.get('page'),
                'score': score,
                'snippet': _snippet(text, groups, context_chars=context_chars),
            }
        )

    matches.sort(key=lambda item: (-item['score'], item['page'] or 0))
    return json.dumps(
        {
            'query': query,
            'searched_pages': pages or _format_page_range(page_nums),
            'matches': matches[:limit],
            'total_matches': len(matches),
            'truncated': len(matches) > limit,
        },
        ensure_ascii=False,
    )


def get_page_content(documents: dict, doc_id: str, pages: str) -> str:
    """
    Retrieve page content for a document.

    pages format: '5-7', '3,8', or '12'
    For PDF: pages are physical page numbers (1-indexed).
    For Markdown: pages are line numbers corresponding to node headers.

    Returns JSON list of {'page': int, 'content': str}.
    """
    doc_info = documents.get(doc_id)
    if not doc_info:
        return json.dumps({'error': f'Document {doc_id} not found'})

    try:
        page_nums = _parse_pages(pages)
    except (ValueError, AttributeError) as e:
        return json.dumps({'error': f'Invalid pages format: {pages!r}. Use "5-7", "3,8", or "12". Error: {e}'})

    try:
        if doc_info.get('type') == 'pdf':
            content = _get_pdf_page_content(doc_info, page_nums)
        else:
            content = _get_md_page_content(doc_info, page_nums)
    except Exception as e:
        return json.dumps({'error': f'Failed to read page content: {e}'})

    return json.dumps(content, ensure_ascii=False)

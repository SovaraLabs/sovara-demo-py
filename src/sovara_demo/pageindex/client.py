import os
import uuid
import json
import asyncio
import concurrent.futures

from .page_index import page_index
from .page_index_md import md_to_tree
from .retrieve import (
    get_document,
    get_document_outline,
    get_document_structure,
    get_page_content,
    get_section_content,
    search_document_structure,
    search_page_content,
)
from .utils import ConfigLoader, get_page_tokens, remove_fields

META_INDEX = "_meta.json"


def _extract_pdf_pages(file_path: str, model: str = None) -> list[dict]:
    return [
        {'page': i, 'content': text}
        for i, (text, _) in enumerate(get_page_tokens(file_path, model=model), 1)
    ]


def _normalize_retrieve_model(model: str) -> str:
    """Preserve supported Agents SDK prefixes and route other provider paths via LiteLLM."""
    passthrough_prefixes = ("litellm/", "openai/")
    if not model or "/" not in model:
        return model
    if model.startswith(passthrough_prefixes):
        return model
    return f"litellm/{model}"


class PageIndexClient:
    """
    A client for indexing and retrieving document content.
    Flow: index() -> get_document() / get_document_structure() / get_page_content()

    For agent-based QA, see examples/agentic_vectorless_rag_demo.py.
    """
    def __init__(self, api_key: str = None, model: str = None, cheap_model: str = None, healing_model: str = None, retrieve_model: str = None, workspace: str = None):
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        self.workspace = os.path.abspath(os.path.expanduser(workspace)) if workspace else None
        overrides = {}
        if model:
            overrides["model"] = model
        if cheap_model:
            overrides["cheap_model"] = cheap_model
        if healing_model:
            overrides["healing_model"] = healing_model
        if retrieve_model:
            overrides["retrieve_model"] = retrieve_model
        opt = ConfigLoader().load(overrides or None)
        self.model = opt.model
        self.cheap_model = opt.cheap_model
        self.healing_model = opt.healing_model or self.model
        self.retrieve_model = _normalize_retrieve_model(opt.retrieve_model or self.model)
        if self.workspace:
            os.makedirs(self.workspace, exist_ok=True)
        self.documents = {}
        if self.workspace:
            self._load_workspace()

    def index(self, file_path: str, mode: str = "auto") -> str:
        """Index a document. Returns a document_id."""
        # Persist a canonical absolute path so workspace reloads do not
        # reinterpret caller-relative paths against the workspace directory.
        file_path = os.path.abspath(os.path.expanduser(file_path))
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        doc_id = str(uuid.uuid4())
        ext = os.path.splitext(file_path)[1].lower()

        is_pdf = ext == '.pdf'
        is_md = ext in ['.md', '.markdown']

        if mode == "pdf" or (mode == "auto" and is_pdf):
            print(f"Indexing PDF: {file_path}")
            result = page_index(
                doc=file_path,
                model=self.model,
                cheap_model=self.cheap_model,
                healing_model=self.healing_model,
                if_add_node_summary='yes',
                if_add_node_text='yes',
                if_add_node_id='yes',
                if_add_doc_description='yes'
            )
            # Extract per-page text so queries don't need the original PDF
            pages = _extract_pdf_pages(file_path, model=self.model)

            self.documents[doc_id] = {
                'id': doc_id,
                'type': 'pdf',
                'path': file_path,
                'doc_name': result.get('doc_name', ''),
                'doc_description': result.get('doc_description', ''),
                'page_count': len(pages),
                'structure': result['structure'],
                'pages': pages,
            }

        elif mode == "md" or (mode == "auto" and is_md):
            print(f"Indexing Markdown: {file_path}")
            coro = md_to_tree(
                md_path=file_path,
                if_thinning=False,
                if_add_node_summary='yes',
                summary_token_threshold=200,
                model=self.model,
                if_add_doc_description='yes',
                if_add_node_text='yes',
                if_add_node_id='yes'
            )
            try:
                asyncio.get_running_loop()
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    result = pool.submit(asyncio.run, coro).result()
            except RuntimeError:
                result = asyncio.run(coro)
            self.documents[doc_id] = {
                'id': doc_id,
                'type': 'md',
                'path': file_path,
                'doc_name': result.get('doc_name', ''),
                'doc_description': result.get('doc_description', ''),
                'line_count': result.get('line_count', 0),
                'structure': result['structure'],
            }
        else:
            raise ValueError(f"Unsupported file format for: {file_path}")

        print(f"Indexing complete. Document ID: {doc_id}")
        if self.workspace:
            self._save_doc(doc_id)
        return doc_id

    @staticmethod
    def _make_meta_entry(doc: dict) -> dict:
        """Build a lightweight meta entry from a document dict."""
        entry = {
            'type': doc.get('type', ''),
            'doc_name': doc.get('doc_name', ''),
            'doc_description': doc.get('doc_description', ''),
            'path': doc.get('path', ''),
        }
        if doc.get('type') == 'pdf':
            entry['page_count'] = doc.get('page_count')
        elif doc.get('type') == 'md':
            entry['line_count'] = doc.get('line_count')
        return entry

    @staticmethod
    def _read_json(path) -> dict | None:
        """Read a JSON file, returning None on any error."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: corrupt {os.path.basename(path)}: {e}")
            return None

    def _save_doc(self, doc_id: str):
        doc = self.documents[doc_id].copy()
        # Strip text from structure nodes — redundant with pages (PDF only)
        if doc.get('structure') and doc.get('type') == 'pdf':
            doc['structure'] = remove_fields(doc['structure'], fields=['text'])
        path = os.path.join(self.workspace, f"{doc_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
        self._save_meta(doc_id, self._make_meta_entry(doc))
        # Drop heavy fields; will lazy-load on demand
        self.documents[doc_id].pop('structure', None)
        self.documents[doc_id].pop('pages', None)

    def _rebuild_meta(self) -> dict:
        """Scan individual doc JSON files and return a meta dict."""
        meta = {}
        for filename in os.listdir(self.workspace):
            if not filename.endswith(".json") or filename == META_INDEX:
                continue
            path = os.path.join(self.workspace, filename)
            doc = self._read_json(path)
            if doc and isinstance(doc, dict):
                meta[os.path.splitext(filename)[0]] = self._make_meta_entry(doc)
        return meta

    def _read_meta(self) -> dict | None:
        """Read and validate _meta.json, returning None on any corruption."""
        meta = self._read_json(os.path.join(self.workspace, META_INDEX))
        if meta is not None and not isinstance(meta, dict):
            print(f"Warning: {META_INDEX} is not a JSON object, ignoring")
            return None
        return meta

    def _save_meta(self, doc_id: str, entry: dict):
        meta = self._read_meta() or self._rebuild_meta()
        meta[doc_id] = entry
        meta_path = os.path.join(self.workspace, META_INDEX)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def _load_workspace(self):
        meta = self._read_meta()
        if meta is None:
            meta = self._rebuild_meta()
            if meta:
                print(f"Loaded {len(meta)} document(s) from workspace (legacy mode).")
        for doc_id, entry in meta.items():
            doc = dict(entry, id=doc_id)
            if doc.get('path') and not os.path.isabs(doc['path']):
                doc['path'] = os.path.abspath(os.path.join(self.workspace, doc['path']))
            self.documents[doc_id] = doc

    def _ensure_doc_loaded(self, doc_id: str):
        """Load full document JSON on demand (structure, pages, etc.)."""
        doc = self.documents.get(doc_id)
        if not doc or doc.get('structure') is not None:
            return
        full = self._read_json(os.path.join(self.workspace, f"{doc_id}.json"))
        if not full:
            return
        doc['structure'] = full.get('structure', [])
        if full.get('pages'):
            doc['pages'] = full['pages']

    def get_document(self, doc_id: str) -> str:
        """Return document metadata JSON."""
        return get_document(self.documents, doc_id)

    def get_document_structure(self, doc_id: str) -> str:
        """Return document tree structure JSON (without text fields)."""
        if self.workspace:
            self._ensure_doc_loaded(doc_id)
        return get_document_structure(self.documents, doc_id)

    def get_document_outline(self, doc_id: str, depth: int = 4, limit: int = 100, include_summary: bool = False) -> str:
        """Return compact document outline JSON."""
        if self.workspace:
            self._ensure_doc_loaded(doc_id)
        return get_document_outline(self.documents, doc_id, depth=depth, limit=limit, include_summary=include_summary)

    def search_document_structure(self, doc_id: str, query: str, fields=None, limit: int = 10, context_chars: int = 240) -> str:
        """Search document structure titles/summaries."""
        if self.workspace:
            self._ensure_doc_loaded(doc_id)
        return search_document_structure(
            self.documents,
            doc_id,
            query=query,
            fields=fields,
            limit=limit,
            context_chars=context_chars,
        )

    def get_section_content(
        self,
        doc_id: str,
        node_id: str | None = None,
        title_query: str | None = None,
        max_pages: int = 5,
    ) -> str:
        """Return page content for a document section."""
        if self.workspace:
            self._ensure_doc_loaded(doc_id)
        return get_section_content(
            self.documents,
            doc_id,
            node_id=node_id,
            title_query=title_query,
            max_pages=max_pages,
        )

    def search_page_content(
        self,
        doc_id: str,
        query: str,
        pages: str | None = None,
        limit: int = 10,
        context_chars: int = 240,
    ) -> str:
        """Search page text and return compact snippets."""
        if self.workspace:
            self._ensure_doc_loaded(doc_id)
        return search_page_content(
            self.documents,
            doc_id,
            query=query,
            pages=pages,
            limit=limit,
            context_chars=context_chars,
        )

    def get_page_content(self, doc_id: str, pages: str) -> str:
        """Return page content for the given pages string (e.g. '5-7', '3,8', '12')."""
        if self.workspace:
            self._ensure_doc_loaded(doc_id)
        return get_page_content(self.documents, doc_id, pages)

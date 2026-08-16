from __future__ import annotations
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import CitationQueryEngine
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import NodeWithScore
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from google.genai.types import EmbedContentConfig
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy import make_url
logger = logging.getLogger('satgas_ai.rag')

class RAGError(Exception):
    pass

class RAGConfigError(RAGError):
    pass

class RAGIngestionError(RAGError):
    pass

class RAGQueryError(RAGError):
    pass
KB_CATEGORIES = ('Pembelajaran', 'Penelitian', 'Etika Penggunaan AI', 'Administrasi Akademik')
DEFAULT_CATEGORY_MANIFEST: Dict[str, Dict[str, str]] = {'pembelajaran.txt': {'category': 'Pembelajaran'}, 'penelitian.txt': {'category': 'Penelitian'}, 'etika.txt': {'category': 'Etika Penggunaan AI'}, 'administrasi.txt': {'category': 'Administrasi Akademik'}}
DEFAULT_CATEGORY_KEYWORDS: Dict[str, str] = {'pembelajaran': 'Pembelajaran', 'belajar': 'Pembelajaran', 'penelitian': 'Penelitian', 'riset': 'Penelitian', 'etika': 'Etika Penggunaan AI', 'disclosure': 'Etika Penggunaan AI', 'administrasi': 'Administrasi Akademik', 'admin': 'Administrasi Akademik'}
DEFAULT_CATEGORY = 'Umum'
_NOISY_META_KEYS = ('file_path', 'file_type', 'file_size', 'creation_date', 'last_modified_date')

def _strip_nul_bytes(value: str) -> str:
    """Postgres TEXT/JSONB tidak bisa menyimpan byte NUL (0x00). Karakter ini kadang
    ikut terekstrak dari file .docx/.pdf yang punya konten biner/kontrol tersembunyi."""
    if not value:
        return value
    return value.replace("\x00", "")


def _sanitize_documents(documents: List[Document]) -> List[Document]:
    for doc in documents:
        if doc.text:
            doc.set_content(_strip_nul_bytes(doc.text))
        if doc.metadata:
            doc.metadata = {
                k: (_strip_nul_bytes(v) if isinstance(v, str) else v)
                for k, v in doc.metadata.items()
            }
    return documents


def _infer_category_from_filename(file_name: str, keyword_map: Optional[Dict[str, str]]=None, default: str=DEFAULT_CATEGORY) -> str:
    name = (file_name or '').lower()
    for keyword, category in (keyword_map or DEFAULT_CATEGORY_KEYWORDS).items():
        if keyword in name:
            return category
    return default

def _extract_leading_heading(text: str) -> Optional[str]:
    for line in (text or '').splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith('#'):
            return stripped.lstrip('#').strip() or None
        return None
    return None

def attach_category(documents: Sequence[Document], manifest: Optional[Dict[str, Dict[str, str]]]=None, keyword_map: Optional[Dict[str, str]]=None, default_category: str=DEFAULT_CATEGORY) -> List[Document]:
    manifest = manifest or {}
    prepared: List[Document] = []
    for doc in documents:
        meta = doc.metadata or {}
        file_name = meta.get('file_name') or meta.get('file_path') or ''
        entry = manifest.get(file_name, {})
        category = entry.get('category') or _infer_category_from_filename(file_name, keyword_map, default_category)
        section = entry.get('section') or _extract_leading_heading(doc.text)
        doc.metadata['category'] = category
        if section:
            doc.metadata['section'] = section
        for key in _NOISY_META_KEYS:
            if key not in doc.excluded_embed_metadata_keys:
                doc.excluded_embed_metadata_keys.append(key)
            if key not in doc.excluded_llm_metadata_keys:
                doc.excluded_llm_metadata_keys.append(key)
        prepared.append(doc)
        logger.debug('attach_category: %s -> kategori=%s section=%s', file_name, category, section)
    return prepared

@dataclass
class RAGConfig:
    gemini_api_key: str
    database_url: str
    table_name: str = 'satgas_ai_kb'
    llm_model: str = 'gemini-2.5-flash'
    embed_model: str = 'gemini-embedding-2-preview'
    embed_dim: int = 768
    chunk_size: int = 512
    chunk_overlap: int = 100
    similarity_top_k: int = 4
    temperature: float = 0.1
    request_timeout: float = 60.0
    hnsw_kwargs: Dict[str, Any] = field(default_factory=lambda: {'hnsw_m': 16, 'hnsw_ef_construction': 64, 'hnsw_ef_search': 40, 'hnsw_dist_method': 'vector_cosine_ops'})

    @classmethod
    def from_env(cls) -> 'RAGConfig':
        api_key = os.getenv('GEMINI_API_KEY', '').strip()
        db_url = os.getenv('DATABASE_URL', '').strip()
        missing = [name for name, val in (('GEMINI_API_KEY', api_key), ('DATABASE_URL', db_url)) if not val]
        if missing:
            raise RAGConfigError(f'Environment variable wajib belum diset: {', '.join(missing)}')
        return cls(gemini_api_key=api_key, database_url=db_url, table_name=os.getenv('RAG_TABLE_NAME', 'satgas_ai_kb'), llm_model=os.getenv('RAG_LLM_MODEL', 'gemini-2.5-flash'), embed_model=os.getenv('RAG_EMBED_MODEL', 'gemini-embedding-2-preview'), embed_dim=int(os.getenv('RAG_EMBED_DIM', '768')), similarity_top_k=int(os.getenv('RAG_TOP_K', '4')))

@dataclass
class Citation:
    marker: int
    document: str
    category: Optional[str] = None
    page: Optional[str] = None
    section: Optional[str] = None
    score: Optional[float] = None
    snippet: str = ''
    full_text: str = ''

@dataclass
class AnswerResult:
    answer: str
    citations: List[Citation]
    metrics: Dict[str, Any]
    no_answer: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {'answer': self.answer, 'citations': [{k: v for k, v in asdict(c).items() if k != 'full_text'} for c in self.citations], 'metrics': self.metrics, 'no_answer': self.no_answer}

    def grouped_citations(self) -> List[Dict[str, Any]]:
        grouped: Dict[tuple, Dict[str, Any]] = {}
        for c in self.citations:
            key = (c.document, c.section)
            if key not in grouped:
                grouped[key] = {'document': c.document, 'category': c.category, 'section': c.section, 'page': c.page, 'score': c.score, 'snippet': c.snippet, 'markers': []}
            grouped[key]['markers'].append(c.marker)
            if c.score is not None and (grouped[key]['score'] is None or c.score > grouped[key]['score']):
                grouped[key]['score'] = c.score
        return list(grouped.values())

@dataclass
class IngestResult:
    documents_loaded: int
    nodes_indexed: int
    table_name: str
    categories: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
_CITATION_QA_TEMPLATE_ID = PromptTemplate("Anda adalah AI Guideline Assistant untuk Satgas AI FIF. Jawablah pertanyaan HANYA berdasarkan sumber bernomor di bawah. Setiap klaim harus mencantumkan sumbernya memakai penanda nomor, mis. [1] atau [1][2]. Jika informasi tidak ditemukan pada sumber, jawab dengan jujur: 'Maaf, informasi tersebut tidak ditemukan pada dokumen panduan Satgas AI FIF.' Jangan mengarang di luar konteks. Jawab dalam Bahasa Indonesia yang jelas.\nContoh:\nSource 1: AI disclosure ditulis di akhir dokumen tugas.\nSource 2: Disclosure memuat nama tools dan bagian yang dibantu AI.\nPertanyaan: Bagaimana menulis AI disclosure?\nJawaban: Tuliskan disclosure di akhir dokumen tugas [1], memuat nama tools AI dan bagian yang dibantu [2].\n\nSekarang giliran Anda. Berikut sumber-sumber bernomor:\n------\n{context_str}\n------\nPertanyaan: {query_str}\nJawaban: ")
_CITATION_REFINE_TEMPLATE_ID = PromptTemplate('Anda punya jawaban awal untuk pertanyaan berikut, dan ada sumber tambahan. Perbaiki jawaban bila sumber baru relevan; tetap cantumkan penanda [nomor] dan jangan keluar dari konteks sumber.\nPertanyaan: {query_str}\nJawaban awal: {existing_answer}\nSumber tambahan bernomor:\n------\n{context_msg}\n------\nJawaban yang diperbaiki: ')
_NO_ANSWER_MARKER = 'tidak ditemukan pada dokumen panduan'

class RAGEngine:

    def __init__(self, config: RAGConfig) -> None:
        self.config = config
        self._configure_settings()
        self._vector_store: Optional[PGVectorStore] = None
        self._index: Optional[VectorStoreIndex] = None
        logger.info('RAGEngine siap (llm=%s, embed=%s, table=%s)', config.llm_model, config.embed_model, config.table_name)

    @classmethod
    def from_env(cls) -> 'RAGEngine':
        return cls(RAGConfig.from_env())

    def _configure_settings(self) -> None:
        try:
            Settings.llm = GoogleGenAI(model=self.config.llm_model, api_key=self.config.gemini_api_key, temperature=self.config.temperature)
            Settings.embed_model = GoogleGenAIEmbedding(model_name=self.config.embed_model, api_key=self.config.gemini_api_key, embedding_config=EmbedContentConfig(output_dimensionality=self.config.embed_dim))
            Settings.node_parser = SentenceSplitter(chunk_size=self.config.chunk_size, chunk_overlap=self.config.chunk_overlap)
        except Exception as exc:
            raise RAGConfigError(f'Gagal inisialisasi model Gemini: {exc}') from exc

    def _get_vector_store(self) -> PGVectorStore:
        if self._vector_store is not None:
            return self._vector_store
        try:
            url = make_url(self.config.database_url)
            self._vector_store = PGVectorStore.from_params(host=url.host, port=str(url.port or 5432), database=url.database, user=url.username, password=url.password, table_name=self.config.table_name, embed_dim=self.config.embed_dim, hnsw_kwargs=self.config.hnsw_kwargs)
            return self._vector_store
        except Exception as exc:
            raise RAGConfigError(f'Gagal konek ke pgvector/Supabase: {exc}') from exc

    def _get_index(self) -> VectorStoreIndex:
        if self._index is None:
            self._index = VectorStoreIndex.from_vector_store(self._get_vector_store())
        return self._index

    def ingest_directory(self, source_dir: str, manifest: Optional[Dict[str, Dict[str, str]]]=None) -> IngestResult:
        if not os.path.isdir(source_dir):
            raise RAGIngestionError(f'Folder tidak ditemukan: {source_dir}')
        try:
            documents = SimpleDirectoryReader(input_dir=source_dir, filename_as_id=True).load_data()
        except Exception as exc:
            raise RAGIngestionError(f'Gagal memuat dokumen dari {source_dir}: {exc}') from exc
        documents = attach_category(documents, manifest or DEFAULT_CATEGORY_MANIFEST)
        return self._index_documents(documents)

    def ingest_files(
        self,
        file_paths: Sequence[str],
        manifest: Optional[Dict[str, Dict[str, str]]] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> IngestResult:
        missing = [p for p in file_paths if not os.path.isfile(p)]
        if missing:
            raise RAGIngestionError(f'File tidak ditemukan: {missing}')
        try:
            documents = SimpleDirectoryReader(input_files=list(file_paths), filename_as_id=True).load_data()
        except Exception as exc:
            raise RAGIngestionError(f'Gagal memuat file: {exc}') from exc

        if extra_metadata:
            for document in documents:
                document.metadata.update(extra_metadata)

        documents = attach_category(documents, manifest or DEFAULT_CATEGORY_MANIFEST)
        return self._index_documents(documents)

    def ingest_documents(self, documents: Sequence[Document], manifest: Optional[Dict[str, Dict[str, str]]]=None, auto_category: bool=True) -> IngestResult:
        docs = list(documents)
        if auto_category:
            docs = attach_category(docs, manifest or DEFAULT_CATEGORY_MANIFEST)
        return self._index_documents(docs)

    def _index_documents(self, documents: List[Document]) -> IngestResult:
        if not documents:
            raise RAGIngestionError('Tidak ada dokumen untuk diindeks.')
        documents = _sanitize_documents(documents)
        try:
            storage_context = StorageContext.from_defaults(vector_store=self._get_vector_store())
            index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, show_progress=True)
            self._index = index
            node_count = len(index.docstore.docs) if index.docstore.docs else 0
            categories: Dict[str, int] = {}
            for doc in documents:
                cat = (doc.metadata or {}).get('category', DEFAULT_CATEGORY)
                categories[cat] = categories.get(cat, 0) + 1
            logger.info("Ingestion selesai: %d dokumen -> %d node ke '%s' | kategori=%s", len(documents), node_count, self.config.table_name, categories)
            return IngestResult(documents_loaded=len(documents), nodes_indexed=node_count, table_name=self.config.table_name, categories=categories)
        except RAGError:
            raise
        except Exception as exc:
            raise RAGIngestionError(f'Gagal mengindeks dokumen: {exc}') from exc

    def answer(self, question: str, top_k: Optional[int]=None) -> AnswerResult:
        if not question or not question.strip():
            raise RAGQueryError('Pertanyaan kosong.')
        k = top_k or self.config.similarity_top_k
        started = time.perf_counter()
        try:
            query_engine = CitationQueryEngine.from_args(index=self._get_index(), similarity_top_k=k, citation_qa_template=_CITATION_QA_TEMPLATE_ID, citation_refine_template=_CITATION_REFINE_TEMPLATE_ID, citation_chunk_size=1024)
            response = query_engine.query(question.strip())
        except Exception as exc:
            raise RAGQueryError(f'Gagal menghasilkan jawaban: {exc}') from exc
        latency_ms = int((time.perf_counter() - started) * 1000)
        answer_text = str(response).strip()
        citations = self._extract_citations(response.source_nodes)
        no_answer = _NO_ANSWER_MARKER in answer_text.lower()
        result = AnswerResult(answer=answer_text, citations=citations, no_answer=no_answer, metrics={'retrieval_count': len(response.source_nodes), 'latency_ms': latency_ms, 'model': self.config.llm_model, 'top_k': k})
        logger.info('Query dijawab dalam %d ms (%d sumber, no_answer=%s)', latency_ms, len(citations), no_answer)
        return result

    @staticmethod
    def _extract_citations(source_nodes: Sequence[NodeWithScore]) -> List[Citation]:
        citations: List[Citation] = []
        for i, sn in enumerate(source_nodes, start=1):
            meta = sn.node.metadata or {}
            full_text = sn.node.get_content() or ''
            citations.append(Citation(marker=i, document=meta.get('file_name') or meta.get('file_path') or 'dokumen_tak_dikenal', category=meta.get('category'), page=meta.get('page_label'), section=meta.get('section') or meta.get('header_path'), score=round(float(sn.score), 4) if sn.score is not None else None, snippet=full_text[:280] + '…' if len(full_text) > 280 else full_text, full_text=full_text))
        return citations
    
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    from dotenv import load_dotenv
    load_dotenv()
    engine = RAGEngine.from_env()
    # ingest = engine.ingest_directory('data')
    # print('INGEST:', ingest.to_dict())
    print('\n--- RAG Engine Satgas AI Siap! ---')
    while True:
        try:
            pertanyaan = input("\nPertanyaan (ketik 'keluar' untuk selesai): ").strip()
        except (EOFError, KeyboardInterrupt):
            print('\nSampai jumpa!')
            break
        if pertanyaan.lower() == 'keluar':
            print('Terima kasih, sampai jumpa!')
            break
        if not pertanyaan:
            continue
        try:
            res = engine.answer(pertanyaan)
        except RAGQueryError as exc:
            print(f'\n[Gagal menjawab] {exc}')
            continue
        if res.no_answer:
            print('\n(Informasi tidak ditemukan di knowledge base)')
        print('\nJAWABAN:\n', res.answer)
        print('\nSITASI (Sumber Informasi):')
        for g in res.grouped_citations():
            skor = f'{g['score']:.2f}' if g['score'] is not None else '-'
            markers = ''.join((f'[{m}]' for m in g['markers']))
            print(f'  {markers} {g['document']} — {g['section'] or g['category']} (skor {skor})')
        print('\nMETRIK:', res.metrics)
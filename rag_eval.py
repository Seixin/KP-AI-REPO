from __future__ import annotations
import json
import logging
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Tuple
from rag_engine import RAGEngine
from golden_dataset import GOLDEN_DATASET, build_samples, precision_recall_at_k
from rouge_eval import rouge_report
import psycopg2
from sqlalchemy import make_url
logger = logging.getLogger('satgas_ai.eval')
DEFAULT_TARGET_LATENCY_MS = 5000

@dataclass
class EvalReport:
    retrieval_rows: List[Dict[str, Any]]
    retrieval_summary: Dict[str, float]
    rouge_rows: List[Dict[str, Any]] = field(default_factory=list)
    rouge_summary: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path_prefix: str='laporan_evaluasi') -> None:
        with open(f'{path_prefix}.json', 'w', encoding='utf-8') as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=2)
        logger.info('Laporan JSON disimpan: %s.json', path_prefix)
        try:
            import pandas as pd
            if self.rouge_rows:
                pd.DataFrame(self.rouge_rows).to_csv(f'{path_prefix}_rouge.csv', index=False)
                logger.info('Tabel ROUGE disimpan: %s_rouge.csv', path_prefix)
            pd.DataFrame(self.retrieval_rows).to_csv(f'{path_prefix}_retrieval.csv', index=False)
        except ImportError:
            logger.info('pandas tidak terpasang -> lewati ekspor CSV.')

    def print_summary(self) -> None:
        print('\n==== RINGKASAN EVALUASI RAG — Satgas AI FIF ====')
        print('\n[Retrieval & Operasional] (tanpa LLM)')
        for key, val in self.retrieval_summary.items():
            print(f'  {key:<24}: {val}')
        if self.rouge_summary:
            print('\n[ROUGE — generation, tanpa LLM, bebas kuota]')
            for key, val in self.rouge_summary.items():
                print(f'  {key:<24}: {val}')
        print('================================================\n')

def _count_table_rows(engine: RAGEngine) -> int:
    url = make_url(engine.config.database_url)
    table = f'data_{engine.config.table_name}'
    try:
        conn = psycopg2.connect(host=url.host, port=url.port or 5432, dbname=url.database, user=url.username, password=url.password)
        try:
            with conn.cursor() as cur:
                cur.execute('SELECT to_regclass(%s)', (f'public.{table}',))
                if cur.fetchone()[0] is None:
                    return 0
                cur.execute(f'SELECT COUNT(*) FROM {table}')
                return int(cur.fetchone()[0])
        finally:
            conn.close()
    except Exception as exc:
        logger.warning('Tidak bisa mengecek isi tabel %s: %s', table, exc)
        return -1

def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)

def retrieval_report(samples: List[Dict[str, Any]], target_latency_ms: int=DEFAULT_TARGET_LATENCY_MS) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    rows: List[Dict[str, Any]] = []
    latencies: List[float] = []
    for s in samples:
        pr = precision_recall_at_k(s.get('_retrieved_sources', []), s.get('_expected_sources', []))
        f1 = _f1(pr['precision_at_k'], pr['recall_at_k'])
        latency = s.get('_latency_ms') or 0
        latencies.append(latency)
        rows.append({'question': s['user_input'], 'expected_category': s.get('_expected_category'), 'expected_sources': s.get('_expected_sources'), 'retrieved_sources': s.get('_retrieved_sources'), 'precision_at_k': pr['precision_at_k'], 'recall_at_k': pr['recall_at_k'], 'f1_at_k': f1, 'latency_ms': latency})
    n = len(rows) or 1
    within_target = sum((1 for x in latencies if x and x <= target_latency_ms))
    summary = {'avg_precision_at_k': round(sum((r['precision_at_k'] for r in rows)) / n, 4), 'avg_recall_at_k': round(sum((r['recall_at_k'] for r in rows)) / n, 4), 'avg_f1_at_k': round(sum((r['f1_at_k'] for r in rows)) / n, 4), 'avg_latency_ms': round(sum(latencies) / n, 1) if latencies else 0.0, 'p95_latency_ms': round(_percentile(latencies, 95), 1) if latencies else 0.0, 'max_latency_ms': max(latencies) if latencies else 0.0, 'target_latency_ms': float(target_latency_ms), 'pct_within_target': round(100 * within_target / n, 1)}
    return (rows, summary)

def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, int(round(pct / 100 * len(ordered) + 0.5)) - 1))
    return float(ordered[k])

def run_full_evaluation(engine: RAGEngine, target_latency_ms: int=DEFAULT_TARGET_LATENCY_MS) -> EvalReport:
    logger.info('Menjalankan %d pertanyaan golden lewat engine...', len(GOLDEN_DATASET))
    samples = build_samples(engine)
    retrieval_rows, retrieval_summary = retrieval_report(samples, target_latency_ms)
    rouge_rows, rouge_summary = rouge_report(samples)
    logger.info('ROUGE selesai (tanpa LLM): %s', rouge_summary)
    return EvalReport(retrieval_rows=retrieval_rows, retrieval_summary=retrieval_summary, rouge_rows=rouge_rows, rouge_summary=rouge_summary)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    from dotenv import load_dotenv
    load_dotenv()
    engine = RAGEngine.from_env()
    if os.getenv('EVAL_SKIP_INGEST', '0').strip() == '1':
        print('EVAL_SKIP_INGEST=1 -> ingest dilewati (dipaksa manual).')
    else:
        jumlah_baris = _count_table_rows(engine)
        if jumlah_baris > 0:
            print(f'Tabel "{engine.config.table_name}" sudah berisi {jumlah_baris} baris -> INGEST DILEWATI otomatis.')
            print('(Set FORCE_REINGEST=1 kalau memang mau ingest ulang / menambah data.)')
            if os.getenv('FORCE_REINGEST', '0').strip() == '1':
                print('FORCE_REINGEST=1 -> tetap ingest ulang...')
                engine.ingest_directory('data')
        else:
            print('Tabel kosong / tidak terdeteksi -> ingest data/ sekarang...')
            engine.ingest_directory('data')
    report = run_full_evaluation(engine)
    report.print_summary()
    report.save('laporan_evaluasi')

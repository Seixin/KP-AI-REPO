from __future__ import annotations
import re
from collections import Counter
from typing import Any, Dict, List, Tuple

def _tokenize(text: str) -> List[str]:
    return re.findall('\\w+', str(text).lower())

def _ngrams(tokens: List[str], n: int) -> Counter:
    if len(tokens) < n:
        return Counter()
    return Counter((tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)))

def _lcs_length(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        curr = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[len(b)]

def _prf(match: int, hyp_total: int, ref_total: int) -> Dict[str, float]:
    p = match / hyp_total if hyp_total else 0.0
    r = match / ref_total if ref_total else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {'precision': round(p, 4), 'recall': round(r, 4), 'f': round(f, 4)}

def rouge_n(hypothesis: str, reference: str, n: int) -> Dict[str, float]:
    h = _ngrams(_tokenize(hypothesis), n)
    r = _ngrams(_tokenize(reference), n)
    match = sum((h & r).values())
    return _prf(match, sum(h.values()), sum(r.values()))

def rouge_l(hypothesis: str, reference: str) -> Dict[str, float]:
    ht = _tokenize(hypothesis)
    rt = _tokenize(reference)
    lcs = _lcs_length(ht, rt)
    return _prf(lcs, len(ht), len(rt))

def score_pair(hypothesis: str, reference: str) -> Dict[str, Dict[str, float]]:
    return {'rouge1': rouge_n(hypothesis, reference, 1), 'rouge2': rouge_n(hypothesis, reference, 2), 'rougeL': rouge_l(hypothesis, reference)}

def rouge_report(samples: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    rows: List[Dict[str, Any]] = []
    for s in samples:
        hyp = s.get('response', '')
        ref = s.get('reference', '')
        sc = score_pair(hyp, ref)
        rows.append({'question': s.get('user_input', ''), 'rouge1_f': sc['rouge1']['f'], 'rouge2_f': sc['rouge2']['f'], 'rougeL_f': sc['rougeL']['f'], 'rouge1_p': sc['rouge1']['precision'], 'rouge1_r': sc['rouge1']['recall']})
    n = len(rows) or 1
    summary = {'avg_rouge1_f': round(sum((r['rouge1_f'] for r in rows)) / n, 4), 'avg_rouge2_f': round(sum((r['rouge2_f'] for r in rows)) / n, 4), 'avg_rougeL_f': round(sum((r['rougeL_f'] for r in rows)) / n, 4)}
    return (rows, summary)
if __name__ == '__main__':
    ref = 'AI disclosure ditulis di akhir dokumen tugas memuat nama tools dan bagian yang dibantu'
    hyp = 'Tuliskan AI disclosure di bagian akhir dokumen, sebutkan nama tools dan bagian yang dibantu AI'
    from pprint import pprint
    pprint(score_pair(hyp, ref))

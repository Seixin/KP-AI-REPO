from __future__ import annotations
import logging
import time
from typing import Any, Dict, List

logger = logging.getLogger('satgas_ai.golden_dataset')

GOLDEN_DATASET: List[Dict[str, Any]] = [
    {
        'id': 'q1_chatgpt_tugas', 
        'question': 'Apakah mahasiswa boleh menggunakan ChatGPT untuk tugas?', 
        'ground_truth': 'Boleh. Mahasiswa diperbolehkan menggunakan ChatGPT untuk mendukung proses belajar seperti brainstorming, memahami konsep, membuat ringkasan, atau memeriksa tata bahasa, selama AI menjadi alat bantu dan bukan pengganti proses berpikir. Hasil mentah AI tidak boleh diserahkan sebagai karya sendiri, dan jika AI dipakai untuk tugas yang dinilai, mahasiswa wajib mencantumkan AI disclosure.', 
        'reference_contexts': [
            'Mahasiswa diperbolehkan menggunakan asisten AI seperti ChatGPT untuk mendukung proses belajar, misalnya melakukan brainstorming ide, memahami konsep yang sulit, membuat ringkasan materi, atau memeriksa tata bahasa.', 
            'Menyerahkan hasil mentah dari AI sebagai karya sendiri tanpa proses berpikir tidak diperbolehkan. Jika AI digunakan untuk mengerjakan bagian dari tugas yang dinilai, mahasiswa wajib mencantumkan AI disclosure.'
        ], 
        'expected_sources': ['pembelajaran.txt'], 
        'expected_category': 'Pembelajaran'
    }, 
    {
        'id': 'q2_cara_disclosure', 
        'question': 'Bagaimana cara menulis AI disclosure?', 
        'ground_truth': 'AI disclosure ditulis di bagian akhir dokumen tugas atau paper, memuat nama tools AI yang digunakan, bagian pekerjaan yang dibantu AI, serta tanggal penggunaan, agar penilai dapat memverifikasi kontribusi AI terhadap karya.', 
        'reference_contexts': [
            'AI disclosure adalah pernyataan penggunaan AI yang ditulis di bagian akhir dokumen tugas atau paper. Disclosure yang baik memuat nama tools AI yang digunakan, bagian pekerjaan yang dibantu AI, serta tanggal penggunaan.'
        ], 
        'expected_sources': ['etika.txt'], 
        'expected_category': 'Etika Penggunaan AI'
    }, 
    {
        'id': 'q3_contoh_penggunaan_diperbolehkan', 
        'question': 'Apa saja contoh penggunaan AI yang diperbolehkan?', 
        'ground_truth': 'Contoh penggunaan yang diperbolehkan antara lain: untuk pembelajaran (brainstorming ide, memahami konsep, membuat ringkasan, memeriksa tata bahasa); untuk penelitian (merangkum paper, tinjauan literatur awal, menyusun kerangka, brainstorming ide riset); dan untuk administrasi (menyusun draft surat, merangkum notulen, membuat rekap laporan, menstandarkan format dokumen).', 
        'reference_contexts': [
            'Mahasiswa diperbolehkan menggunakan asisten AI seperti ChatGPT untuk mendukung proses belajar, misalnya melakukan brainstorming ide, memahami konsep yang sulit, membuat ringkasan materi, atau memeriksa tata bahasa.', 
            'Dalam penelitian, AI generatif dapat dimanfaatkan untuk merangkum paper, membantu tinjauan literatur awal, menyusun kerangka tulisan, dan brainstorming ide riset.', 
            'Tenaga kependidikan dan dosen dapat menggunakan AI untuk mempercepat pekerjaan administrasi, seperti menyusun draft surat, merangkum notulen rapat, membuat rekap laporan, dan menstandarkan format dokumen.'
        ], 
        'expected_sources': ['pembelajaran.txt', 'penelitian.txt', 'administrasi.txt'], 
        'expected_category': 'Pembelajaran'
    }, 
    {
        'id': 'q4_risiko_genai_penelitian', 
        'question': 'Apa risiko penggunaan GenAI dalam penelitian?', 
        'ground_truth': 'Risiko utamanya adalah halusinasi berupa fakta atau sitasi yang tampak meyakinkan tetapi tidak nyata, bias dari data pelatihan, dan risiko kebocoran data bila data penelitian yang belum dipublikasikan dimasukkan ke layanan AI publik.', 
        'reference_contexts': [
            'Namun terdapat sejumlah risiko yang harus diwaspadai: AI dapat menghasilkan halusinasi berupa fakta atau sitasi yang tampak meyakinkan tetapi tidak nyata, memuat bias dari data pelatihan, serta menimbulkan risiko kebocoran data bila peneliti memasukkan data penelitian yang belum dipublikasikan ke layanan AI publik.'
        ], 
        'expected_sources': ['penelitian.txt'], 
        'expected_category': 'Penelitian'
    }, 
    {
        'id': 'q5_prompt_yang_baik', 
        'question': 'Bagaimana cara membuat prompt yang baik?', 
        'ground_truth': 'Tulis instruksi yang spesifik, sertakan konteks dan tujuan, berikan contoh bila perlu, lalu lakukan iterasi dengan memperbaiki prompt berdasarkan hasil yang diperoleh.', 
        'reference_contexts': [
            'Untuk membuat prompt yang baik, tuliskan instruksi yang spesifik, sertakan konteks dan tujuan, berikan contoh bila perlu, lalu lakukan iterasi dengan memperbaiki prompt berdasarkan hasil yang diperoleh.'
        ], 
        'expected_sources': ['pembelajaran.txt'], 
        'expected_category': 'Pembelajaran'
    }, 
    {
        'id': 'q6_ai_untuk_paper', 
        'question': 'Apakah AI boleh digunakan untuk membuat paper?', 
        'ground_truth': 'AI boleh membantu penulisan paper, misalnya memperbaiki susunan kalimat atau menyunting bahasa, tetapi tidak boleh dicantumkan sebagai penulis. Substansi ilmiah, analisis, dan kesimpulan tetap menjadi tanggung jawab peneliti, dan penggunaannya wajib diungkapkan melalui AI disclosure.', 
        'reference_contexts': [
            'AI boleh digunakan untuk membantu penulisan paper, misalnya memperbaiki susunan kalimat atau menyunting bahasa, tetapi AI tidak boleh dicantumkan sebagai penulis. Substansi ilmiah, analisis, dan kesimpulan tetap menjadi tanggung jawab penuh peneliti, dan setiap penggunaan AI wajib diungkapkan melalui AI disclosure sesuai ketentuan jurnal atau institusi.'
        ], 
        'expected_sources': ['penelitian.txt'], 
        'expected_category': 'Penelitian'
    }
]

def build_samples(engine, max_retries: int=3, retry_wait_s: float=8.0) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    skipped: List[str] = []
    
    for item in GOLDEN_DATASET:
        result = None
        last_exc: Exception | None = None
        
        for attempt in range(1, max_retries + 1):
            try:
                result = engine.answer(item['question'])
                break
            except Exception as exc:
                last_exc = exc
                if attempt < max_retries:
                    wait = retry_wait_s * attempt
                    logger.warning('[%s] percobaan %d/%d gagal (%s). Coba lagi dalam %.0f detik...', item['id'], attempt, max_retries, exc, wait)
                    time.sleep(wait)
                    
        if result is None:
            logger.error('[%s] DILEWATI setelah %d percobaan gagal. Error terakhir: %s', item['id'], max_retries, last_exc)
            skipped.append(item['id'])
            continue
            
        samples.append({
            'user_input': item['question'], 
            'response': result.answer, 
            'reference': item['ground_truth'], 
            '_expected_sources': item['expected_sources'], 
            '_expected_category': item['expected_category'], 
            '_retrieved_sources': [c.document for c in result.citations], 
            '_latency_ms': result.metrics.get('latency_ms')
        })
        
    if skipped:
        logger.warning('build_samples selesai dengan %d/%d pertanyaan DILEWATI: %s. Laporan tetap dibuat dari %d pertanyaan yang berhasil.', len(skipped), len(GOLDEN_DATASET), skipped, len(samples))
        
    return samples

def precision_recall_at_k(retrieved_sources: List[str], expected_sources: List[str]) -> Dict[str, float]:
    retrieved = list(dict.fromkeys(retrieved_sources))
    expected = set(expected_sources)
    
    if not retrieved:
        return {'precision_at_k': 0.0, 'recall_at_k': 0.0}
        
    hit = sum((1 for s in retrieved if s in expected))
    precision = hit / len(retrieved)
    recall = hit / len(expected) if expected else 0.0
    
    return {
        'precision_at_k': round(precision, 4), 
        'recall_at_k': round(recall, 4)
    }

if __name__ == '__main__':
    print(f'Total pertanyaan golden: {len(GOLDEN_DATASET)}')
    for item in GOLDEN_DATASET:
        print(f"- [{item['expected_category']:<22}] {item['question']}")
import os
import chromadb
from rank_bm25 import BM25Okapi
from pypdf import PdfReader

def embed(texts):
    return model.encode(texts, show_progress_bar=False).tolist()

def tokenize_vietnamese(text):
    return text.lower().split()

def chunk_text(text, chunk_size=1000, overlap=200):
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks, current = [], ""
    for p in paragraphs:
        while len(p) > chunk_size:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.append(p[:chunk_size].strip())
            p = p[chunk_size - overlap:]
        if current:
            if len(current) + len(p) + 1 <= chunk_size:
                current += p + "\n"
            else:
                chunks.append(current.strip())
                current = (current[-overlap:] + p + "\n") if overlap else (p + "\n")
        else:
            current = p + "\n"
    if current.strip():
        chunks.append(current.strip())
    return chunks

def process_pdf(uploaded_file):
    """Xử lý PDF và tạo index"""
    # Lưu file tạm
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    
    # ============================================================
    # 🔥 PHÂN TÍCH THÔNG TIN PDF
    # ============================================================
    pdf_info = analyze_pdf(tmp_path)
    # ============================================================
    
    # Đọc PDF
    reader = PdfReader(tmp_path)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text
    
    os.unlink(tmp_path)
    
    chunks = chunk_text(full_text)
    
    # Tạo ChromaDB (in-memory)
    client = chromadb.Client()
    collection = client.get_or_create_collection("rag")
    
    # Xóa dữ liệu cũ
    try:
        existing_ids = collection.get()['ids']
        if existing_ids:
            collection.delete(ids=existing_ids)
    except:
        pass
    
    # Thêm dữ liệu
    collection.add(
        ids=[str(i) for i in range(len(chunks))],
        documents=chunks,
        embeddings=embed(chunks)
    )
    
    # Tạo BM25
    tokenized_chunks = [tokenize_vietnamese(chunk) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    
    # ============================================================
    # 🔥 TRẢ VỀ pdf_info
    # ============================================================
    return collection, chunks, bm25, pdf_info

def semantic_search(collection, query, k=8):
    results = collection.query(
        query_embeddings=embed([query]),
        n_results=k
    )
    return results["documents"][0], [1 - d for d in results["distances"][0]]

def keyword_search(bm25, query, chunks, k=8):
    tokenized_query = tokenize_vietnamese(query)
    scores = bm25.get_scores(tokenized_query)
    indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [chunks[i] for i in indices], [scores[i] for i in indices]

def hybrid_search(collection, bm25, chunks, query, k=4, alpha=0.5):
    # Lấy kết quả từ cả 2
    sem_chunks, sem_scores = semantic_search(collection, query, k=8)
    kw_chunks, kw_scores = keyword_search(bm25, query, chunks, k=8)
    
    # Kết hợp
    combined = {}
    
    if sem_scores:
        max_sem = max(sem_scores)
        for chunk, score in zip(sem_chunks, sem_scores):
            norm_score = score / max_sem
            combined[chunk] = combined.get(chunk, 0) + (1 - alpha) * norm_score
    
    if kw_scores:
        max_kw = max(kw_scores)
        for chunk, score in zip(kw_chunks, kw_scores):
            norm_score = score / max_kw
            combined[chunk] = combined.get(chunk, 0) + alpha * norm_score
    
    sorted_results = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:k]
    return [chunk for chunk, _ in sorted_results]
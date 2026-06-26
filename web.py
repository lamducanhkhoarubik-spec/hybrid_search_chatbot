import streamlit as st
import tempfile
import os
import chromadb
import google.generativeai as genai
import time

# ============================
# 🔑 API KEY CÓ SẴN TRONG CODE
# ============================
GEMINI_API_KEY = ""  # <- Paste API key của bạn vào đây

# ============================
# CẤU HÌNH
# ============================
st.set_page_config(page_title="🤖 RAG Chatbot", layout="wide")
st.title("🤖 RAG Chatbot - Hỏi đáp tự động")

# ============================
# KHỞI TẠO SESSION STATE
# ============================
if "collection" not in st.session_state:
    st.session_state.collection = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ============================
# CÁC HÀM XỬ LÝ
# ============================
@st.cache_resource
def load_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

def embed(texts):
    return model.encode(texts, show_progress_bar=False).tolist()

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
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    
    from pypdf import PdfReader
    reader = PdfReader(tmp_path)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text
    
    os.unlink(tmp_path)
    
    chunks = chunk_text(full_text)
    
    client = chromadb.Client()
    collection = client.get_or_create_collection("rag")
    
    try:
        existing_ids = collection.get()['ids']
        if existing_ids:
            collection.delete(ids=existing_ids)
    except:
        pass
    
    collection.add(
        ids=[str(i) for i in range(len(chunks))],
        documents=chunks,
        embeddings=embed(chunks)
    )
    
    return collection, len(chunks)

def retrieve(collection, query, k=4):
    results = collection.query(
        query_embeddings=embed([query]),
        n_results=k
    )
    return results["documents"][0], results["distances"][0]

# ============================
# PROMPT + GEMINI (DÙNG API KEY CÓ SẴN)
# ============================
PROMPT_TEMPLATE = """
Bạn là trợ lý hỏi đáp. Dùng các đoạn ngữ cảnh dưới đây để trả lời câu hỏi.
Nếu ngữ cảnh không có thông tin, hãy nói là bạn không biết, đừng bịa.
Trả lời ngắn gọn, chính xác, bằng tiếng Việt.

Ngữ cảnh:
{context}

Câu hỏi: {question}

Trả lời:
"""

def answer_with_gemini(prompt):
    """Gọi Gemini API với key có sẵn trong code"""
    genai.configure(api_key=GEMINI_API_KEY)
    
    models = [
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-3.5-flash",
        "gemini-2.5-pro"
    ]
    
    for model_name in models:
        try:
            gemini_model = genai.GenerativeModel(model_name)
            response = gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            if "Quota exceeded" in str(e):
                st.warning(f"⚠️ Hết quota cho {model_name}, thử model khác...")
                continue
            else:
                raise e
    
    return "❌ Hết quota tất cả model. Vui lòng thử lại sau."

def rag(question, collection, k=4):
    context_chunks, distances = retrieve(collection, question, k)
    context = "\n\n".join(context_chunks)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    answer = answer_with_gemini(prompt)
    return answer, context_chunks, prompt

# ============================
# SIDEBAR - UPLOAD PDF
# ============================
with st.sidebar:
    st.header("📄 Tải lên PDF")
    uploaded_file = st.file_uploader("Chọn file PDF", type="pdf")
    
    if uploaded_file:
        if st.button("🔄 Xử lý PDF", use_container_width=True):
            with st.spinner("Đang xử lý PDF..."):
                collection, num_chunks = process_pdf(uploaded_file)
                st.session_state.collection = collection
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.chat_history = []
            st.success(f"✅ Xử lý xong {num_chunks} chunks")
    
    if st.session_state.pdf_name:
        st.info(f"📂 {st.session_state.pdf_name}")
        if st.session_state.collection:
            st.write(f"📊 {st.session_state.collection.count()} chunks")
    
    st.divider()
    st.caption("⚡ API Key đã được cấu hình sẵn")
    
    if st.button("🗑️ Xóa lịch sử", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# ============================
# MAIN CHAT
# ============================
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if question := st.chat_input("Nhập câu hỏi về nội dung PDF..."):
    if st.session_state.collection is None:
        st.warning("⚠️ Vui lòng upload PDF trước!")
    else:
        with st.chat_message("user"):
            st.write(question)
        st.session_state.chat_history.append({"role": "user", "content": question})
        
        with st.chat_message("assistant"):
            with st.spinner("🤔 Đang suy nghĩ..."):
                try:
                    answer, context_chunks, full_prompt = rag(
                        question, 
                        st.session_state.collection
                    )
                    
                    with st.expander("📚 Xem context đã tìm"):
                        for i, chunk in enumerate(context_chunks, 1):
                            st.write(f"**Chunk {i}:**")
                            st.write(chunk)
                    
                    with st.expander("📝 Xem prompt"):
                        st.code(full_prompt, language="text")
                    
                    st.write(answer)
                    
                except Exception as e:
                    st.error(f"❌ Lỗi: {e}")
                    st.info("💡 Kiểm tra API key hoặc kết nối internet")
        
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from processing_pdf import show_pdf_info
from model import process_pdf, hybrid_search
#=============================
#   PDF INFO
#=============================



# ============================
# 🔑 LẤY API KEY TỪ ENVIRONMENT
# ============================
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")

if not GEMINI_API_KEY:
    st.error("⚠️ Vui lòng cấu hình GEMINI_API_KEY trong Streamlit Secrets!")
    st.stop()

# ============================
# CẤU HÌNH
# ============================
st.set_page_config(page_title="📚 RAG Chatbot", layout="wide")
st.title("📚 RAG Chatbot - Hybrid Search")

# ============================
# KHỞI TẠO SESSION STATE
# ============================
if "collection" not in st.session_state:
    st.session_state.collection = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "bm25" not in st.session_state:
    st.session_state.bm25 = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False
# ============================================================
# 🔥 THÊM DÒNG NÀY: KHỞI TẠO pdf_info
# ============================================================
if "pdf_info" not in st.session_state:
    st.session_state.pdf_info = None
# ============================================================

# ============================
# LOAD MODEL (CACHE)
# ============================
@st.cache_resource
def load_embedding_model():
    """Load model embedding - chỉ load 1 lần"""
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_gemini():
    """Cấu hình Gemini - chỉ 1 lần"""
    genai.configure(api_key=GEMINI_API_KEY)
    return genai

# ============================
# CÁC HÀM XỬ LÝ
# ============================
model = load_embedding_model()
genai_model = load_gemini()



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
    """Gọi Gemini API"""
    try:
        gemini_model = genai.GenerativeModel('gemini-2.5-flash')
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        if "Quota exceeded" in str(e):
            return "⚠️ Hết quota API. Vui lòng thử lại sau."
        raise e

def rag_hybrid(question, collection, bm25, chunks, k=4, alpha=0.5):
    context_chunks = hybrid_search(collection, bm25, chunks, question, k, alpha)
    context = "\n\n".join(context_chunks)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    answer = answer_with_gemini(prompt)
    return answer, context_chunks, prompt

# ============================
# SIDEBAR
# ============================
with st.sidebar:
    st.header("📄 Tải lên PDF")
    uploaded_file = st.file_uploader("Chọn file PDF", type="pdf")
    
    if uploaded_file and not st.session_state.is_processing:
        if st.button("🔄 Xử lý PDF", use_container_width=True):
            st.session_state.is_processing = True
            with st.spinner("Đang xử lý PDF..."):
                # ============================================================
                # 🔥 SỬA: NHẬN ĐỦ 4 GIÁ TRỊ TRẢ VỀ
                # ============================================================
                collection, chunks, bm25, pdf_info = process_pdf(uploaded_file)
                # ============================================================
                
                st.session_state.collection = collection
                st.session_state.chunks = chunks
                st.session_state.bm25 = bm25
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.chat_history = []
                # ============================================================
                # 🔥 THÊM: LƯU pdf_info VÀO SESSION_STATE
                # ============================================================
                st.session_state.pdf_info = pdf_info
                # ============================================================
                
            st.session_state.is_processing = False
            st.success(f"✅ Xử lý xong {len(chunks)} chunks")
            
            # ============================================================
            # 🔥 THÊM: HIỂN THỊ THÔNG TIN PDF
            # ============================================================
            with st.expander("📊 Thông tin PDF", expanded=True):
                show_pdf_info(st.session_state.pdf_info)
            # ============================================================
    
    if st.session_state.pdf_name:
        st.info(f"📂 {st.session_state.pdf_name}")
        if st.session_state.collection:
            st.write(f"📊 {len(st.session_state.chunks)} chunks")
    
    st.divider()
    
    st.subheader("⚙️ Cấu hình Hybrid Search")
    alpha = st.slider(
        "🔀 Keyword vs Semantic",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1
    )
    k_results = st.slider("📝 Số kết quả", 2, 10, 4)
    
    if st.button("🗑️ Xóa lịch sử", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# ============================
# MAIN CHAT
# ============================
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if question := st.chat_input("Nhập câu hỏi..."):
    if st.session_state.collection is None:
        st.warning("⚠️ Vui lòng upload PDF trước!")
    else:
        with st.chat_message("user"):
            st.write(question)
        st.session_state.chat_history.append({"role": "user", "content": question})
        
        with st.chat_message("assistant"):
            with st.spinner("🤔 Đang suy nghĩ..."):
                try:
                    answer, context_chunks, full_prompt = rag_hybrid(
                        question,
                        st.session_state.collection,
                        st.session_state.bm25,
                        st.session_state.chunks,
                        k=k_results,
                        alpha=alpha
                    )
                    
                    with st.expander("📚 Xem context"):
                        for i, chunk in enumerate(context_chunks, 1):
                            st.write(f"**Chunk {i}:**")
                            st.write(chunk)
                    
                    with st.expander("📝 Xem prompt"):
                        st.code(full_prompt, language="text")
                    
                    st.write(answer)
                    
                except Exception as e:
                    st.error(f"❌ Lỗi: {e}")
        
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
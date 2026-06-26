import streamlit as st
from pypdf import PdfReader


def analyze_pdf(file_path):
    """Phân tích file PDF và trả về thông tin dạng dict"""
    try:
        reader = PdfReader(file_path)
        
        info = {
            "page_count": len(reader.pages),
            "metadata": {},
            "text_stats": {},
            "page_info": {}
        }
        
        # Metadata
        metadata = reader.metadata
        if metadata:
            info["metadata"] = {
                "author": metadata.author or "N/A",
                "title": metadata.title or "N/A",
                "subject": metadata.subject or "N/A",
                "creator": metadata.creator or "N/A",
                "producer": metadata.producer or "N/A",
                "creation_date": str(metadata.creation_date) if metadata.creation_date else "N/A",
                "modification_date": str(metadata.modification_date) if metadata.modification_date else "N/A"
            }
        
        # Trích xuất text
        full_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text
        
        # Thống kê text
        word_count = len(full_text.split())
        line_count = full_text.count('\n') + 1 if full_text else 0
        paragraph_count = len([p for p in full_text.split('\n\n') if p.strip()]) if full_text else 0
        char_count_no_space = len(full_text.replace(" ", "").replace("\n", "").replace("\r", ""))
        
        info["text_stats"] = {
            "total_chars": len(full_text),
            "chars_no_space": char_count_no_space,
            "word_count": word_count,
            "line_count": line_count,
            "paragraph_count": paragraph_count
        }
        
        # Page info
        page_sizes = []
        for page in reader.pages:
            box = page.mediabox
            page_sizes.append((float(box.width), float(box.height)))
        
        all_same = all(size == page_sizes[0] for size in page_sizes) if page_sizes else True
        
        info["page_info"] = {
            "first_page_size": page_sizes[0] if page_sizes else "N/A",
            "all_same_size": all_same,
            "avg_chars_per_page": len(full_text) / info["page_count"] if info["page_count"] > 0 else 0,
            "avg_words_per_page": word_count / info["page_count"] if info["page_count"] > 0 else 0
        }
        
        info["is_encrypted"] = reader.is_encrypted
        
        # Attachments & Outline
        info["attachments"] = len(reader.attachments) if hasattr(reader, 'attachments') and reader.attachments else 0
        info["outline"] = len(reader.outline) if hasattr(reader, 'outline') and reader.outline else 0
        
        info["success"] = True
        return info
        
    except FileNotFoundError:
        return {"success": False, "error": f"File '{file_path}' not found."}
    except Exception as e:
        return {"success": False, "error": str(e)}

def show_pdf_info(info):
    """Hiển thị thông tin PDF lên Streamlit"""
    if not info.get("success", False):
        st.error(f"❌ {info.get('error', 'Unknown error')}")
        return
    
    st.success("✅ Phân tích PDF thành công!")
    
    # Tạo 2 cột
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📄 Thông tin cơ bản")
        st.metric("Số trang", info["page_count"])
        st.metric("Tổng ký tự", f"{info['text_stats']['total_chars']:,}")
        st.metric("Số từ", f"{info['text_stats']['word_count']:,}")
        st.metric("Số dòng", f"{info['text_stats']['line_count']:,}")
        st.metric("Số đoạn", f"{info['text_stats']['paragraph_count']:,}")
    
    with col2:
        st.subheader("📊 Thống kê")
        st.metric("Ký tự (không space)", f"{info['text_stats']['chars_no_space']:,}")
        st.metric("Trung bình ký tự/trang", f"{info['page_info']['avg_chars_per_page']:.1f}")
        st.metric("Trung bình từ/trang", f"{info['page_info']['avg_words_per_page']:.1f}")
        st.metric("Mã hóa", "🔒 Có" if info["is_encrypted"] else "🔓 Không")
        
        if info["page_info"]["first_page_size"] != "N/A":
            w, h = info["page_info"]["first_page_size"]
            st.metric("Kích thước trang", f"{w:.0f} x {h:.0f}")
    
    # Metadata
    with st.expander("📋 Metadata"):
        meta = info["metadata"]
        st.write(f"**Tác giả:** {meta.get('author', 'N/A')}")
        st.write(f"**Tiêu đề:** {meta.get('title', 'N/A')}")
        st.write(f"**Chủ đề:** {meta.get('subject', 'N/A')}")
        st.write(f"**Người tạo:** {meta.get('creator', 'N/A')}")
        st.write(f"**Nhà sản xuất:** {meta.get('producer', 'N/A')}")
        st.write(f"**Ngày tạo:** {meta.get('creation_date', 'N/A')}")
        st.write(f"**Ngày sửa:** {meta.get('modification_date', 'N/A')}")
    
    # Thông tin bổ sung
    with st.expander("📎 Thông tin bổ sung"):
        st.write(f"**Tất cả trang cùng kích thước:** {'✅ Có' if info['page_info']['all_same_size'] else '❌ Không'}")
        st.write(f"**File đính kèm:** {info.get('attachments', 0)} file")
        st.write(f"**Bookmarks/Outline:** {info.get('outline', 0)} items")

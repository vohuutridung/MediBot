import os
import streamlit as st
from core_logic import load_rag_pipeline, get_rag_response

# --- Giao diện Streamlit ---
st.set_page_config(page_title="Cẩm Nang Y Khoa Local", layout="wide")
st.title("⚕️ Cẩm Nang Y Khoa (Local & Privacy-focused)")
st.info("Chào mừng bạn! Hệ thống này chạy hoàn toàn trên máy tính của bạn, đảm bảo an toàn và bảo mật. Hãy đặt câu hỏi về các bệnh tim mạch.")

# Tải pipeline RAG (chỉ tải 1 lần nhờ caching của Streamlit)
# Hàm load_rag_pipeline đã có @st.cache_resource
try:
    qa_chain = load_rag_pipeline()
except Exception as e:
    st.error(f"Không thể khởi tạo hệ thống. Vui lòng kiểm tra lại. Lỗi: {e}")
    st.stop()

# Khởi tạo session state để lưu lịch sử chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị các tin nhắn cũ từ lịch sử
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Nhận input mới từ người dùng
if prompt := st.chat_input("Hỏi tôi về triệu chứng, nguyên nhân của bệnh tim..."):
    # Hiển thị câu hỏi của người dùng lên giao diện
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Lấy câu trả lời từ bot và hiển thị
    with st.chat_message("assistant"):
        with st.spinner("Bác sĩ AI đang suy nghĩ..."):
            # **ĐIỂM SỬA CHÍNH YẾU 1**
            # Bây giờ get_rag_response trả về một dictionary duy nhất
            result = get_rag_response(qa_chain, prompt)
            
            # Lấy câu trả lời và nguồn từ dictionary
            response = result.get("result", "Không có câu trả lời.")
            st.markdown(response)
            
            # **ĐIỂM SỬA CHÍNH YẾU 2**
            # Hiển thị nguồn tham khảo một cách chính xác
            source_docs = result.get("source_documents", [])
            if source_docs:
                with st.expander("Xem nguồn tham khảo"):
                    for i, doc in enumerate(source_docs):
                        # Lấy metadata từ đối tượng Document
                        source = doc.metadata.get('source', 'Không rõ nguồn')
                        section = doc.metadata.get('section', 'Không rõ mục')
                        
                        st.info(f"**Nguồn #{i+1}:** Bệnh '{source}' - Mục: '{section}'")
                        # (Tùy chọn) Hiển thị một phần nội dung của chunk
                        # st.write(f"> {doc.page_content[:150]}...")
                        
    # Thêm câu trả lời của bot vào lịch sử chat
    st.session_state.messages.append({"role": "assistant", "content": response})
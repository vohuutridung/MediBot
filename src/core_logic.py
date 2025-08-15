import logging
import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA

# --- Logging setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

for noisy in [
    "sentence_transformers",
    "faiss",
    "langchain",
    "langchain_community",
    "langchain_ollama",
    "urllib3",
    "httpx"
]:
    logging.getLogger(noisy).setLevel(logging.WARNING)

# --- App Config ---
VECTOR_STORE_PATH = "data/processed/faiss_index_medical"
EMBEDDING_MODEL_NAME = "bkai-foundation-models/vietnamese-bi-encoder"
# LOCAL_MODEL_NAME = "vinallama_sic:latest"
LOCAL_MODEL_NAME = "tinyllama:latest"

DEFAULT_TEMPLATE = """
Bạn là một trợ lý y khoa hữu ích và chuyên nghiệp. Nhiệm vụ của bạn là trả lời câu hỏi của người dùng **chỉ dựa trên những thông tin được cung cấp trong phần ngữ cảnh dưới đây**.  

- Tuyệt đối **không tự bịa đặt** hoặc thêm thông tin ngoài ngữ cảnh.  
- Nếu không tìm thấy câu trả lời trong ngữ cảnh, hãy **thẳng thắn nói rằng bạn không có đủ thông tin để trả lời**.  
- Luôn **nhắc nhở người dùng rằng thông tin chỉ mang tính tham khảo và họ nên tham vấn ý kiến bác sĩ chuyên khoa để được chẩn đoán và điều trị chính xác**.

---

**Ngữ cảnh cung cấp:**  
{context}

**Câu hỏi của người dùng:**  
{question}

---

**Yêu cầu:**  
Hãy viết câu trả lời tiếng Việt thật tự nhiên, rõ ràng, chính xác và dễ hiểu nhất có thể, tuân thủ các nguyên tắc trên.
"""

# --- Utility Loaders (cached separately) ---
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

@st.cache_resource(show_spinner="Loading vector store...")
def load_vector_store(embedding_model_name):
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
    try:
        return FAISS.load_local(
            VECTOR_STORE_PATH,
            embedding_model,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        logger.error(f"Error loading FAISS index: {e}")
        st.error("Không load được FAISS index. Vui lòng kiểm tra dữ liệu vector.")
        return None
    
@st.cache_resource(show_spinner="Loading LLM via Ollama...")
def load_llm():
    try:
        return OllamaLLM(model=LOCAL_MODEL_NAME)
    except Exception as e:
        logger.error(f"Error loading Ollama: {e}")
        st.error("Không load được mô hình qua Ollama. Vui lòng kiểm tra cài đặt.")
        return None
    
@st.cache_resource(show_spinner="Building RetrievalQA chain...")
def build_rag_chain(embedding_model_name, vector_store_path, llm_model_name):
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
    vector_store = FAISS.load_local(
        vector_store_path,
        embedding_model,
        allow_dangerous_deserialization=True
    )
    llm = OllamaLLM(model=llm_model_name)
    if not all([embedding_model, vector_store, llm]):
        return None
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=DEFAULT_TEMPLATE
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

# --- Full pipeline ---
@st.cache_resource(show_spinner="Loading full RAG pipeline...")
def load_rag_pipeline():
    logger.info("Starting RAG pipeline loading...")
    embedding_model = load_embedding_model()
    vector_store = load_vector_store(EMBEDDING_MODEL_NAME)
    llm = load_llm()
    qa_chain = build_rag_chain(
        embedding_model_name=EMBEDDING_MODEL_NAME,
        vector_store_path=VECTOR_STORE_PATH,
        llm_model_name=LOCAL_MODEL_NAME
    )
    logger.info("RAG pipeline loaded.")
    return qa_chain

# --- Query Function ---
# def get_rag_response(qa_chain, query):
#     if qa_chain is None:
#         st.error("Hệ thống chưa sẵn sàng. Vui lòng kiểm tra cấu hình.")
#         return "", []

#     try:
#         result = qa_chain.invoke(query)
#         answer = result.get("result", "").strip()
#         sources = []
        
#         for doc in result.get("source_documents", []):
#             meta = doc.metadata
            
#             # Lấy metadata chính xác theo cách tạo trong code xử lý
#             disease_name = meta.get("source", "Unknown")  # "source" chứa tên bệnh
#             section_info = meta.get("section", "Unknown")  # "section" chứa tên section + câu đầu
            
#             # Tách section name từ section_info (phần trước dấu ":")
#             section_name = section_info.split(":")[0].strip() if ":" in section_info else section_info
            
#             source_info = {
#                 "disease": disease_name,           # Tên bệnh từ trường "source"
#                 "section": section_name,          # Tên section (đã tách khỏi câu đầu)
#                 "section_full": section_info,     # Section đầy đủ (bao gồm câu đầu)
#                 "source": disease_name            # Giữ lại để tương thích
#             }
#             sources.append(source_info)

#         return answer, sources
        
#     except Exception as e:
#         logger.error(f"Error during QA invocation: {e}")
#         st.error("Đã xảy ra lỗi khi truy vấn. Vui lòng thử lại.")
#         return "", []
    
def get_rag_response(qa_chain, query: str):
    """
    Nhận một câu hỏi, thực thi RAG chain và trả về một dictionary kết quả.
    Hàm này được thiết kế để luôn trả về một dictionary nhất quán.
    """
    try:
        # qa_chain.invoke sẽ tự động trả về một dictionary
        result_dict = qa_chain.invoke({"query": query})
        return result_dict
    except Exception as e:
        print(f"Lỗi khi thực thi RAG chain: {e}")
        # Trả về một dictionary lỗi để code trong app.py hoặc evaluate.py không bị crash
        return {
            "query": query,
            "result": "Xin lỗi, đã có lỗi xảy ra trong quá trình xử lý. Vui lòng thử lại.",
            "source_documents": []
        }
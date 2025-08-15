import os
import re
from bs4 import BeautifulSoup, NavigableString, Tag
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
import time

# --- CONFIG ---
# Lấy đường dẫn tuyệt đối của thư mục chứa script hiện tại (process_data.py)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Đi ngược lên 2 cấp để đến thư mục gốc của dự án (Medical_Chatbot)
PROJECT_ROOT = os.path.join(CURRENT_DIR, '..')

# <<< QUAN TRỌNG: SỬA LẠI ĐƯỜNG DẪN NÀY CHO ĐÚNG >>>
# DATA_SOURCE_DIR = "data/raw/Corpus"
# VECTOR_STORE_PATH = "data/processed/faiss_index_medical"
# EMBEDDING_MODEL_NAME = "bkai-foundation-models/vietnamese-bi-encoder"

DATA_SOURCE_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "Corpus")
VECTOR_STORE_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "faiss_index_medical")
EMBEDDING_MODEL_NAME = "bkai-foundation-models/vietnamese-bi-encoder"

# Danh sách bệnh tim mạch mục tiêu
TARGET_DISEASES = [
    'benh-ho-van-tim.html', 'benh-lao-phoi.html', 'khan-tieng.html', 'nhoi-mau-co-tim-khong-st-chenh-len.html',
    'nhoi-mau-co-tim-that-phai.html', 'nhoi-mau-nao.html', 'suy-gian-tinh-mach-chi-duoi.html',
    'suy-gian-tinh-mach-sau-chi-duoi.html', 'suy-ho-hap.html', 'suy-tim-giai-doan-cuoi.html',
    'suy-tim-man-tinh.html', 'suy-tim-mat-bu.html', 'suy-tim-phai.html', 'suy-tim-sung-huyet.html',
    'suy-tim-trai.html', 'suy-tim.html', 'suy-tinh-mach-man-tinh.html', 'thieu-mau-co-tim-cuc-bo-man-tinh.html',
    'thieu-mau-co-tim.html', 'tim-dap-nhanh.html', 'ung-thu-phoi-khong-te-bao-nho-giai-doan-1.html',
    'ung-thu-phoi-khong-te-bao-nho-giai-doan-2.html', 'ung-thu-phoi-khong-te-bao-nho-giai-doan-3.html',
    'ung-thu-phoi.html', 'ung-thu-thanh-quan.html', 'ung-thu-thuc-quan.html', 'ung-thu-vom-hong-giai-doan-0.html',
    'ung-thu-vom-hong-giai-doan-1.html', 'ung-thu-vom-hong-giai-doan-2.html', 'ung-thu-vom-hong-giai-doan-3.html',
    'ung-thu-vom-hong-giai-doan-dau.html', 'ung-thu-vom-hong.html', 'viem-amidan-hoc-mu.html', 'viem-amidan-man-tinh.html',
    'viem-amidan.html', 'viem-phe-quan-phoi.html', 'viem-phoi-do-metapneumovirus.html', 'viem-phoi.html',
    'viem-thanh-quan.html', 'xo-phoi.html', 'xo-vua-dong-mach-vanh.html', 'xo-vua-dong-mach.html',
]

# TARGET_DISEASES = [
#     'suy-tim-sung-huyet.html'
# ]

def clean_text(text):
    text = re.sub(r'\s*\n\s*', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def extract_content_from_html(filepath):
    CUTOFF_STRING = "HỆ THỐNG BỆNH VIỆN ĐA KHOA TÂM ANH"
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'lxml')
            # Bỏ bọc <strong>
            for tag in soup.find_all('strong'):
                tag.unwrap()
            for tag in soup.find_all('em'):
                tag.unwrap()
            for tag in soup.find_all('b'):
                tag.unwrap()

            print(soup)
    except Exception as e:
        print(f"  Lỗi khi đọc file {filepath}: {e}")
        return None, []

    # Lấy tên bệnh từ h1
    disease_name_tag = soup.find('h1')
    raw_name = disease_name_tag.get_text(strip=True) if disease_name_tag else "Không rõ tên bệnh"
    disease_name = raw_name.split(':')[0].strip()
    print(f"DEBUG: H1 tag found for {os.path.basename(filepath)}: {disease_name}")

    structured_content = []
    h2_tags = soup.find_all('h2')

    overview_content = []
    if disease_name_tag:
        current = disease_name_tag.find_next_sibling()
        while current:
            if isinstance(current, Tag) and current.name == 'h3' and 'mục lục' in current.get_text(strip=True).lower():
                break
            if isinstance(current, Tag):
                if current.find(['script', 'style', 'nav', 'iframe']):
                    current = current.next_sibling
                    continue
                text = current.get_text(separator=' ', strip=True)
                if text:
                    overview_content.append(text)
            elif isinstance(current, NavigableString):
                if current.strip():
                    overview_content.append(current.strip())
            current = current.next_sibling

        if overview_content:
            structured_content.append({
                'section': 'Tổng quan',
                'content': clean_text('\n'.join(overview_content))
            })

    if h2_tags:
        for h2 in h2_tags:
            section_title = h2.get_text(strip=True)
            section_content = []
            current = h2

            # Duyệt từng sibling tiếp theo cho đến khi gặp <h2> khác hoặc CUTOFF_STRING
            while True:
                current = current.next_sibling
                if current is None:
                    break

                # Cắt nếu chứa chuỗi không mong muốn
                if isinstance(current, NavigableString) and CUTOFF_STRING in current:
                    break
                if isinstance(current, Tag) and CUTOFF_STRING in current.get_text():
                    break

                # Nếu gặp h2 mới thì kết thúc section hiện tại
                if isinstance(current, Tag) and current.name == 'h2':
                    break

                # Xử lý nội dung có thể là Tag hoặc plain text
                if isinstance(current, NavigableString):
                    if current.strip():
                        section_content.append(current.strip())
                elif isinstance(current, Tag):
                    # Bỏ thẻ không mong muốn
                    if current.find(['script', 'style', 'nav', 'iframe']):
                        continue
                    text = current.get_text(separator=' ', strip=True)
                    if text:
                        section_content.append(text)

            content_text = clean_text('\n'.join(section_content).strip())
            print("")
            print("section:", section_title)
            print("content:", content_text)

            if content_text:
                structured_content.append({
                    'section': section_title,
                    'content': content_text
                })

    else:
        # Nếu không có thẻ h2 nào thì gom nội dung sau h1
        section_content = []
        for sibling in disease_name_tag.find_next_siblings():
            if CUTOFF_STRING in str(sibling):
                break
            if isinstance(sibling, Tag):
                if sibling.find(['script', 'style', 'nav', 'iframe']):
                    continue
                if not sibling.get_text(strip=True):
                    continue
                text = sibling.get_text(separator=' ', strip=True)
                if text:
                    section_content.append(text)
            elif isinstance(sibling, NavigableString):
                if sibling.strip():
                    section_content.append(sibling.strip())

        cleaned = clean_text('\n'.join(section_content).strip())
        if cleaned:
            structured_content.append({
                'section': "Nội dung chính",
                'content': cleaned
            })

    return disease_name, structured_content


def load_and_process_data():
    all_chunks = []

    print(f"Bắt đầu quét thư mục: '{DATA_SOURCE_DIR}'")
    if not os.path.exists(DATA_SOURCE_DIR):
        print(f"LỖI: Thư mục '{DATA_SOURCE_DIR}' không tồn tại.")
        return []

    for disease_filename in TARGET_DISEASES:
        filepath = os.path.join(DATA_SOURCE_DIR, disease_filename)
        if os.path.exists(filepath):
            print(f"-> Đang xử lý file: {disease_filename}")
            disease_name, sections = extract_content_from_html(filepath)
            if not sections:
                print(f"  Cảnh báo: Không trích xuất được nội dung từ file {filepath}")
                continue
            for section_data in sections:
                section = section_data.get('section', '').strip()
                content = section_data.get('content', '').strip()
                if not section or not content:
                    continue

                # ✅ Tạo section mới = section gốc + câu đầu tiên
                first_sentence = content.split('. ')[0].strip()
                if not first_sentence.endswith('.'):
                    first_sentence += '.'
                section_full = f"{section}: {first_sentence}"

                # ✅ Tạo document
                full_text = f"Thông tin về bệnh {disease_name}, mục {section_full}: {content}"
                doc = Document(page_content=full_text, metadata={"source": disease_name, "section": section_full})
                all_chunks.append(doc)

                print("\n--- Chunk mới ---")
                print(f"Source     : {doc.metadata['source']}")
                print(f"Section    : {doc.metadata['section']}")
                print(f"Nội dung   :\n{doc.page_content}")
                print("--- Hết chunk ---\n")

        else:
            print(f"  Cảnh báo: Không tìm thấy file '{disease_filename}'")
    
    print(f"\n=> Đã tạo tổng cộng {len(all_chunks)} chunks.")
    return all_chunks


def create_vector_store(chunks):
    print("\nBắt đầu quá trình embedding với mô hình local...")
    start_time = time.time()
    
    # Sử dụng mô hình embedding local
    model_kwargs = {'device': 'cpu'} # Chạy trên CPU, đổi thành 'cuda' nếu có GPU
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    vector_store = FAISS.from_documents(chunks, embedding=embeddings)
    print("vector_store: " + str(vector_store))
    
    end_time = time.time()
    print(f"=> Embedding hoàn tất trong {end_time - start_time:.2f} giây.")
    
    vector_store.save_local(VECTOR_STORE_PATH)
    print(f"=> Đã lưu Vector Store vào thư mục: {VECTOR_STORE_PATH}")


if __name__ == '__main__':
    processed_chunks = load_and_process_data()
    if processed_chunks:
        create_vector_store(processed_chunks)
    else:
        print("\nKhông có chunk nào được tạo. Vui lòng kiểm tra lại cấu trúc thư mục và code.")
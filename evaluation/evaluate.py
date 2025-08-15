import sys
import os
import getpass
import time
import torch
import psutil
import pandas as pd
from ragas import EvaluationDataset
from ragas import evaluate
from ragas.metrics import (
    ContextPrecision, ContextRecall, Faithfulness, ResponseRelevancy, FactualCorrectness, SemanticSimilarity, AnswerAccuracy, AnswerCorrectness
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from dotenv import load_dotenv
from src.core_logic import load_rag_pipeline, get_rag_response
from langchain_together import Together
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chat_models import init_chat_model

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# LLM_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
LLM_MODEL = "gemini-2.0-flash-lite"
EMBEDDING_MODEL_NAME = "bkai-foundation-models/vietnamese-bi-encoder"

# --- Setup llm and embeddings ---
def setup_llm_and_embeddings():
    if not os.environ.get("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key: ")
    
    try:
        # llm_raw = init_chat_model(LLM_MODEL, model_provider="together")
        llm_raw = init_chat_model(LLM_MODEL, model_provider="google_genai")
        llm = LangchainLLMWrapper(langchain_llm=llm_raw)

        embeddings_raw = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        embeddings = LangchainEmbeddingsWrapper(embeddings=embeddings_raw)
        return llm, embeddings
    
    except Exception as e:
        print(f"Lỗi khi setup LLM/Embeddings: {e}")
        raise
    
# --- Main eval function ---
def run_evaluation():
    print("--- BẮT ĐẦU QUÁ TRÌNH ĐÁNH GIÁ HỆ THỐNG RAG ---")

    try:
        qa_chain = load_rag_pipeline()
    except Exception as e:
        print(f"Lỗi khi tải pipeline: {e}")
        return
    
    eval_df = pd.read_csv("evaluation/evaluation_dataset.csv")
    print(f"Đã tải {len(eval_df)} câu hỏi từ bộ dữ liệu đánh giá.")

    total_latency = 0 # end2end latency
    results = []
    print("\nĐang chạy RAG trên bộ dữ liệu...")
    for index, row in eval_df.iterrows():
        start_time = time.time()
        answer, sources = get_rag_response(qa_chain, row["question"])
        end_time = time.time()
        
        latency = end_time - start_time
        total_latency += latency

        retrieved_contexts = []
        for src in sources:
            disease = src.get("disease", "Unknown")
            section = src.get("section", "Unknown") 
            section_full = src.get("section_full", "Unknown")

            context_info = f"Bệnh: {disease}, Mục: {section_full}"

            retrieved_contexts.append(context_info)
 
        results.append({
            "user_input": row["question"], # query
            "reference": row["ground_truth"], # ground-truth answer
            "response": answer, # RAG's output to the query
            "retrieved_contexts": retrieved_contexts # contexts retrieved from DB
        })
        print(f"  - Hoàn thành câu hỏi {index + 1}/{len(eval_df)} trong {latency:.2f}s")
        # print(f"Response là: {answer}")
        # print(f"Reference là: {row['ground_truth']}")
        
    avg_latency = total_latency / len(eval_df)
    print(f"\nThời gian phản hồi trung bình: {avg_latency:.2f} giây/câu hỏi.")

    results_df = pd.DataFrame(results)
    ragas_dataset = EvaluationDataset.from_pandas(results_df)

    print("\nĐang tính toán các chỉ số RAGAs...")
    metrics = [ContextPrecision(), ContextRecall(), Faithfulness(), ResponseRelevancy(), AnswerAccuracy()]
    try:
        load_dotenv()
        llm, embeddings = setup_llm_and_embeddings()

        result = evaluate(
            dataset=ragas_dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings
        )
        # result["answer_correctness"] = (result["factual_correctness(mode=f1)"] + result["semantic_similarity"]) / 2

        print("\n--- KẾT QUẢ ĐÁNH GIÁ ---")
        print(result)

    except Exception as e:
        print(f"\nLỗi khi chạy RAGAs: {e}")

    print("\n--- KẾT QUẢ HIỆU NĂNG ---")
    print(f"Thời gian phản hồi trung bình: {avg_latency:.2f} giây")



if __name__ == '__main__':
    run_evaluation()

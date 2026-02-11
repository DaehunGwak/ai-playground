"""LangGraph 기반 Milvus RAG 검색 스크립트

사용법:
    uv run scripts/rag_search.py "푸리에 변환이란 무엇인가?"
    uv run scripts/rag_search.py "음악 동기화 알고리즘" --collection music_processing_book
    uv run scripts/rag_search.py "STFT 윈도우 함수" --top-k 5 --chapter "Chapter 2"
"""

import argparse
from dataclasses import dataclass, field
from typing import Annotated

import ollama
from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from pymilvus import MilvusClient

EMBEDDING_MODEL = "qwen3-embedding:8b"
EMBEDDING_DIM = 4096
LLM_MODEL = "qwen3:14b"
MILVUS_URI = "http://localhost:19530"


@dataclass
class RAGState:
    query: str = ""
    collection: str = "music_processing_book"
    top_k: int = 3
    chapter_filter: str = ""
    documents: list[dict] = field(default_factory=list)
    answer: str = ""


def retrieve(state: RAGState) -> dict:
    """Milvus에서 유사 문서 검색"""
    client = MilvusClient(uri=MILVUS_URI)
    client.load_collection(state.collection)

    # 쿼리 임베딩
    response = ollama.embed(model=EMBEDDING_MODEL, input=[state.query])
    query_vector = response.embeddings[0]

    # 검색 파라미터
    search_params = {"metric_type": "COSINE"}
    filter_expr = f'chapter == "{state.chapter_filter}"' if state.chapter_filter else ""

    results = client.search(
        collection_name=state.collection,
        data=[query_vector],
        limit=state.top_k,
        output_fields=["text", "heading", "chapter", "chunk_index"],
        search_params=search_params,
        filter=filter_expr if filter_expr else None,
    )

    documents = []
    for hit in results[0]:
        doc = hit["entity"]
        doc["score"] = hit["distance"]
        documents.append(doc)

    return {"documents": documents}


def generate(state: RAGState) -> dict:
    """검색된 문서 기반으로 LLM 응답 생성"""
    if not state.documents:
        return {"answer": "관련 문서를 찾지 못했습니다."}

    # 컨텍스트 구성
    context_parts = []
    for i, doc in enumerate(state.documents, 1):
        context_parts.append(
            f"[문서 {i}] (챕터: {doc['chapter']}, 섹션: {doc['heading']}, 유사도: {doc['score']:.3f})\n{doc['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    llm = ChatOllama(model=LLM_MODEL)
    messages = [
        HumanMessage(content=f"""아래 문서를 참고하여 질문에 답변해주세요.
답변은 한국어로 작성하되, 전문 용어는 원문 그대로 사용하세요.
문서에 없는 내용은 추측하지 마세요.

## 참고 문서
{context}

## 질문
{state.query}
""")
    ]

    response = llm.invoke(messages)
    return {"answer": response.content}


# LangGraph 구성
def build_graph():
    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


def main():
    parser = argparse.ArgumentParser(description="LangGraph RAG 검색")
    parser.add_argument("query", type=str, help="검색 질문")
    parser.add_argument("--collection", type=str, default="music_processing_book", help="Milvus 컬렉션")
    parser.add_argument("--top-k", type=int, default=3, help="검색 문서 수")
    parser.add_argument("--chapter", type=str, default="", help="챕터 필터 (예: 'Chapter 2')")
    args = parser.parse_args()

    app = build_graph()

    initial_state = RAGState(
        query=args.query,
        collection=args.collection,
        top_k=args.top_k,
        chapter_filter=args.chapter,
    )

    print(f"질문: {args.query}")
    if args.chapter:
        print(f"필터: {args.chapter}")
    print(f"검색 대상: {args.collection} (top-{args.top_k})")
    print("-" * 60)

    result = app.invoke(initial_state)

    # 검색 결과 출력
    print("\n[검색된 문서]")
    for i, doc in enumerate(result["documents"], 1):
        print(f"  {i}. [{doc['chapter']}] {doc['heading']} (유사도: {doc['score']:.3f})")

    print("\n" + "=" * 60)
    print(f"\n{result['answer']}")


if __name__ == "__main__":
    main()

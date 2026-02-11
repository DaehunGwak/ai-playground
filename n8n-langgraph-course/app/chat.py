"""Music Processing 교재 RAG 챗봇 (Streamlit + LangGraph)

실행:
    uv run streamlit run app/chat.py
"""

from dataclasses import dataclass, field

import ollama
import streamlit as st
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from pymilvus import MilvusClient

EMBEDDING_MODEL = "qwen3-embedding:8b"
LLM_MODEL = "qwen3:14b"
MILVUS_URI = "http://localhost:19530"
DEFAULT_COLLECTION = "music_processing_book"


# --- LangGraph State & Nodes ---


@dataclass
class RAGState:
    query: str = ""
    collection: str = DEFAULT_COLLECTION
    top_k: int = 3
    chapter_filter: str = ""
    documents: list[dict] = field(default_factory=list)
    answer: str = ""


def retrieve(state: RAGState) -> dict:
    client = MilvusClient(uri=MILVUS_URI)
    client.load_collection(state.collection)

    response = ollama.embed(model=EMBEDDING_MODEL, input=[state.query])
    query_vector = response.embeddings[0]

    filter_expr = f'chapter == "{state.chapter_filter}"' if state.chapter_filter else None

    results = client.search(
        collection_name=state.collection,
        data=[query_vector],
        limit=state.top_k,
        output_fields=["text", "heading", "chapter", "chunk_index"],
        search_params={"metric_type": "COSINE"},
        filter=filter_expr,
    )

    documents = []
    for hit in results[0]:
        doc = hit["entity"]
        doc["score"] = hit["distance"]
        documents.append(doc)

    return {"documents": documents}


def generate(state: RAGState) -> dict:
    if not state.documents:
        return {"answer": "관련 문서를 찾지 못했습니다."}

    context_parts = []
    for i, doc in enumerate(state.documents, 1):
        context_parts.append(
            f"[문서 {i}] (챕터: {doc['chapter']}, 섹션: {doc['heading']})\n{doc['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    # 대화 히스토리 구성
    messages = []
    for msg in st.session_state.get("messages", [])[:-1]:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(HumanMessage(content=msg["content"]))

    messages.append(
        HumanMessage(content=f"""아래 문서를 참고하여 질문에 답변해주세요.
답변은 한국어로 작성하되, 전문 용어는 원문 그대로 사용하세요.
문서에 없는 내용은 추측하지 마세요.

## 참고 문서
{context}

## 질문
{state.query}
""")
    )

    llm = ChatOllama(model=LLM_MODEL)
    response = llm.invoke(messages)
    return {"answer": response.content}


@st.cache_resource
def build_graph():
    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


# --- Streamlit UI ---


st.set_page_config(page_title="Music Processing RAG", page_icon="🎵", layout="wide")
st.title("Music Processing 교재 RAG 챗봇")

# 사이드바 설정
with st.sidebar:
    st.header("설정")
    collection = st.text_input("컬렉션", value=DEFAULT_COLLECTION)
    top_k = st.slider("검색 문서 수", min_value=1, max_value=10, value=3)

    chapters = [
        "", "Chapter 1", "Chapter 2", "Chapter 3", "Chapter 4",
        "Chapter 5", "Chapter 6", "Chapter 7", "Chapter 8",
    ]
    chapter_filter = st.selectbox("챕터 필터", chapters, format_func=lambda x: "전체" if x == "" else x)

    if st.button("대화 초기화"):
        st.session_state.messages = []
        st.rerun()

# 메시지 히스토리
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "documents" in msg:
            with st.expander("참고 문서"):
                for i, doc in enumerate(msg["documents"], 1):
                    st.markdown(f"**{i}. [{doc['chapter']}] {doc['heading']}** (유사도: {doc['score']:.3f})")
                    st.text(doc["text"][:300] + "..." if len(doc["text"]) > 300 else doc["text"])
                    st.divider()

# 채팅 입력
if prompt := st.chat_input("질문을 입력하세요"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("검색 및 답변 생성 중..."):
            app = build_graph()
            state = RAGState(
                query=prompt,
                collection=collection,
                top_k=top_k,
                chapter_filter=chapter_filter,
            )
            result = app.invoke(state)

        st.markdown(result["answer"])

        if result["documents"]:
            with st.expander("참고 문서"):
                for i, doc in enumerate(result["documents"], 1):
                    st.markdown(f"**{i}. [{doc['chapter']}] {doc['heading']}** (유사도: {doc['score']:.3f})")
                    st.text(doc["text"][:300] + "..." if len(doc["text"]) > 300 else doc["text"])
                    st.divider()

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "documents": result["documents"],
    })

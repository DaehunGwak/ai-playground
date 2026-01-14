"""
LangGraph를 사용한 Gemini LLM 채팅 그래프 구성
"""
from typing import TypedDict, Annotated

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver


class State(TypedDict):
    """그래프 상태 정의"""
    messages: Annotated[list, add_messages]


def create_chat_graph(api_key: str, model_name: str = "gemini-pro"):
    """
    Gemini LLM을 사용하는 LangGraph 채팅 그래프 생성
    
    Args:
        api_key: Google Gemini API 키
        model_name: 사용할 모델 이름 (기본값: gemini-pro)
    
    Returns:
        컴파일된 LangGraph 그래프
    """
    # Gemini LLM 초기화
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.7,
    )
    
    # 그래프 노드 정의
    def chatbot(state: State):
        """채팅봇 응답 생성"""
        response = llm.invoke(state["messages"])
        return {"messages": [response]}
    
    # 그래프 구성
    graph = StateGraph(State)
    graph.add_node("chatbot", chatbot)
    graph.set_entry_point("chatbot")
    graph.add_edge("chatbot", END)
    
    # 메모리 체크포인터 추가하여 히스토리 저장
    memory = MemorySaver()
    
    # 그래프 컴파일 및 반환
    return graph.compile(checkpointer=memory)

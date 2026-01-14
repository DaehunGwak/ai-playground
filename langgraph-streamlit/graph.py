"""
LangGraph를 사용한 Gemini LLM 채팅 그래프 구성
"""
from typing import TypedDict, Annotated, Literal

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

from tools.higgsfield import create_higgsfield_tools


# 시스템 프롬프트 (Tool 사용 안내)
SYSTEM_PROMPT = """당신은 친절한 AI 어시스턴트입니다.

사용 가능한 도구가 있으면, 사용자의 요청에 따라 적절한 도구를 사용해야 합니다.

**이미지 생성 관련 도구:**
- generate_image: 사용자가 이미지를 생성해달라고 요청하면 반드시 이 도구를 사용하세요.
  - "이미지 생성해줘", "그림 그려줘", "이미지 만들어줘" 등의 요청에 사용
  - 도구가 자동으로 완료될 때까지 대기하고 결과를 반환합니다.

- check_image_status: 이전에 요청한 이미지 생성 상태를 확인할 때 사용
- cancel_image_generation: 진행 중인 이미지 생성을 취소할 때 사용

중요: 이미지 생성 요청 시, 직접 응답하지 말고 반드시 generate_image 도구를 호출하세요.
도구 결과를 받은 후에 사용자에게 결과를 설명해주세요."""


class State(TypedDict):
    """그래프 상태 정의"""
    messages: Annotated[list, add_messages]


def create_chat_graph(
    gemini_api_key: str, 
    model_name: str = "gemini-pro",
    higgsfield_api_key: str | None = None,
    higgsfield_api_secret: str | None = None
):
    """
    Gemini LLM을 사용하는 LangGraph 채팅 그래프 생성
    
    Args:
        gemini_api_key: Google Gemini API 키
        model_name: 사용할 모델 이름 (기본값: gemini-pro)
        higgsfield_api_key: Higgsfield API 키 (선택, 이미지 생성 기능 사용 시 필요)
        higgsfield_api_secret: Higgsfield API Secret (선택, 이미지 생성 기능 사용 시 필요)
    
    Returns:
        컴파일된 LangGraph 그래프
    """
    # Tool 목록 구성
    tools = []
    
    if higgsfield_api_key and higgsfield_api_secret:
        higgsfield_tools = create_higgsfield_tools(higgsfield_api_key, higgsfield_api_secret)
        tools.extend(higgsfield_tools)
    
    # Gemini LLM 초기화
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=gemini_api_key,
        temperature=0.7,
    )
    
    # Tool이 있으면 LLM에 바인딩
    if tools:
        llm_with_tools = llm.bind_tools(tools)
    else:
        llm_with_tools = llm
    
    # 시스템 프롬프트 (tools가 있을 때만 추가)
    system_message = SystemMessage(content=SYSTEM_PROMPT) if tools else None
    
    # 그래프 노드 정의
    def chatbot(state: State):
        """채팅봇 응답 생성"""
        messages = state["messages"]
        
        # 시스템 메시지가 없으면 추가
        if system_message and (not messages or not isinstance(messages[0], SystemMessage)):
            messages = [system_message] + list(messages)
        
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}
    
    def should_continue(state: State) -> Literal["tools", "__end__"]:
        """Tool 호출이 필요한지 판단"""
        messages = state["messages"]
        last_message = messages[-1]
        
        # Tool 호출이 있으면 tools 노드로
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        # 그렇지 않으면 종료
        return "__end__"
    
    # 그래프 구성
    graph = StateGraph(State)
    graph.add_node("chatbot", chatbot)
    
    if tools:
        # Tool 노드 추가
        tool_node = ToolNode(tools)
        graph.add_node("tools", tool_node)
        
        # 진입점 설정
        graph.set_entry_point("chatbot")
        
        # 조건부 엣지: chatbot -> tools 또는 END
        graph.add_conditional_edges(
            "chatbot",
            should_continue,
            {
                "tools": "tools",
                "__end__": END
            }
        )
        
        # tools -> chatbot (tool 실행 후 다시 chatbot으로)
        graph.add_edge("tools", "chatbot")
    else:
        # Tool이 없으면 단순 chatbot -> END
        graph.set_entry_point("chatbot")
        graph.add_edge("chatbot", END)
    
    # 메모리 체크포인터 추가하여 히스토리 저장
    memory = MemorySaver()
    
    # 그래프 컴파일 및 반환
    return graph.compile(checkpointer=memory)

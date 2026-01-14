import os
import uuid
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from graph import create_chat_graph

# 페이지 설정
st.set_page_config(
    page_title="LangGraph Chat",
    page_icon="💬",
    layout="wide"
)

# 헬퍼 함수: LangGraph state에서 메시지 가져오기
def get_messages():
    """LangGraph의 state에서 메시지 히스토리를 가져옵니다."""
    if not st.session_state.graph:
        return []
    
    try:
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        state = st.session_state.graph.get_state(config)
        
        messages = []
        if state.values.get("messages"):
            for msg in state.values["messages"]:
                if hasattr(msg, '__class__'):
                    msg_type = msg.__class__.__name__
                    if msg_type == "HumanMessage":
                        messages.append({"role": "user", "content": msg.content})
                    elif msg_type == "AIMessage":
                        messages.append({"role": "assistant", "content": msg.content})
        return messages
    except Exception:
        return []

# 세션 상태 초기화
if "graph" not in st.session_state:
    st.session_state.graph = None
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# 사이드바
with st.sidebar:
    st.title("💬 LangGraph Chat")
    st.divider()
    
    # API 키 설정
    api_key = st.text_input(
        "Google Gemini API 키",
        type="password",
        value=os.getenv("GOOGLE_API_KEY", ""),
        help="Google AI Studio에서 API 키를 발급받으세요: https://makersuite.google.com/app/apikey"
    )
    
    model_option = st.radio(
        "모델 선택",
        ["기본 모델", "직접 입력"],
        index=0,
        horizontal=True
    )
    
    if model_option == "기본 모델":
        model_name = st.selectbox(
            "모델",
            [
                "gemini-3-pro-preview",
                "gemini-3-flash-preview",
                "gemini-2.5-flash",
                "gemini-2.5-pro",
                "gemini-2.0-flash-exp",
                "gemini-2.0-flash-lite",
                "gemini-1.5-pro",
                "gemini-1.5-flash",
                "gemini-1.5-flash-8b",
                "gemini-pro",
            ],
            index=2,
            label_visibility="collapsed"
        )
    else:
        model_name = st.text_input(
            "모델 이름",
            value="gemini-2.5-pro",
            placeholder="예: gemini-3-pro-preview, gemini-2.0-flash-lite, gemini-1.5-flash-8b",
            help="사용 가능한 모델 이름을 입력하세요"
        )
    
    if api_key:
        if st.session_state.graph is None or st.session_state.get("current_api_key") != api_key or st.session_state.get("current_model") != model_name:
            try:
                with st.spinner("그래프 초기화 중..."):
                    st.session_state.graph = create_chat_graph(api_key, model_name)
                    st.session_state.current_api_key = api_key
                    st.session_state.current_model = model_name
                
                st.success("✅ 그래프가 초기화되었습니다!")
            except Exception as e:
                st.error(f"❌ 그래프 초기화 실패: {str(e)}")
                st.session_state.graph = None
    else:
        st.warning("⚠️ API 키를 입력해주세요")
        st.session_state.graph = None
    
    st.divider()
    
    if st.button("🗑️ 대화 기록 지우기", use_container_width=True):
        # 새로운 thread_id 생성으로 대화 초기화
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()
    
    st.divider()
    st.caption("LangGraph와 Streamlit을 사용한 채팅 인터페이스")

# 메인 채팅 인터페이스
st.title("💬 채팅")

# 메시지 히스토리 표시 (LangGraph state에서 가져오기)
messages = get_messages()
for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력
if prompt := st.chat_input("메시지를 입력하세요..."):
    # API 키 확인
    if not st.session_state.graph:
        st.error("⚠️ 사이드바에서 API 키를 입력해주세요.")
        st.stop()
    
    # 사용자 메시지 표시
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # AI 응답 생성
    with st.chat_message("assistant"):
        with st.spinner("생성 중..."):
            try:
                # LangGraph 실행 (checkpointer가 자동으로 히스토리 관리)
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                result = st.session_state.graph.invoke(
                    {"messages": [HumanMessage(content=prompt)]},
                    config=config
                )
                
                # 마지막 AI 응답 추출 및 표시
                if result.get("messages"):
                    last_message = result["messages"][-1]
                    response = last_message.content if hasattr(last_message, 'content') else str(last_message)
                else:
                    response = "응답을 생성할 수 없습니다."
                
                st.markdown(response)
                
                # 성공 시에만 페이지 리로드
                st.rerun()
                
            except Exception as e:
                error_msg = f"❌ 오류 발생: {str(e)}"
                st.error(error_msg)
                # 에러 발생 시에는 rerun하지 않아서 에러 메시지가 화면에 남도록 함

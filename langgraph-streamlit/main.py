import os
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from graph import create_chat_graph

# 페이지 설정
st.set_page_config(
    page_title="LangGraph Chat",
    page_icon="💬",
    layout="wide"
)

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "graph" not in st.session_state:
    st.session_state.graph = None

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
                "gemini-2.5-flash",
                "gemini-2.5-pro",
                "gemini-2.0-flash-exp",
                "gemini-1.5-pro",
                "gemini-1.5-flash",
                "gemini-pro",
            ],
            index=0,
            label_visibility="collapsed"
        )
    else:
        model_name = st.text_input(
            "모델 이름",
            value="gemini-2.5-pro",
            placeholder="예: gemini-2.5-pro, gemini-2.5-flash",
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
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    st.caption("LangGraph와 Streamlit을 사용한 채팅 인터페이스")

# 메인 채팅 인터페이스
st.title("💬 채팅")

# 메시지 히스토리 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력
if prompt := st.chat_input("메시지를 입력하세요..."):
    # API 키 확인
    if not st.session_state.graph:
        st.error("⚠️ 사이드바에서 API 키를 입력해주세요.")
        st.stop()
    
    # 사용자 메시지 추가 및 표시
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # AI 응답 생성
    with st.chat_message("assistant"):
        with st.spinner("생성 중..."):
            try:
                # LangChain 메시지 형식으로 변환
                langchain_messages = []
                for msg in st.session_state.messages:
                    if msg["role"] == "user":
                        langchain_messages.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        langchain_messages.append(AIMessage(content=msg["content"]))
                
                # LangGraph 실행
                result = st.session_state.graph.invoke({"messages": langchain_messages})
                
                # 마지막 AI 응답 추출
                if result.get("messages"):
                    last_message = result["messages"][-1]
                    response = last_message.content if hasattr(last_message, 'content') else str(last_message)
                else:
                    response = "응답을 생성할 수 없습니다."
                
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                error_msg = f"❌ 오류 발생: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

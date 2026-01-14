import os
import re
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


def extract_text_from_content(content) -> str:
    """
    메시지 content에서 텍스트를 추출합니다.
    content가 문자열, 리스트, 또는 다른 형태일 수 있음을 처리합니다.
    """
    if content is None:
        return ""
    
    # 이미 문자열인 경우
    if isinstance(content, str):
        return content
    
    # 리스트인 경우 (Gemini 응답 형식)
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict):
                # {'type': 'text', 'text': '...'} 형식 처리
                if item.get('type') == 'text' and 'text' in item:
                    text_parts.append(item['text'])
                # 다른 형태의 dict도 처리
                elif 'text' in item:
                    text_parts.append(item['text'])
                elif 'content' in item:
                    text_parts.append(str(item['content']))
        return "\n".join(text_parts) if text_parts else str(content)
    
    # dict인 경우
    if isinstance(content, dict):
        if 'text' in content:
            return content['text']
        elif 'content' in content:
            return str(content['content'])
        return str(content)
    
    # 그 외의 경우 문자열로 변환
    return str(content)


def extract_image_urls(text: str) -> list[str]:
    """텍스트에서 모든 이미지 URL을 추출합니다."""
    if not isinstance(text, str):
        text = extract_text_from_content(text)
    
    # URL 패턴 찾기
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    urls = re.findall(url_pattern, text)
    
    image_urls = []
    for url in urls:
        # 이미지 확장자 확인
        if any(ext in url.lower() for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp']):
            image_urls.append(url)
        # Higgsfield나 다른 이미지 서비스 URL
        elif any(service in url.lower() for service in ['higgsfield', 'cloudinary', 'imgur', 's3.amazonaws']):
            image_urls.append(url)
    
    return image_urls


def render_message_content(content):
    """메시지 내용을 렌더링하며, 이미지 URL이 있으면 이미지도 표시합니다."""
    # content를 문자열로 변환
    text = extract_text_from_content(content)
    
    st.markdown(text)
    
    # 이미지 URL이 있으면 이미지도 표시
    image_urls = extract_image_urls(text)
    for image_url in image_urls:
        try:
            st.image(image_url, caption="생성된 이미지", use_container_width=True)
        except Exception:
            pass  # 이미지 로드 실패 시 무시


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
                        content = extract_text_from_content(msg.content)
                        messages.append({"role": "user", "content": content})
                    elif msg_type == "AIMessage":
                        # Tool 호출 메시지는 건너뛰기
                        if hasattr(msg, 'tool_calls') and msg.tool_calls:
                            continue
                        content = extract_text_from_content(msg.content)
                        if content:  # 빈 content는 건너뛰기
                            messages.append({"role": "assistant", "content": content})
                    elif msg_type == "ToolMessage":
                        # Tool 실행 결과 메시지
                        content = extract_text_from_content(msg.content)
                        messages.append({"role": "assistant", "content": f"🔧 Tool 결과:\n{content}"})
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
    st.subheader("🔑 API 키 설정")
    
    gemini_api_key = st.text_input(
        "Google Gemini API 키",
        type="password",
        value=os.getenv("GOOGLE_API_KEY", ""),
        help="Google AI Studio에서 API 키를 발급받으세요: https://makersuite.google.com/app/apikey"
    )
    
    st.divider()
    st.subheader("🖼️ Higgsfield (이미지 생성)")
    
    higgsfield_api_key = st.text_input(
        "Higgsfield API Key (hf-api-key)",
        type="password",
        value=os.getenv("HIGGSFIELD_API_KEY", ""),
        help="Higgsfield Platform에서 발급받은 API Key (UUID 형식)"
    )
    
    higgsfield_api_secret = st.text_input(
        "Higgsfield API Secret (hf-secret)",
        type="password",
        value=os.getenv("HIGGSFIELD_API_SECRET", ""),
        help="Higgsfield Platform에서 발급받은 API Secret"
    )
    
    # Higgsfield 활성화 상태 표시
    higgsfield_enabled = bool(higgsfield_api_key and higgsfield_api_secret)
    if higgsfield_enabled:
        st.success("✅ 이미지 생성 기능 활성화")
    elif higgsfield_api_key or higgsfield_api_secret:
        st.warning("⚠️ API Key와 Secret 모두 입력해주세요")
    else:
        st.info("💡 Higgsfield API 키를 입력하면 이미지 생성 기능을 사용할 수 있습니다.")
    
    st.divider()
    
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
    
    # 그래프 초기화 조건 확인
    need_reinit = (
        st.session_state.graph is None or 
        st.session_state.get("current_gemini_key") != gemini_api_key or 
        st.session_state.get("current_higgsfield_key") != higgsfield_api_key or
        st.session_state.get("current_higgsfield_secret") != higgsfield_api_secret or
        st.session_state.get("current_model") != model_name
    )
    
    if gemini_api_key:
        if need_reinit:
            try:
                with st.spinner("그래프 초기화 중..."):
                    st.session_state.graph = create_chat_graph(
                        gemini_api_key=gemini_api_key,
                        model_name=model_name,
                        higgsfield_api_key=higgsfield_api_key if higgsfield_enabled else None,
                        higgsfield_api_secret=higgsfield_api_secret if higgsfield_enabled else None
                    )
                    st.session_state.current_gemini_key = gemini_api_key
                    st.session_state.current_higgsfield_key = higgsfield_api_key
                    st.session_state.current_higgsfield_secret = higgsfield_api_secret
                    st.session_state.current_model = model_name
                
                st.success("✅ 그래프가 초기화되었습니다!")
            except Exception as e:
                st.error(f"❌ 그래프 초기화 실패: {str(e)}")
                st.session_state.graph = None
    else:
        st.warning("⚠️ Gemini API 키를 입력해주세요")
        st.session_state.graph = None
    
    st.divider()
    
    if st.button("🗑️ 대화 기록 지우기", use_container_width=True):
        # 새로운 thread_id 생성으로 대화 초기화
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()
    
    st.divider()
    
    # 사용 가능한 기능 표시
    st.subheader("🛠️ 사용 가능한 기능")
    st.markdown("- 💬 **일반 대화**: Gemini와 자유롭게 대화")
    if higgsfield_enabled:
        st.markdown("- 🖼️ **이미지 생성**: 텍스트로 이미지 생성")
        st.markdown("- 📊 **상태 확인**: 생성 상태 조회")
        st.markdown("- ❌ **생성 취소**: 진행 중인 생성 취소")
        with st.expander("💡 사용 예시"):
            st.markdown("""
            **이미지 생성:**
            - "고양이가 우주복을 입고 있는 이미지 생성해줘"
            - "16:9 비율로 산과 호수 풍경 이미지 만들어줘"
            - "png 포맷으로 미래 도시 이미지 2개 생성해줘"
            
            **상태 확인:**
            - "request_id가 xxx인 요청 상태 확인해줘"
            
            **생성 취소:**
            - "request_id가 xxx인 생성 요청 취소해줘"
            """)
    
    st.divider()
    st.caption("LangGraph와 Streamlit을 사용한 채팅 인터페이스")

# 메인 채팅 인터페이스
st.title("💬 채팅")

# 메시지 히스토리 표시 (LangGraph state에서 가져오기)
messages = get_messages()
for message in messages:
    with st.chat_message(message["role"]):
        render_message_content(message["content"])

# 사용자 입력
if prompt := st.chat_input("메시지를 입력하세요..."):
    # API 키 확인
    if not st.session_state.graph:
        st.error("⚠️ 사이드바에서 Gemini API 키를 입력해주세요.")
        st.stop()
    
    # 사용자 메시지 표시
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # AI 응답 생성
    with st.chat_message("assistant"):
        with st.spinner("생성 중... (이미지 생성 시 최대 2분 소요될 수 있습니다)"):
            try:
                # LangGraph 실행 (checkpointer가 자동으로 히스토리 관리)
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                result = st.session_state.graph.invoke(
                    {"messages": [HumanMessage(content=prompt)]},
                    config=config
                )
                
                # 마지막 AI 응답 추출 및 표시
                response = "응답을 생성할 수 없습니다."
                
                if result.get("messages"):
                    # 마지막 실제 응답 찾기 (tool call이 아닌 메시지)
                    for msg in reversed(result["messages"]):
                        if hasattr(msg, '__class__'):
                            msg_type = msg.__class__.__name__
                            if msg_type == "AIMessage":
                                # Tool 호출만 있는 메시지는 건너뛰기
                                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                    # content가 있는 경우에만 처리
                                    content = extract_text_from_content(msg.content)
                                    if not content:
                                        continue
                                
                                response = extract_text_from_content(msg.content)
                                if response:
                                    break
                            elif msg_type == "ToolMessage":
                                content = extract_text_from_content(msg.content)
                                response = f"🔧 Tool 결과:\n{content}"
                                break
                
                render_message_content(response)
                
                # 성공 시에만 페이지 리로드
                st.rerun()
                
            except Exception as e:
                error_msg = f"❌ 오류 발생: {str(e)}"
                st.error(error_msg)
                # 에러 발생 시에는 rerun하지 않아서 에러 메시지가 화면에 남도록 함

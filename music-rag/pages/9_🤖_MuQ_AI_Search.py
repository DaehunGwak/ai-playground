import os
import numpy as np
import streamlit as st
import torch
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv(".env.local", override=True)

# 커스텀 CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Sora:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Sora', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1a0a2e 0%, #4a1942 50%, #2d1b4e 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        border: 1px solid #6b2d5b;
    }
    
    .main-header h1 {
        color: #ff6b9d;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 0 30px rgba(255, 107, 157, 0.5);
        font-family: 'Space Mono', monospace;
    }
    
    .main-header p {
        color: #d4a5c9;
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    .query-box {
        background: linear-gradient(145deg, #1e1e2e, #2d2d44);
        border: 1px solid #4a3d5c;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    
    .result-card {
        background: linear-gradient(145deg, #1e1e2e, #2d2d44);
        border: 1px solid #4a3d5c;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .result-card:hover {
        border-color: #ff6b9d;
        box-shadow: 0 4px 15px rgba(255, 107, 157, 0.2);
        transform: translateY(-2px);
    }
    
    .result-title {
        color: #ff6b9d;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        font-family: 'Space Mono', monospace;
    }
    
    .result-tags {
        color: #d4a5c9;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    
    .similarity-badge {
        background: linear-gradient(135deg, #ff6b9d, #d4a5c9);
        color: #1a0a2e;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    .converted-query {
        background: rgba(255, 107, 157, 0.1);
        border-left: 4px solid #ff6b9d;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    
    .converted-query code {
        color: #ff6b9d;
        background: rgba(0, 0, 0, 0.3);
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
    }
    
    .language-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 500;
        margin-right: 0.5rem;
    }
    
    .badge-en {
        background: rgba(78, 205, 196, 0.2);
        color: #4ecdc4;
    }
    
    .badge-zh {
        background: rgba(255, 230, 109, 0.2);
        color: #ffe66d;
    }
    
    .badge-ko {
        background: rgba(255, 107, 157, 0.2);
        color: #ff6b9d;
    }
</style>
""", unsafe_allow_html=True)

# 헤더
st.markdown("""
<div class="main-header">
    <h1>🤖 MuQ AI Music Search</h1>
    <p>한국어/영어/중국어로 원하는 음악을 설명하면 AI가 검색해드립니다</p>
    <div style="margin-top: 0.5rem;">
        <span class="language-badge badge-ko">한국어</span>
        <span class="language-badge badge-en">English</span>
        <span class="language-badge badge-zh">中文</span>
    </div>
</div>
""", unsafe_allow_html=True)


# LLM 체인 로드 (캐싱)
@st.cache_resource
def load_llm_chain():
    """LangChain + Claude 체인을 로드합니다."""
    from langchain_anthropic import ChatAnthropic
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    
    SYSTEM_PROMPT = """You are a music query translator. Your task is to convert user's music search query (in any language) into a structured English description format that can be used for music embedding models like MuQ-MuLan.

Output Format (strictly follow this template):
[genre], [mood] mood, [tempo or energy], [main instruments]

Guidelines:
1. Genre: Identify the most likely music genre
2. Mood: Describe the emotional atmosphere
3. Tempo/Energy: Describe the speed or energy level
4. Main instruments: List likely instruments. If uncertain, use "soft instruments" or "electronic sounds"

Important:
- Output ONLY the formatted query string, nothing else.
- If some information is not clear from the query, make reasonable assumptions based on the context.
- Keep the output concise and focused on musical characteristics.
- All output must be in English."""

    llm = ChatAnthropic(
        model="claude-sonnet-4-5-20250929",
        temperature=0.3,
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Convert this music search query to the template format: {query}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    return chain


# MuQ-MuLan 모델 로드 (캐싱)
@st.cache_resource
def load_muq_model():
    """MuQ-MuLan 모델을 로드합니다."""
    from muq import MuQMuLan
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large")
    model = model.to(device).eval()
    return model, device


# Supabase 클라이언트 (캐싱)
@st.cache_resource
def get_supabase_client():
    """Supabase 클라이언트를 생성합니다."""
    from supabase import create_client
    
    return create_client(
        supabase_url=os.getenv("SUPABASE_URL"),
        supabase_key=os.getenv("SUPABASE_SECRET_KEY"),
    )


def extract_music_query(user_query: str, chain) -> str:
    """한국어 쿼리를 영어 템플릿 형식으로 변환합니다."""
    result = chain.invoke({"query": user_query})
    return result.strip()


def get_text_embedding(query: str, model, device) -> list:
    """텍스트 쿼리를 MuQ-MuLan 임베딩으로 변환합니다."""
    with torch.no_grad():
        text_emb = model(texts=[query])
    
    emb_np = text_emb.cpu().numpy().squeeze()
    
    # L2 정규화
    l2_norm = np.linalg.norm(emb_np)
    if l2_norm > 0:
        emb_np = emb_np / l2_norm
    
    return emb_np.tolist()


def search_music(query_embedding: list, match_count: int, supabase_client) -> list:
    """Supabase에서 MuQ 벡터 검색을 수행합니다."""
    response = supabase_client.rpc("search_tracks_muq", {
        "query_embedding": query_embedding,
        "match_count": match_count
    }).execute()
    
    return response.data if hasattr(response, 'data') else response


# API 키 체크
api_key_valid = bool(os.getenv("ANTHROPIC_API_KEY"))
supabase_valid = bool(os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_SECRET_KEY"))

if not api_key_valid:
    st.error("⚠️ ANTHROPIC_API_KEY가 설정되지 않았습니다. .env.local 파일을 확인해주세요.")
if not supabase_valid:
    st.error("⚠️ Supabase 환경변수가 설정되지 않았습니다. .env.local 파일을 확인해주세요.")

# 사이드바
with st.sidebar:
    st.header("⚙️ Settings")
    
    match_count = st.slider(
        "검색 결과 수",
        min_value=1,
        max_value=20,
        value=5,
        help="반환할 검색 결과의 개수"
    )
    
    show_converted_query = st.checkbox(
        "변환된 쿼리 표시",
        value=True,
        help="LLM이 변환한 영어 쿼리를 표시합니다"
    )
    
    st.markdown("---")
    
    # 언어 지원 안내
    st.markdown("### 🌐 Language Support")
    st.markdown("""
    <span class="language-badge badge-ko">한국어</span>
    <span class="language-badge badge-en">English</span>
    <span class="language-badge badge-zh">中文</span>
    """, unsafe_allow_html=True)
    st.caption("MuQ-MuLan은 다국어 쿼리를 지원합니다")
    
    st.markdown("---")
    st.markdown("### 📝 예시 쿼리")
    
    example_queries = [
        "🇰🇷 료칸 여행에 어울리는 숏폼 음악",
        "🇰🇷 카페에서 공부할 때 듣기 좋은 음악",
        "🇰🇷 새벽 드라이브에 어울리는 시티팝",
        "🇺🇸 Chill lo-fi beats for studying",
        "🇨🇳 轻松的钢琴音乐适合学习",
    ]
    
    for eq in example_queries:
        if st.button(eq, key=eq, use_container_width=True):
            st.session_state['muq_ai_selected_query'] = eq

# 메인 검색 영역
st.markdown("### 🔍 음악 검색")

# 선택된 예시 쿼리가 있으면 사용
default_query = st.session_state.get('muq_ai_selected_query', '')
# 이모지 플래그 제거
if default_query.startswith(('🇰🇷 ', '🇺🇸 ', '🇨🇳 ')):
    default_query = default_query[3:]

user_query = st.text_input(
    "어떤 음악을 찾고 계신가요?",
    value=default_query,
    placeholder="예: 아침에 일어나서 듣는 상쾌한 팝송 / Relaxing music for meditation / 浪漫的爵士乐",
    help="한국어, 영어, 중국어로 자유롭게 설명해주세요"
)

# 검색 버튼
search_button = st.button(
    "🚀 검색하기",
    type="primary",
    use_container_width=True,
    disabled=not (api_key_valid and supabase_valid and user_query)
)

# 검색 실행
if search_button and user_query:
    try:
        # 1단계: LLM으로 쿼리 변환
        with st.spinner("🤖 AI가 쿼리를 분석하고 있습니다..."):
            chain = load_llm_chain()
            english_query = extract_music_query(user_query, chain)
        
        if show_converted_query:
            st.markdown(f"""
            <div class="converted-query">
                <strong>🔄 변환된 쿼리:</strong><br/>
                <code>{english_query}</code>
            </div>
            """, unsafe_allow_html=True)
        
        # 2단계: MuQ-MuLan 임베딩 생성
        with st.spinner("🎹 MuQ-MuLan 임베딩을 생성하고 있습니다..."):
            model, device = load_muq_model()
            query_embedding = get_text_embedding(english_query, model, device)
        
        # 3단계: Supabase 검색
        with st.spinner("🔍 데이터베이스에서 검색 중..."):
            supabase_client = get_supabase_client()
            results = search_music(query_embedding, match_count, supabase_client)
        
        # 결과 표시
        if results:
            st.success(f"✅ {len(results)}개의 음악을 찾았습니다!")
            st.markdown("---")
            st.markdown("### 🎶 검색 결과")
            
            for i, item in enumerate(results):
                title = item.get('title', 'Unknown')
                tags = item.get('tags', '')
                similarity = item.get('similarity', 0)
                audio_url = item.get('audioUrl', '')
                
                # 거리를 유사도 점수로 변환 (거리가 작을수록 유사도가 높음)
                similarity_score = similarity
                
                # 결과 카드
                with st.container():
                    col1, col2 = st.columns([4, 1])
                    
                    with col1:
                        st.markdown(f"#### {i+1}. {title}")
                        
                        # 태그 (길면 자르기)
                        if tags:
                            display_tags = tags[:200] + "..." if len(tags) > 200 else tags
                            st.caption(f"🏷️ {display_tags}")
                    
                    with col2:
                        st.metric("유사도(거리)", f"{similarity_score:.6f}")
                    
                    # 오디오 플레이어
                    if audio_url:
                        st.audio(audio_url)
                    
                    st.markdown("---")
        else:
            st.warning("🔍 검색 결과가 없습니다. 다른 쿼리로 시도해보세요.")
            
    except Exception as e:
        st.error(f"❌ 오류가 발생했습니다: {str(e)}")

# 안내 메시지
if not user_query and not search_button:
    st.markdown("---")
    st.info("👆 검색창에 원하는 음악을 자유롭게 설명해주세요. 사이드바의 예시를 클릭해도 됩니다!")
    
    # 사용 예시
    st.markdown("### 💡 이런 식으로 검색해보세요")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🇰🇷 한국어**")
        st.markdown("""
        - "아침에 일어나서 듣기 좋은 음악"
        - "카페에서 공부할 때 집중되는 음악"
        - "드라이브하면서 듣는 신나는 음악"
        """)
    
    with col2:
        st.markdown("**🇺🇸 English**")
        st.markdown("""
        - "Relaxing lo-fi beats for studying"
        - "Energetic workout music"
        - "Romantic jazz for dinner"
        """)
    
    with col3:
        st.markdown("**🇨🇳 中文**")
        st.markdown("""
        - "轻松的钢琴音乐，适合学习"
        - "充满活力的舞曲，快节奏"
        - "浪漫的爵士乐，萨克斯和钢琴"
        """)
    
    # MuQ-MuLan 정보
    st.markdown("---")
    st.markdown("### ℹ️ About MuQ-MuLan")
    st.markdown("""
    <div class="query-box">
        <h4 style="color: #ff6b9d;">🎹 MuQ-MuLan Large</h4>
        <p style="color: #d4a5c9;">
            MuQ-MuLan은 음악과 텍스트를 공유 임베딩 공간에 매핑하는 다국어 음악-언어 모델입니다.
            영어와 중국어 텍스트 쿼리를 네이티브로 지원하며, 한국어 쿼리는 AI가 영어로 변환하여 처리합니다.
        </p>
        <br>
        <h4 style="color: #ff6b9d;">🤖 AI Query Translation</h4>
        <p style="color: #d4a5c9;">
            사용자의 자연어 쿼리를 Claude AI가 분석하여 음악 검색에 최적화된 구조화된 형식으로 변환합니다.
            이를 통해 더 정확하고 의미 있는 음악 검색이 가능합니다.
        </p>
    </div>
    """, unsafe_allow_html=True)

# 푸터
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>Powered by 🤖 Claude + 🎹 MuQ-MuLan + ⚡ Supabase</div>",
    unsafe_allow_html=True
)


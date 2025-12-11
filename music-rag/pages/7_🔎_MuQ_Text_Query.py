import os
import numpy as np
import dotenv
import streamlit as st
import plotly.graph_objects as go
import torch

# 환경변수 로드
dotenv.load_dotenv(".env.local", override=True)

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
    
    .template-info {
        background: linear-gradient(145deg, #1e1e2e, #2d2d44);
        border: 1px solid #4a3d5c;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    
    .template-info code {
        background: rgba(255, 107, 157, 0.2);
        color: #ff6b9d;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
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
    }
    
    .result-title {
        color: #ff6b9d;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        font-family: 'Space Mono', monospace;
    }
    
    .result-similarity {
        color: #4ecdc4;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    .result-tags {
        color: #d4a5c9;
        font-size: 0.85rem;
        margin-top: 0.5rem;
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
</style>
""", unsafe_allow_html=True)

# 헤더
st.markdown("""
<div class="main-header">
    <h1>🔎 MuQ-MuLan Text Query</h1>
    <p>Generate embeddings from natural language music descriptions (English & Chinese supported)</p>
</div>
""", unsafe_allow_html=True)


# Supabase 클라이언트
@st.cache_resource
def get_supabase_client():
    from supabase import create_client
    return create_client(
        supabase_url=os.getenv("SUPABASE_URL"),
        supabase_key=os.getenv("SUPABASE_SECRET_KEY"),
    )


# MuQ-MuLan 모델 로드 (캐싱)
@st.cache_resource
def load_muq_model():
    """MuQ-MuLan 모델을 로드합니다."""
    from muq import MuQMuLan
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large")
    model = model.to(device).eval()
    return model, device


def get_text_embedding(query: str, model, device):
    """텍스트 쿼리를 MuQ-MuLan 임베딩으로 변환"""
    with torch.no_grad():
        text_emb = model(texts=[query])
    
    emb_np = text_emb.cpu().numpy().squeeze()
    
    # L2 정규화
    l2_norm = np.linalg.norm(emb_np)
    if l2_norm > 0:
        emb_np = emb_np / l2_norm
    
    return emb_np


def build_music_query(genre, mood, tempo, instruments, vocal_info, use_case):
    """템플릿에 맞춰 음악 검색 쿼리를 생성"""
    parts = []
    if genre:
        parts.append(genre)
    if mood:
        parts.append(f"{mood} mood")
    if tempo:
        parts.append(tempo)
    if instruments:
        parts.append(instruments)
    if vocal_info:
        parts.append(vocal_info)
    if use_case:
        parts.append(f"suitable for {use_case}")
    
    return ", ".join(parts)


def plot_embedding(embedding: np.ndarray):
    """임베딩 벡터를 시각화합니다."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(range(len(embedding))),
        y=embedding,
        mode='lines',
        name='Query Embedding',
        line=dict(color='#ff6b9d', width=1),
        fill='tozeroy',
        fillcolor='rgba(255, 107, 157, 0.2)'
    ))
    
    fig.update_layout(
        title="Query Embedding Vector",
        xaxis_title="Dimension",
        yaxis_title="Value",
        plot_bgcolor='rgba(30,30,46,0.8)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#d4a5c9'),
        title_font=dict(color='#ff6b9d', size=16),
        xaxis=dict(gridcolor='#3d3d5c'),
        yaxis=dict(gridcolor='#3d3d5c'),
        height=300
    )
    return fig


def search_tracks_muq(embedding: list, match_count: int = 5):
    """MuQ-MuLan 임베딩으로 유사한 트랙을 검색합니다."""
    client = get_supabase_client()
    try:
        response = client.rpc("search_tracks_muq", {
            "query_embedding": embedding,
            "match_count": match_count
        }).execute()
        
        return response.data if hasattr(response, 'data') else response
    except Exception as e:
        st.error(f"검색 오류: {e}")
        return []


# 사이드바 - 입력 모드 선택
with st.sidebar:
    st.header("⚙️ Settings")
    
    input_mode = st.radio(
        "Input Mode",
        ["📝 Direct Text", "🧩 Template Builder"],
        help="직접 텍스트 입력 또는 템플릿 빌더 사용"
    )
    
    st.markdown("---")
    
    # 언어 선택
    st.markdown("### 🌐 Language Support")
    st.markdown("""
    <span class="language-badge badge-en">English</span>
    <span class="language-badge badge-zh">中文</span>
    """, unsafe_allow_html=True)
    st.caption("MuQ-MuLan supports both English and Chinese queries")
    
    st.markdown("---")
    
    # 검색 설정
    st.markdown("### 🔍 Search Settings")
    match_count = st.slider("Number of results", min_value=3, max_value=10, value=5)

# 메인 컨텐츠
st.markdown("### 🎵 Query Template")
st.markdown("""
<div class="template-info">
    <code>{genre}</code>, <code>{mood}</code> mood, <code>{tempo or energy}</code>, 
    <code>{main instruments}</code>, <code>{vocal info + language}</code>, 
    suitable for <code>{use-case or scene}</code>
</div>
""", unsafe_allow_html=True)

# 입력 영역
query_text = ""

if input_mode == "📝 Direct Text":
    st.markdown("### ✍️ Enter Your Query")
    query_text = st.text_area(
        "Music description (English or Chinese)",
        placeholder="예: Electronic, energetic mood, fast tempo, synthesizer, no vocals, suitable for workout\n或者: 轻松的钢琴音乐，适合学习",
        height=100,
        help="음악을 설명하는 텍스트를 입력하세요 (영어 또는 중국어)"
    )
    
    # 예시 쿼리 버튼들
    st.markdown("**Quick Examples:**")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🎧 Chill Lo-fi", use_container_width=True):
            st.session_state['muq_example_query'] = "Lo-fi hip hop, relaxed mood, slow tempo, piano and vinyl crackle, no vocals, suitable for studying"
    with col2:
        if st.button("🎸 Energetic Rock", use_container_width=True):
            st.session_state['muq_example_query'] = "Rock, intense mood, fast energy, electric guitar and heavy drums, male vocals, suitable for workout"
    with col3:
        if st.button("🎹 Calm Piano", use_container_width=True):
            st.session_state['muq_example_query'] = "Classical, peaceful mood, slow tempo, piano solo, no vocals, suitable for meditation"
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🇨🇳 轻松钢琴", use_container_width=True):
            st.session_state['muq_example_query'] = "轻松的钢琴音乐，适合学习"
    with col2:
        if st.button("🇨🇳 活力舞曲", use_container_width=True):
            st.session_state['muq_example_query'] = "充满活力的舞曲，快节奏，适合运动"
    with col3:
        if st.button("🇨🇳 浪漫爵士", use_container_width=True):
            st.session_state['muq_example_query'] = "浪漫的爵士乐，萨克斯和钢琴，适合约会"
    
    # 예시 쿼리가 선택된 경우
    if 'muq_example_query' in st.session_state:
        query_text = st.session_state['muq_example_query']
        st.info(f"Selected: {query_text}")

else:  # Template Builder
    st.markdown("### 🧩 Build Your Query")
    
    col1, col2 = st.columns(2)
    
    with col1:
        genre = st.selectbox(
            "Genre",
            ["", "Electronic", "Lo-fi hip hop", "Pop", "Rock", "Jazz", "Classical", 
             "Ambient", "R&B", "Hip hop", "House", "Techno", "Indie", "Folk", "Metal"],
            help="음악 장르"
        )
        
        mood = st.selectbox(
            "Mood",
            ["", "energetic", "relaxed", "happy", "sad", "romantic", "intense", 
             "peaceful", "melancholic", "uplifting", "dark", "dreamy", "aggressive"],
            help="음악의 분위기"
        )
        
        tempo = st.selectbox(
            "Tempo / Energy",
            ["", "fast tempo", "medium tempo", "slow tempo", "high energy", 
             "low energy", "moderate pace", "upbeat", "downtempo"],
            help="템포 또는 에너지 레벨"
        )
    
    with col2:
        instruments = st.text_input(
            "Main Instruments",
            placeholder="예: piano and strings, synthesizer, electric guitar",
            help="주요 악기들"
        )
        
        vocal_info = st.selectbox(
            "Vocal Info",
            ["", "no vocals", "female vocals in English", "male vocals in English",
             "female vocals in Korean", "male vocals in Korean", "choir",
             "instrumental only", "vocal harmonies"],
            help="보컬 정보"
        )
        
        use_case = st.text_input(
            "Use Case / Scene",
            placeholder="예: studying, workout, relaxation, party",
            help="사용 용도나 장면"
        )
    
    query_text = build_music_query(genre, mood, tempo, instruments, vocal_info, use_case)
    
    if query_text:
        st.markdown("**Generated Query:**")
        st.info(query_text)

# 검색 버튼
st.markdown("---")

search_button = st.button(
    "🚀 Generate Embedding & Search",
    type="primary",
    use_container_width=True,
    disabled=not query_text
)

# 결과 처리
if search_button:
    if not query_text:
        st.warning("⚠️ Please enter a query first")
    else:
        # 모델 로드
        with st.spinner("🔄 Loading MuQ-MuLan model..."):
            try:
                model, device = load_muq_model()
                model_loaded = True
                st.sidebar.success(f"✅ Model on {device}")
            except Exception as e:
                st.error(f"❌ Failed to load model: {e}")
                model_loaded = False
        
        if model_loaded:
            # 임베딩 생성
            with st.spinner("🔄 Generating text embedding..."):
                embedding = get_text_embedding(query_text, model, device)
            
            st.success("✅ Embedding generated successfully!")
            
            # 두 개의 탭으로 구성
            tab1, tab2 = st.tabs(["📊 Embedding Analysis", "🎵 Search Results"])
            
            with tab1:
                # 임베딩 통계
                st.markdown("### 📊 Embedding Statistics")
                
                l2_norm = np.linalg.norm(embedding)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Dimensions", embedding.shape[0])
                with col2:
                    st.metric("L2 Norm", f"{l2_norm:.6f}")
                with col3:
                    st.metric("Min Value", f"{embedding.min():.4f}")
                with col4:
                    st.metric("Max Value", f"{embedding.max():.4f}")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean", f"{embedding.mean():.6f}")
                with col2:
                    st.metric("Std Dev", f"{embedding.std():.6f}")
                with col3:
                    st.metric("Median", f"{np.median(embedding):.6f}")
                with col4:
                    st.metric("Non-zero", f"{np.count_nonzero(embedding)}")
                
                # 시각화
                st.markdown("### 📈 Embedding Visualization")
                st.plotly_chart(plot_embedding(embedding), use_container_width=True)
                
                # Raw 임베딩 데이터
                with st.expander("🔢 View Raw Embedding Data"):
                    st.code(f"Shape: {embedding.shape}\n\nValues:\n{embedding.tolist()}")
                
                # 다운로드 옵션
                st.markdown("### 💾 Export")
                col1, col2 = st.columns(2)
                with col1:
                    import io
                    buffer = io.BytesIO()
                    np.save(buffer, embedding)
                    buffer.seek(0)
                    st.download_button(
                        label="📥 Download as .npy",
                        data=buffer,
                        file_name="muq_query_embedding.npy",
                        mime="application/octet-stream"
                    )
                with col2:
                    import json
                    json_data = json.dumps({
                        "query": query_text,
                        "model": "OpenMuQ/MuQ-MuLan-large",
                        "embedding": embedding.tolist(),
                        "l2_norm": float(l2_norm),
                        "dimensions": int(embedding.shape[0])
                    }, indent=2)
                    st.download_button(
                        label="📥 Download as .json",
                        data=json_data,
                        file_name="muq_query_embedding.json",
                        mime="application/json"
                    )
            
            with tab2:
                # 검색 수행
                with st.spinner("🔍 Searching similar tracks..."):
                    results = search_tracks_muq(embedding.tolist(), match_count)
                
                if results:
                    st.markdown(f"### 🎵 Found {len(results)} Similar Tracks")
                    st.markdown(f"**Query:** {query_text}")
                    st.markdown("---")
                    
                    for i, track in enumerate(results):
                        with st.container():
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.markdown(f"""
                                <div class="result-card">
                                    <div class="result-title">#{i+1} {track.get('title', 'Untitled')}</div>
                                    <div class="result-similarity">📏 Distance: {track.get('similarity', 0):.6f}</div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # 태그 표시
                                if track.get('tags'):
                                    tags = track['tags'][:150] + "..." if len(track.get('tags', '')) > 150 else track.get('tags', '')
                                    st.caption(f"🏷️ {tags}")
                            
                            with col2:
                                similarity = track.get('similarity', 0)
                                # 거리가 작을수록 유사도가 높음 (코사인 거리 기준)
                                # 대략적인 유사도 점수로 변환 (1 - distance를 percentage로)
                                similarity_score = max(0, (1 - similarity) * 100)
                                st.metric("Similarity", f"{similarity_score:.1f}%")
                            
                            # 오디오 플레이어
                            if track.get('audioUrl'):
                                st.audio(track['audioUrl'])
                            
                            st.markdown("---")
                else:
                    st.info("🔍 No similar tracks found. Try a different query.")

# 안내 메시지 (쿼리가 없을 때)
if not query_text and not search_button:
    st.markdown("---")
    st.info("👆 Enter a text description or use the template builder to search for similar music")
    
    # MuQ-MuLan 정보
    st.markdown("### ℹ️ About MuQ-MuLan")
    st.markdown("""
    <div class="template-info">
        <h4 style="color: #ff6b9d;">Multilingual Support</h4>
        <p style="color: #d4a5c9;">
            MuQ-MuLan supports both <strong>English</strong> and <strong>Chinese</strong> text queries, 
            making it ideal for cross-lingual music search.
        </p>
        <br>
        <h4 style="color: #ff6b9d;">Example Queries</h4>
        <ul style="color: #d4a5c9;">
            <li><strong>English:</strong> "relaxing piano music for studying"</li>
            <li><strong>中文:</strong> "轻松的钢琴音乐，适合学习"</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# 푸터
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>Powered by 🎹 MuQ-MuLan & OpenMuQ</div>",
    unsafe_allow_html=True
)


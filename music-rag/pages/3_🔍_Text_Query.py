import numpy as np
import streamlit as st
import plotly.graph_objects as go

# 커스텀 CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #4a1942 50%, #16213e 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .main-header h1 {
        color: #ff6b9d;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 0 30px rgba(255, 107, 157, 0.5);
    }
    
    .main-header p {
        color: #b8c5d6;
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    .template-info {
        background: linear-gradient(145deg, #1e1e2e, #2d2d44);
        border: 1px solid #3d3d5c;
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
</style>
""", unsafe_allow_html=True)

# 헤더
st.markdown("""
<div class="main-header">
    <h1>🔍 Text Query Embedding</h1>
    <p>Generate CLAP embeddings from natural language music descriptions</p>
</div>
""", unsafe_allow_html=True)


# CLAP 모델 로드 (캐싱)
@st.cache_resource
def load_clap_model():
    """CLAP 모델과 프로세서를 로드합니다."""
    from transformers import ClapProcessor, ClapModel
    
    processor = ClapProcessor.from_pretrained("laion/larger_clap_music")
    model = ClapModel.from_pretrained("laion/larger_clap_music")
    return processor, model


def get_text_embedding(query: str, processor, model):
    """텍스트 쿼리를 CLAP 임베딩으로 변환"""
    import torch
    
    inputs = processor(text=query, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        text_emb = model.get_text_features(**inputs)
    
    return text_emb.cpu().numpy().squeeze()


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
        title="Embedding Vector (512 dimensions)",
        xaxis_title="Dimension",
        yaxis_title="Value",
        plot_bgcolor='rgba(30,30,46,0.8)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#b8c5d6'),
        title_font=dict(color='#ff6b9d', size=16),
        xaxis=dict(gridcolor='#3d3d5c'),
        yaxis=dict(gridcolor='#3d3d5c'),
    )
    return fig


# 사이드바 - 입력 모드 선택
with st.sidebar:
    st.header("⚙️ Settings")
    
    input_mode = st.radio(
        "Input Mode",
        ["📝 Direct Text", "🧩 Template Builder"],
        help="직접 텍스트 입력 또는 템플릿 빌더 사용"
    )

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
        "Music description",
        placeholder="예: Electronic, energetic mood, fast tempo, synthesizer and drum machine, no vocals, suitable for workout",
        height=100,
        help="음악을 설명하는 텍스트를 입력하세요"
    )
    
    # 예시 쿼리 버튼들
    st.markdown("**Quick Examples:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🎧 Chill Lo-fi", use_container_width=True):
            st.session_state['example_query'] = "Lo-fi hip hop, relaxed mood, slow tempo, piano and vinyl crackle, no vocals, suitable for studying"
    with col2:
        if st.button("🎸 Energetic Rock", use_container_width=True):
            st.session_state['example_query'] = "Rock, intense mood, fast energy, electric guitar and heavy drums, male vocals in English, suitable for workout"
    with col3:
        if st.button("🎹 Calm Piano", use_container_width=True):
            st.session_state['example_query'] = "Classical, peaceful mood, slow tempo, piano solo, no vocals, suitable for meditation"
    
    # 예시 쿼리가 선택된 경우
    if 'example_query' in st.session_state:
        query_text = st.session_state['example_query']
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

# 임베딩 생성 버튼
st.markdown("---")

generate_button = st.button(
    "🚀 Generate Embedding",
    type="primary",
    use_container_width=True,
    disabled=not query_text
)

# 결과 처리
if generate_button:
    if not query_text:
        st.warning("⚠️ Please enter a query first")
    else:
        # 모델 로드
        with st.spinner("🔄 Loading CLAP model..."):
            try:
                processor, model = load_clap_model()
                model_loaded = True
            except Exception as e:
                st.error(f"❌ Failed to load model: {e}")
                model_loaded = False
        
        if model_loaded:
            # 임베딩 생성
            with st.spinner("🔄 Generating text embedding..."):
                embedding = get_text_embedding(query_text, processor, model)
            
            st.success("✅ Embedding generated successfully!")
            
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
            st.markdown("---")
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
                    file_name="query_embedding.npy",
                    mime="application/octet-stream"
                )
            with col2:
                import json
                json_data = json.dumps({
                    "query": query_text,
                    "embedding": embedding.tolist(),
                    "l2_norm": float(l2_norm),
                    "dimensions": int(embedding.shape[0])
                }, indent=2)
                st.download_button(
                    label="📥 Download as .json",
                    data=json_data,
                    file_name="query_embedding.json",
                    mime="application/json"
                )

# 안내 메시지 (쿼리가 없을 때)
if not query_text and not generate_button:
    st.markdown("---")
    st.info("👆 Enter a text description or use the template builder to generate an embedding")

# 푸터
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>Powered by 🤗 Transformers & CLAP</div>",
    unsafe_allow_html=True
)

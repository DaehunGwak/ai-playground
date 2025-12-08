import os
import numpy as np
import dotenv
import streamlit as st
import plotly.graph_objects as go

# 환경변수 로드
dotenv.load_dotenv(".env.local", override=True)

# 페이지 설정
st.set_page_config(
    page_title="🎯 Audio Embedding",
    page_icon="🎯",
    layout="wide",
)

# 커스텀 CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .main-header h1 {
        color: #00d9ff;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 0 30px rgba(0, 217, 255, 0.5);
    }
    
    .main-header p {
        color: #b8c5d6;
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    .info-card {
        background: linear-gradient(145deg, #1e1e2e, #2d2d44);
        border: 1px solid #3d3d5c;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .metric-value {
        color: #00d9ff;
        font-size: 1.8rem;
        font-weight: 700;
    }
    
    .metric-label {
        color: #b8c5d6;
        font-size: 0.9rem;
    }
    
    .embedding-stats {
        background: linear-gradient(145deg, #1a1a2e, #252545);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .stat-row {
        display: flex;
        justify-content: space-between;
        padding: 0.5rem 0;
        border-bottom: 1px solid #3d3d5c;
    }
    
    .stat-row:last-child {
        border-bottom: none;
    }
</style>
""", unsafe_allow_html=True)

# 헤더
st.markdown("""
<div class="main-header">
    <h1>🎯 Audio Embedding Extractor</h1>
    <p>Extract CLAP embeddings from audio files using laion/larger_clap_music</p>
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


def extract_embedding(audio_path: str, processor, model):
    """오디오 파일에서 CLAP 임베딩을 추출합니다."""
    import librosa
    import torch
    
    TARGET_SR = 48000
    
    # 오디오 로드 (48kHz, mono)
    audio_data, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
    
    # processor로 전처리
    inputs = processor(
        audio=audio_data,
        sampling_rate=TARGET_SR,
        return_tensors="pt"
    )
    
    # 임베딩 추출
    with torch.no_grad():
        audio_emb = model.get_audio_features(**inputs)
    
    return audio_emb.cpu().numpy(), audio_data, sr


def plot_embedding_line(embedding: np.ndarray):
    """임베딩 벡터를 선 그래프로 시각화합니다."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(embedding.flatten()))),
        y=embedding.flatten(),
        mode='lines',
        line=dict(color='#00d9ff', width=1),
        fill='tozeroy',
        fillcolor='rgba(0, 217, 255, 0.2)'
    ))
    fig.update_layout(
        title="Embedding Vector (512 dimensions)",
        xaxis_title="Dimension",
        yaxis_title="Value",
        plot_bgcolor='rgba(30,30,46,0.8)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#b8c5d6'),
        title_font=dict(color='#00d9ff', size=16),
        xaxis=dict(gridcolor='#3d3d5c'),
        yaxis=dict(gridcolor='#3d3d5c')
    )
    return fig


# 사이드바 - 파일 선택
with st.sidebar:
    st.header("🎵 Select Audio File")
    
    # 파일 소스 선택
    source_type = st.radio(
        "Choose source",
        ["Local Files (suno_mono)", "Upload File"]
    )
    
    selected_file = None
    uploaded_file = None
    
    if source_type == "Local Files (suno_mono)":
        # 로컬 suno_mono 디렉토리의 파일 목록
        mono_dir = "static_files/suno_mono"
        if os.path.exists(mono_dir):
            mp3_files = sorted([f for f in os.listdir(mono_dir) if f.lower().endswith('.mp3')])
            if mp3_files:
                selected_file = st.selectbox(
                    "Select a track",
                    options=mp3_files,
                    format_func=lambda x: x.replace('.mp3', '')
                )
            else:
                st.warning("No MP3 files found in suno_mono directory")
        else:
            st.warning("suno_mono directory not found")
    else:
        uploaded_file = st.file_uploader(
            "Upload an MP3 file",
            type=['mp3', 'wav', 'flac', 'ogg'],
            help="Supported formats: MP3, WAV, FLAC, OGG"
        )

# 메인 컨텐츠
if selected_file or uploaded_file:
    # 모델 로드 상태 표시
    with st.spinner("🔄 Loading CLAP model... (this may take a while on first run)"):
        try:
            processor, model = load_clap_model()
            model_loaded = True
        except Exception as e:
            st.error(f"❌ Failed to load model: {e}")
            model_loaded = False
    
    if model_loaded:
        # 파일 경로 결정
        if selected_file:
            audio_path = os.path.join("static_files/suno_mono", selected_file)
            file_name = selected_file
        else:
            # 업로드된 파일을 임시 저장
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                tmp.write(uploaded_file.read())
                audio_path = tmp.name
            file_name = uploaded_file.name
        
        # 파일 정보 표시
        st.markdown(f"### 🎵 Selected: **{file_name.replace('.mp3', '')}**")
        
        # 오디오 플레이어
        if selected_file:
            st.audio(audio_path)
        else:
            uploaded_file.seek(0)
            st.audio(uploaded_file)
        
        # 임베딩 추출 버튼
        if st.button("🚀 Extract Embedding", type="primary", use_container_width=True):
            with st.spinner("🔄 Extracting embedding..."):
                try:
                    embedding, audio_data, sr = extract_embedding(audio_path, processor, model)
                    
                    # 임베딩 통계 계산
                    emb_flat = embedding.squeeze()
                    l2_norm = np.linalg.norm(emb_flat)
                    
                    st.success("✅ Embedding extracted successfully!")
                    
                    # 오디오 정보
                    st.markdown("---")
                    st.markdown("### 📊 Audio Information")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Sample Rate", f"{sr} Hz")
                    with col2:
                        st.metric("Duration", f"{len(audio_data) / sr:.2f}s")
                    with col3:
                        st.metric("Total Samples", f"{len(audio_data):,}")
                    with col4:
                        st.metric("Channels", "Mono")
                    
                    # 임베딩 통계
                    st.markdown("---")
                    st.markdown("### 🎯 Embedding Statistics")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Dimensions", embedding.shape[-1])
                    with col2:
                        st.metric("L2 Norm", f"{l2_norm:.6f}")
                    with col3:
                        st.metric("Min Value", f"{emb_flat.min():.4f}")
                    with col4:
                        st.metric("Max Value", f"{emb_flat.max():.4f}")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Mean", f"{emb_flat.mean():.6f}")
                    with col2:
                        st.metric("Std Dev", f"{emb_flat.std():.6f}")
                    with col3:
                        st.metric("Median", f"{np.median(emb_flat):.6f}")
                    with col4:
                        st.metric("Non-zero", f"{np.count_nonzero(emb_flat)}")
                    
                    # 시각화
                    st.markdown("---")
                    st.markdown("### 📈 Visualization")
                    st.plotly_chart(plot_embedding_line(embedding), use_container_width=True)
                    
                    # Raw 임베딩 데이터 표시
                    st.markdown("---")
                    st.markdown("### 🔢 Raw Embedding Data")
                    with st.expander("View raw embedding vector (512 values)", expanded=False):
                        st.code(f"Shape: {embedding.shape}\n\nValues:\n{emb_flat.tolist()}")
                    
                    # 다운로드 옵션
                    st.markdown("---")
                    st.markdown("### 💾 Export")
                    col1, col2 = st.columns(2)
                    with col1:
                        # NumPy 형식으로 다운로드
                        import io
                        buffer = io.BytesIO()
                        np.save(buffer, emb_flat)
                        buffer.seek(0)
                        st.download_button(
                            label="📥 Download as .npy",
                            data=buffer,
                            file_name=f"{file_name.replace('.mp3', '')}_embedding.npy",
                            mime="application/octet-stream"
                        )
                    with col2:
                        # JSON 형식으로 다운로드
                        import json
                        json_data = json.dumps({
                            "file_name": file_name,
                            "embedding": emb_flat.tolist(),
                            "l2_norm": float(l2_norm),
                            "dimensions": int(embedding.shape[-1])
                        }, indent=2)
                        st.download_button(
                            label="📥 Download as .json",
                            data=json_data,
                            file_name=f"{file_name.replace('.mp3', '')}_embedding.json",
                            mime="application/json"
                        )
                    
                except Exception as e:
                    st.error(f"❌ Error extracting embedding: {e}")
                    import traceback
                    st.code(traceback.format_exc())
        
        # 업로드된 임시 파일 정리
        if uploaded_file and 'audio_path' in locals():
            try:
                os.unlink(audio_path)
            except:
                pass

else:
    # 안내 메시지
    st.info("👈 Select an audio file from the sidebar to extract its embedding")
    
    # CLAP 모델 정보
    st.markdown("---")
    st.markdown("### ℹ️ About CLAP Embeddings")
    st.markdown("""
    <div class="info-card">
        <h4 style="color: #00d9ff;">What is CLAP?</h4>
        <p style="color: #b8c5d6;">
            <strong>CLAP (Contrastive Language-Audio Pretraining)</strong> is a neural network trained to 
            understand the relationship between audio and text. It creates meaningful vector representations 
            (embeddings) of audio that capture semantic information about the sound.
        </p>
        <br>
        <h4 style="color: #00d9ff;">Model: laion/larger_clap_music</h4>
        <p style="color: #b8c5d6;">
            This model is specifically fine-tuned for music understanding. It produces 512-dimensional 
            embeddings that encode musical characteristics like genre, mood, instruments, and more.
        </p>
        <br>
        <h4 style="color: #00d9ff;">Use Cases</h4>
        <ul style="color: #b8c5d6;">
            <li>Music similarity search</li>
            <li>Music recommendation systems</li>
            <li>Audio classification</li>
            <li>Cross-modal retrieval (text-to-audio, audio-to-text)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# 푸터
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>Powered by 🤗 Transformers & CLAP</div>",
    unsafe_allow_html=True
)

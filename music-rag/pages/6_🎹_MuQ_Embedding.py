import os
import numpy as np
import dotenv
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    
    .segment-card {
        background: linear-gradient(145deg, #1e1e2e, #2d2d44);
        border: 1px solid #4a3d5c;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .segment-title {
        color: #ff6b9d;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        font-family: 'Space Mono', monospace;
    }
    
    .avg-card {
        background: linear-gradient(145deg, #2a1a3e, #3d2d5e);
        border: 2px solid #ff6b9d;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .info-card {
        background: linear-gradient(145deg, #1e1e2e, #2d2d44);
        border: 1px solid #4a3d5c;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .metric-highlight {
        color: #ff6b9d;
        font-size: 1.5rem;
        font-weight: 700;
        font-family: 'Space Mono', monospace;
    }
</style>
""", unsafe_allow_html=True)

# 헤더
st.markdown("""
<div class="main-header">
    <h1>🎹 MuQ-MuLan Embedding</h1>
    <p>Extract multi-segment embeddings using OpenMuQ/MuQ-MuLan-large</p>
</div>
""", unsafe_allow_html=True)

# 상수
TARGET_SR = 24000
SEGMENT_DURATION = 10  # 초


# MuQ-MuLan 모델 로드 (캐싱)
@st.cache_resource
def load_muq_model():
    """MuQ-MuLan 모델을 로드합니다."""
    from muq import MuQMuLan
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large")
    model = model.to(device).eval()
    return model, device


def extract_segments(audio, sr, segment_duration=SEGMENT_DURATION):
    """
    오디오에서 처음, 중간, 끝 부분의 세그먼트를 추출합니다.
    
    Returns:
        dict: segment_name -> (audio_data, start_time, end_time)
    """
    segment_samples = int(segment_duration * sr)
    total_samples = len(audio)
    total_duration = total_samples / sr
    
    segments = {}
    
    # 오디오가 너무 짧으면 전체를 사용
    if total_samples <= segment_samples:
        segments["Full Audio"] = (audio, 0, total_duration)
        return segments
    
    # 처음 10초
    start_segment = audio[:segment_samples]
    segments["🎬 Start (0-10s)"] = (start_segment, 0, segment_duration)
    
    # 중간 10초
    mid_start = (total_samples - segment_samples) // 2
    mid_time = mid_start / sr
    mid_segment = audio[mid_start:mid_start + segment_samples]
    segments["🎯 Middle"] = (mid_segment, mid_time, mid_time + segment_duration)
    
    # 끝 10초
    end_start = total_samples - segment_samples
    end_time = end_start / sr
    end_segment = audio[-segment_samples:]
    segments["🏁 End"] = (end_segment, end_time, total_duration)
    
    return segments


def extract_embedding(audio_segment, model, device):
    """세그먼트에서 MuQ-MuLan 임베딩을 추출합니다."""
    wav_tensor = torch.tensor(audio_segment).unsqueeze(0).to(device)
    
    with torch.no_grad():
        embedding = model(wavs=wav_tensor)
    
    return embedding.cpu().numpy()


def plot_embedding_line(embedding: np.ndarray, title: str, color: str = '#ff6b9d'):
    """임베딩 벡터를 선 그래프로 시각화합니다."""
    emb_flat = embedding.flatten()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(emb_flat))),
        y=emb_flat,
        mode='lines',
        line=dict(color=color, width=1),
        fill='tozeroy',
        fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.2)'
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Dimension",
        yaxis_title="Value",
        plot_bgcolor='rgba(30,30,46,0.8)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#d4a5c9'),
        title_font=dict(color=color, size=14),
        xaxis=dict(gridcolor='#3d3d5c'),
        yaxis=dict(gridcolor='#3d3d5c'),
        height=250,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig


def plot_all_embeddings_comparison(embeddings_dict: dict, avg_embedding: np.ndarray):
    """모든 임베딩을 하나의 그래프에서 비교합니다."""
    colors = {
        "🎬 Start (0-10s)": "#4ecdc4",
        "🎯 Middle": "#ffe66d",
        "🏁 End": "#ff6b9d",
        "Full Audio": "#ff6b9d"
    }
    
    fig = go.Figure()
    
    # 각 세그먼트 임베딩
    for name, emb in embeddings_dict.items():
        emb_flat = emb.flatten()
        color = colors.get(name, '#ffffff')
        fig.add_trace(go.Scatter(
            x=list(range(len(emb_flat))),
            y=emb_flat,
            mode='lines',
            name=name,
            line=dict(color=color, width=1.5),
            opacity=0.7
        ))
    
    # 평균 임베딩
    avg_flat = avg_embedding.flatten()
    fig.add_trace(go.Scatter(
        x=list(range(len(avg_flat))),
        y=avg_flat,
        mode='lines',
        name='📊 Average',
        line=dict(color='#ffffff', width=2.5, dash='dot'),
    ))
    
    fig.update_layout(
        title="Embedding Comparison (All Segments + Average)",
        xaxis_title="Dimension",
        yaxis_title="Value",
        plot_bgcolor='rgba(30,30,46,0.8)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#d4a5c9'),
        title_font=dict(color='#ff6b9d', size=16),
        xaxis=dict(gridcolor='#3d3d5c'),
        yaxis=dict(gridcolor='#3d3d5c'),
        height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(30,30,46,0.8)',
            bordercolor='#4a3d5c',
            borderwidth=1
        )
    )
    return fig


# 사이드바 - 파일 선택
with st.sidebar:
    st.header("🎵 Select Audio File")
    
    # 파일 소스 선택
    source_type = st.radio(
        "Choose source",
        ["Local Files (suno)", "Upload File"]
    )
    
    selected_file = None
    uploaded_file = None
    
    if source_type == "Local Files (suno)":
        # 로컬 suno 디렉토리의 파일 목록
        suno_dir = "static_files/suno"
        if os.path.exists(suno_dir):
            mp3_files = sorted([f for f in os.listdir(suno_dir) if f.lower().endswith('.mp3')])
            if mp3_files:
                selected_file = st.selectbox(
                    "Select a track",
                    options=mp3_files,
                    format_func=lambda x: x.replace('.mp3', '')
                )
            else:
                st.warning("No MP3 files found in suno directory")
        else:
            st.warning("suno directory not found")
    else:
        uploaded_file = st.file_uploader(
            "Upload an audio file",
            type=['mp3', 'wav', 'flac', 'ogg'],
            help="Supported formats: MP3, WAV, FLAC, OGG"
        )
    
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    segment_duration = st.slider(
        "Segment Duration (seconds)",
        min_value=5,
        max_value=30,
        value=10,
        step=5
    )

# 메인 컨텐츠
if selected_file or uploaded_file:
    # 모델 로드 상태 표시
    with st.spinner("🔄 Loading MuQ-MuLan model... (this may take a while on first run)"):
        try:
            model, device = load_muq_model()
            model_loaded = True
            st.sidebar.success(f"✅ Model loaded on {device}")
        except Exception as e:
            st.error(f"❌ Failed to load model: {e}")
            model_loaded = False
    
    if model_loaded:
        import librosa
        
        # 파일 경로 결정
        if selected_file:
            audio_path = os.path.join("static_files/suno", selected_file)
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
        
        # 전체 오디오 플레이어
        st.markdown("#### 🔊 Full Audio Preview")
        if selected_file:
            st.audio(audio_path)
        else:
            uploaded_file.seek(0)
            st.audio(uploaded_file)
        
        # 임베딩 추출 버튼
        if st.button("🚀 Extract MuQ-MuLan Embeddings", type="primary", use_container_width=True):
            with st.spinner("🔄 Loading audio and extracting segments..."):
                try:
                    # 오디오 로드 (24kHz, mono)
                    audio_data, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
                    total_duration = len(audio_data) / sr
                    
                    # 세그먼트 추출
                    segments = extract_segments(audio_data, sr, segment_duration)
                    
                    st.success(f"✅ Audio loaded! Duration: {total_duration:.2f}s, Segments: {len(segments)}")
                    
                except Exception as e:
                    st.error(f"❌ Error loading audio: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                    st.stop()
            
            # 오디오 정보
            st.markdown("---")
            st.markdown("### 📊 Audio Information")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Sample Rate", f"{sr} Hz")
            with col2:
                st.metric("Duration", f"{total_duration:.2f}s")
            with col3:
                st.metric("Total Samples", f"{len(audio_data):,}")
            with col4:
                st.metric("Segments", len(segments))
            
            # 각 세그먼트별 임베딩 추출 및 시각화
            st.markdown("---")
            st.markdown("### 🎯 Segment Embeddings")
            
            segment_embeddings = {}
            segment_colors = {
                "🎬 Start (0-10s)": "#4ecdc4",
                "🎯 Middle": "#ffe66d", 
                "🏁 End": "#ff6b9d",
                "Full Audio": "#ff6b9d"
            }
            
            for seg_name, (seg_audio, start_time, end_time) in segments.items():
                with st.container():
                    st.markdown(f"#### {seg_name} ({start_time:.1f}s - {end_time:.1f}s)")
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        # 세그먼트 오디오 미리보기
                        st.markdown("**🔊 Audio Preview**")
                        # numpy array를 bytes로 변환하여 재생
                        import io
                        import soundfile as sf
                        
                        audio_buffer = io.BytesIO()
                        sf.write(audio_buffer, seg_audio, sr, format='WAV')
                        audio_buffer.seek(0)
                        st.audio(audio_buffer, format='audio/wav')
                        
                        # 세그먼트 정보
                        st.markdown(f"**Duration:** {len(seg_audio)/sr:.2f}s")
                        st.markdown(f"**Samples:** {len(seg_audio):,}")
                    
                    with col2:
                        # 임베딩 추출
                        with st.spinner(f"Extracting embedding for {seg_name}..."):
                            embedding = extract_embedding(seg_audio, model, device)
                            segment_embeddings[seg_name] = embedding
                        
                        # 임베딩 시각화
                        color = segment_colors.get(seg_name, '#ff6b9d')
                        fig = plot_embedding_line(embedding, f"Embedding Vector", color)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # 통계
                        emb_flat = embedding.flatten()
                        stat_cols = st.columns(4)
                        with stat_cols[0]:
                            st.metric("Dimensions", len(emb_flat))
                        with stat_cols[1]:
                            st.metric("L2 Norm", f"{np.linalg.norm(emb_flat):.4f}")
                        with stat_cols[2]:
                            st.metric("Mean", f"{emb_flat.mean():.4f}")
                        with stat_cols[3]:
                            st.metric("Std", f"{emb_flat.std():.4f}")
                    
                    st.markdown("---")
            
            # 평균 임베딩 계산 및 시각화
            st.markdown("### 📊 Average Embedding")
            
            avg_embedding = np.mean(list(segment_embeddings.values()), axis=0)
            avg_flat = avg_embedding.flatten()
            
            # L2 정규화
            l2_norm = np.linalg.norm(avg_flat)
            normalized_avg = avg_flat / l2_norm
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📈 Before Normalization**")
                fig_avg = plot_embedding_line(avg_embedding, "Average Embedding (Raw)", '#ffffff')
                st.plotly_chart(fig_avg, use_container_width=True)
                
                stat_cols = st.columns(3)
                with stat_cols[0]:
                    st.metric("L2 Norm", f"{l2_norm:.6f}")
                with stat_cols[1]:
                    st.metric("Min", f"{avg_flat.min():.4f}")
                with stat_cols[2]:
                    st.metric("Max", f"{avg_flat.max():.4f}")
            
            with col2:
                st.markdown("**📈 After L2 Normalization**")
                fig_norm = plot_embedding_line(normalized_avg, "Average Embedding (Normalized)", '#ff6b9d')
                st.plotly_chart(fig_norm, use_container_width=True)
                
                stat_cols = st.columns(3)
                with stat_cols[0]:
                    st.metric("L2 Norm", f"{np.linalg.norm(normalized_avg):.6f}")
                with stat_cols[1]:
                    st.metric("Min", f"{normalized_avg.min():.4f}")
                with stat_cols[2]:
                    st.metric("Max", f"{normalized_avg.max():.4f}")
            
            # 전체 비교 차트
            st.markdown("---")
            st.markdown("### 🔍 All Embeddings Comparison")
            fig_comparison = plot_all_embeddings_comparison(segment_embeddings, avg_embedding)
            st.plotly_chart(fig_comparison, use_container_width=True)
            
            # 세그먼트 간 유사도
            st.markdown("---")
            st.markdown("### 📐 Segment Similarity Matrix")
            
            seg_names = list(segment_embeddings.keys())
            n_segments = len(seg_names)
            similarity_matrix = np.zeros((n_segments, n_segments))
            
            for i, name1 in enumerate(seg_names):
                for j, name2 in enumerate(seg_names):
                    emb1 = segment_embeddings[name1].flatten()
                    emb2 = segment_embeddings[name2].flatten()
                    # 코사인 유사도
                    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                    similarity_matrix[i, j] = similarity
            
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=similarity_matrix,
                x=seg_names,
                y=seg_names,
                colorscale='RdPu',
                text=np.round(similarity_matrix, 3),
                texttemplate='%{text}',
                textfont={"size": 12},
                hoverongaps=False
            ))
            fig_heatmap.update_layout(
                title="Cosine Similarity Between Segments",
                plot_bgcolor='rgba(30,30,46,0.8)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#d4a5c9'),
                title_font=dict(color='#ff6b9d', size=16),
                height=400
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # 다운로드 옵션
            st.markdown("---")
            st.markdown("### 💾 Export")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # 평균 임베딩 (정규화) NumPy
                import io
                buffer = io.BytesIO()
                np.save(buffer, normalized_avg)
                buffer.seek(0)
                st.download_button(
                    label="📥 Average Embedding (.npy)",
                    data=buffer,
                    file_name=f"{file_name.replace('.mp3', '')}_muq_avg_embedding.npy",
                    mime="application/octet-stream"
                )
            
            with col2:
                # 모든 세그먼트 임베딩 JSON
                import json
                all_data = {
                    "file_name": file_name,
                    "model": "OpenMuQ/MuQ-MuLan-large",
                    "sample_rate": TARGET_SR,
                    "segment_duration": segment_duration,
                    "segments": {
                        name: {
                            "embedding": emb.flatten().tolist(),
                            "l2_norm": float(np.linalg.norm(emb.flatten()))
                        }
                        for name, emb in segment_embeddings.items()
                    },
                    "average_embedding": {
                        "raw": avg_flat.tolist(),
                        "normalized": normalized_avg.tolist(),
                        "l2_norm_before": float(l2_norm)
                    }
                }
                json_data = json.dumps(all_data, indent=2)
                st.download_button(
                    label="📥 All Embeddings (.json)",
                    data=json_data,
                    file_name=f"{file_name.replace('.mp3', '')}_muq_all_embeddings.json",
                    mime="application/json"
                )
            
            with col3:
                # 유사도 매트릭스 CSV
                import pandas as pd
                df_sim = pd.DataFrame(similarity_matrix, index=seg_names, columns=seg_names)
                csv_data = df_sim.to_csv()
                st.download_button(
                    label="📥 Similarity Matrix (.csv)",
                    data=csv_data,
                    file_name=f"{file_name.replace('.mp3', '')}_muq_similarity.csv",
                    mime="text/csv"
                )
        
        # 업로드된 임시 파일 정리
        if uploaded_file and 'audio_path' in locals():
            try:
                os.unlink(audio_path)
            except:
                pass

else:
    # 안내 메시지
    st.info("👈 Select an audio file from the sidebar to extract MuQ-MuLan embeddings")
    
    # MuQ-MuLan 모델 정보
    st.markdown("---")
    st.markdown("### ℹ️ About MuQ-MuLan")
    st.markdown("""
    <div class="info-card">
        <h4 style="color: #ff6b9d;">What is MuQ-MuLan?</h4>
        <p style="color: #d4a5c9;">
            <strong>MuQ-MuLan</strong> is a music-language joint embedding model that maps music audio 
            and natural language descriptions into a shared embedding space. It supports both English 
            and Chinese text inputs.
        </p>
        <br>
        <h4 style="color: #ff6b9d;">Model: OpenMuQ/MuQ-MuLan-large</h4>
        <p style="color: #d4a5c9;">
            This model contains approximately 700M parameters and is trained for cross-modal retrieval, 
            similarity computation, and various music-language tasks.
        </p>
        <br>
        <h4 style="color: #ff6b9d;">Multi-Segment Approach</h4>
        <p style="color: #d4a5c9;">
            This tool extracts embeddings from <strong>three segments</strong> of the audio:
        </p>
        <ul style="color: #d4a5c9;">
            <li>🎬 <strong>Start</strong>: First 10 seconds</li>
            <li>🎯 <strong>Middle</strong>: Middle 10 seconds</li>
            <li>🏁 <strong>End</strong>: Last 10 seconds</li>
        </ul>
        <p style="color: #d4a5c9;">
            The average of these embeddings provides a more comprehensive representation of the entire track.
        </p>
        <br>
        <h4 style="color: #ff6b9d;">Technical Details</h4>
        <ul style="color: #d4a5c9;">
            <li>Sample Rate: 24kHz</li>
            <li>Input: Mono audio</li>
            <li>Output: High-dimensional embedding vector</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# 푸터
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>Powered by 🎹 MuQ-MuLan & OpenMuQ</div>",
    unsafe_allow_html=True
)


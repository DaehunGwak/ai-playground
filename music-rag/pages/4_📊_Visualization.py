import os
import json
import numpy as np
import pandas as pd
import dotenv
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from supabase import create_client

# 환경변수 로드
dotenv.load_dotenv(".env.local", override=True)

# 페이지 설정
st.set_page_config(
    page_title="📊 Embedding Visualization",
    page_icon="📊",
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
    
    .method-card {
        background: linear-gradient(145deg, #1a1a2e, #252545);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 3px solid #00d9ff;
    }
    
    .method-card h4 {
        color: #00d9ff;
        margin: 0 0 0.5rem 0;
    }
    
    .method-card p {
        color: #b8c5d6;
        margin: 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# 헤더
st.markdown("""
<div class="main-header">
    <h1>📊 Embedding Visualization</h1>
    <p>Explore music embeddings in 3D space using dimensionality reduction</p>
</div>
""", unsafe_allow_html=True)


# Supabase 클라이언트 초기화
@st.cache_resource
def get_supabase_client():
    """Supabase 클라이언트를 생성합니다."""
    return create_client(
        supabase_url=os.getenv("SUPABASE_URL"),
        supabase_key=os.getenv("SUPABASE_SECRET_KEY"),
    )


# 데이터 로드 함수
@st.cache_data(ttl=300)  # 5분 캐싱
def load_tracks_data():
    """Supabase에서 트랙 데이터를 가져옵니다."""
    supabase = get_supabase_client()
    result = supabase.schema('public').table('tracks') \
        .select('id, title, tags, embeddings') \
        .execute()
    return result.data


def parse_embedding(emb):
    """embedding을 list로 변환"""
    if isinstance(emb, str):
        return json.loads(emb)
    elif isinstance(emb, list):
        return emb
    return emb


# PCA 차원 축소
@st.cache_data
def compute_pca_3d(_embeddings: np.ndarray):
    """PCA로 3D 차원 축소를 수행합니다."""
    from sklearn.decomposition import PCA
    
    pca_3d = PCA(n_components=3, random_state=42)
    embeddings_pca = pca_3d.fit_transform(_embeddings)
    
    return embeddings_pca, pca_3d.explained_variance_ratio_


# t-SNE 차원 축소
@st.cache_data
def compute_tsne_3d(_embeddings: np.ndarray, perplexity: int = 30):
    """t-SNE로 3D 차원 축소를 수행합니다."""
    from sklearn.manifold import TSNE
    
    tsne_3d = TSNE(
        n_components=3,
        random_state=42,
        perplexity=min(perplexity, len(_embeddings) - 1),
        learning_rate='auto',
        init='pca'
    )
    embeddings_tsne = tsne_3d.fit_transform(_embeddings)
    
    return embeddings_tsne


# UMAP 차원 축소 (PCA 50차원 후)
@st.cache_data
def compute_umap_3d(_embeddings: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1):
    """PCA(50) + UMAP으로 3D 차원 축소를 수행합니다."""
    from sklearn.decomposition import PCA
    import umap
    
    # PCA로 50차원 축소
    pca_50 = PCA(n_components=50, random_state=42)
    embeddings_pca50 = pca_50.fit_transform(_embeddings)
    
    # UMAP 3D 적용
    umap_3d = umap.UMAP(
        n_components=3,
        random_state=42,
        metric='cosine',
        n_neighbors=n_neighbors,
        min_dist=min_dist
    )
    embeddings_umap = umap_3d.fit_transform(embeddings_pca50)
    
    return embeddings_umap


def create_3d_scatter(coords: np.ndarray, titles: list, tags_short: list, 
                      method_name: str, color: str, axis_labels: list):
    """3D 산점도를 생성합니다."""
    fig = px.scatter_3d(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        hover_name=titles,
        hover_data={'tags': tags_short},
        title=f'🎵 Music Embeddings - {method_name}',
        labels={'x': axis_labels[0], 'y': axis_labels[1], 'z': axis_labels[2]},
    )
    
    fig.update_traces(
        marker=dict(size=8, opacity=0.8, color=color),
        hovertemplate="<b>%{hovertext}</b><br>Tags: %{customdata[0]}<extra></extra>"
    )
    
    fig.update_layout(
        width=None,
        height=700,
        scene=dict(
            xaxis_title=axis_labels[0],
            yaxis_title=axis_labels[1],
            zaxis_title=axis_labels[2],
            xaxis=dict(backgroundcolor='rgba(30,30,46,0.8)', gridcolor='#3d3d5c'),
            yaxis=dict(backgroundcolor='rgba(30,30,46,0.8)', gridcolor='#3d3d5c'),
            zaxis=dict(backgroundcolor='rgba(30,30,46,0.8)', gridcolor='#3d3d5c'),
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#b8c5d6'),
        title_font=dict(color='#00d9ff', size=18),
    )
    
    return fig


# 메인 로직
try:
    # 데이터 로드
    with st.spinner("🔄 Loading track data from Supabase..."):
        tracks_data = load_tracks_data()
    
    if not tracks_data:
        st.error("❌ No track data found in database")
        st.stop()
    
    # DataFrame으로 변환
    tracks_df = pd.DataFrame(tracks_data)
    
    # embedding이 None이 아닌 데이터만 필터링
    tracks_df = tracks_df[tracks_df['embeddings'].notna()].reset_index(drop=True)
    
    if len(tracks_df) == 0:
        st.warning("⚠️ No tracks with embeddings found")
        st.stop()
    
    # embeddings를 numpy 배열로 변환
    embeddings_list = [parse_embedding(e) for e in tracks_df['embeddings']]
    embeddings = np.array(embeddings_list)
    
    # tags_short 생성
    tracks_df['tags_short'] = tracks_df['tags'].apply(
        lambda x: x[:20] + ('...' if len(x) > 20 else '') if isinstance(x, str) else ''
    )
    
    # 통계 표시
    st.markdown("### 📈 Dataset Statistics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Tracks", len(tracks_df))
    with col2:
        st.metric("Embedding Dimensions", embeddings.shape[1])
    with col3:
        st.metric("Avg L2 Norm", f"{np.mean([np.linalg.norm(e) for e in embeddings]):.4f}")
    with col4:
        st.metric("Data Shape", f"{embeddings.shape}")
    
    st.markdown("---")
    
    # 사이드바 - 시각화 설정
    with st.sidebar:
        st.header("⚙️ Visualization Settings")
        
        method = st.selectbox(
            "Dimensionality Reduction Method",
            ["PCA", "t-SNE", "UMAP (PCA 50 + UMAP)"],
            help="Choose the method for reducing 512D embeddings to 3D"
        )
        
        st.markdown("---")
        
        if method == "t-SNE":
            perplexity = st.slider(
                "Perplexity",
                min_value=5,
                max_value=min(50, len(embeddings) - 1),
                value=min(30, len(embeddings) - 1),
                help="Perplexity balances attention between local and global aspects of the data"
            )
        elif method == "UMAP (PCA 50 + UMAP)":
            n_neighbors = st.slider(
                "n_neighbors",
                min_value=2,
                max_value=min(50, len(embeddings) - 1),
                value=15,
                help="Number of neighboring points used in local approximations of manifold structure"
            )
            min_dist = st.slider(
                "min_dist",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.05,
                help="Minimum distance between points in the low dimensional space"
            )
        
        st.markdown("---")
        st.markdown("### 🎨 Display Options")
        
        color_scheme = st.selectbox(
            "Point Color",
            ["#636efa", "#00d9ff", "coral", "teal", "#FF6692", "#B6E880"],
            index=1
        )
        
        st.markdown("---")
        st.markdown("### ℹ️ About Methods")
        
        st.markdown("""
        <div class="method-card">
            <h4>PCA</h4>
            <p>Principal Component Analysis - 선형 차원 축소, 분산을 최대화하는 방향으로 투영</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="method-card">
            <h4>t-SNE</h4>
            <p>t-Distributed Stochastic Neighbor Embedding - 지역 구조 보존에 탁월</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="method-card">
            <h4>UMAP</h4>
            <p>Uniform Manifold Approximation and Projection - 전역+지역 구조 모두 보존</p>
        </div>
        """, unsafe_allow_html=True)
    
    # 시각화 수행
    st.markdown(f"### 🎯 {method} 3D Projection")
    
    with st.spinner(f"🔄 Computing {method} projection..."):
        if method == "PCA":
            coords, explained_variance = compute_pca_3d(embeddings)
            
            # 설명 분산 비율 표시
            st.markdown("#### 📊 Explained Variance Ratio")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("PC1", f"{explained_variance[0]:.2%}")
            with col2:
                st.metric("PC2", f"{explained_variance[1]:.2%}")
            with col3:
                st.metric("PC3", f"{explained_variance[2]:.2%}")
            with col4:
                st.metric("Total", f"{sum(explained_variance):.2%}")
            
            axis_labels = ['PC1', 'PC2', 'PC3']
            
        elif method == "t-SNE":
            coords = compute_tsne_3d(embeddings, perplexity)
            axis_labels = ['t-SNE 1', 't-SNE 2', 't-SNE 3']
            
        else:  # UMAP
            coords = compute_umap_3d(embeddings, n_neighbors, min_dist)
            axis_labels = ['UMAP 1', 'UMAP 2', 'UMAP 3']
    
    # 3D 플롯 생성 및 표시
    fig = create_3d_scatter(
        coords=coords,
        titles=tracks_df['title'].tolist(),
        tags_short=tracks_df['tags_short'].tolist(),
        method_name=method,
        color=color_scheme,
        axis_labels=axis_labels
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 트랙 목록 표시
    st.markdown("---")
    st.markdown("### 🎵 Track List")
    
    with st.expander("View all tracks", expanded=False):
        display_df = tracks_df[['id', 'title', 'tags_short']].copy()
        display_df.columns = ['ID', 'Title', 'Tags']
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # 좌표 데이터 다운로드
    st.markdown("---")
    st.markdown("### 💾 Export Data")
    
    col1, col2 = st.columns(2)
    with col1:
        # 좌표 데이터 CSV 다운로드
        export_df = tracks_df[['id', 'title', 'tags']].copy()
        export_df['x'] = coords[:, 0]
        export_df['y'] = coords[:, 1]
        export_df['z'] = coords[:, 2]
        
        csv_data = export_df.to_csv(index=False)
        st.download_button(
            label=f"📥 Download {method} Coordinates (CSV)",
            data=csv_data,
            file_name=f"music_embeddings_{method.lower().replace(' ', '_')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # JSON 형식
        json_data = export_df.to_json(orient='records', indent=2, force_ascii=False)
        st.download_button(
            label=f"📥 Download {method} Coordinates (JSON)",
            data=json_data,
            file_name=f"music_embeddings_{method.lower().replace(' ', '_')}.json",
            mime="application/json"
        )

except Exception as e:
    st.error(f"❌ Error: {e}")
    import traceback
    with st.expander("View error details"):
        st.code(traceback.format_exc())

# 푸터
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>Powered by scikit-learn, UMAP & Plotly</div>",
    unsafe_allow_html=True
)


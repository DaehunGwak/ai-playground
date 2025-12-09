import streamlit as st

# 페이지 설정
st.set_page_config(
    page_title="🎧 Music RAG",
    page_icon="🎧",
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
        padding: 3rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .main-header h1 {
        color: #fff;
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 0 40px rgba(255, 255, 255, 0.3);
    }
    
    .main-header p {
        color: #b8c5d6;
        font-size: 1.2rem;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# 헤더
st.markdown("""
<div class="main-header">
    <h1>🎧 Music RAG</h1>
    <p>AI-powered music analysis and retrieval system</p>
</div>
""", unsafe_allow_html=True)

# 페이지 소개
st.markdown("## 📚 Available Pages")

# 리스트 형식 페이지 링크
st.page_link("pages/2_🎵_Music_Player.py", label="Music Player", icon="🎵")
st.page_link("pages/1_🎯_Embedding.py", label="Audio Embedding", icon="🎯")
st.page_link("pages/3_🔍_Text_Query.py", label="Text Query Search", icon="🔍")

# 시스템 정보
st.markdown("---")
st.markdown("### ℹ️ About This Project")

st.markdown("""
<div style="background: linear-gradient(145deg, #1a1a2e, #252545); border-radius: 12px; padding: 1.5rem; margin-top: 1rem;">
    <p style="color: #b8c5d6; margin: 0; line-height: 1.8;">
        <strong style="color: #00d9ff;">Music RAG</strong> is an AI-powered music analysis system that combines:
    </p>
    <ul style="color: #b8c5d6; margin-top: 1rem; line-height: 2;">
        <li><strong style="color: #fff;">CLAP Embeddings</strong> - Convert audio into semantic vector representations</li>
        <li><strong style="color: #fff;">Supabase Storage</strong> - Store and manage music metadata and embeddings</li>
        <li><strong style="color: #fff;">Vector Search</strong> - Find similar music using embedding similarity</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# 푸터
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>Built with Streamlit, Supabase & 🤗 Transformers</div>",
    unsafe_allow_html=True
)

import streamlit as st

# 페이지 정의 - 계층형 네비게이션
pages = {
    "🎵 Music Player": [
        st.Page("pages/2_🎵_Music_Player.py", title="Music Player", icon="🎵"),
    ],
    "🔬 LAION CLAP": [
        st.Page("pages/1_🎯_Embedding.py", title="Embedding", icon="🎯"),
        st.Page("pages/3_🔍_Text_Query.py", title="Text Query", icon="🔍"),
        st.Page("pages/4_📊_Visualization.py", title="Visualization", icon="📊"),
        st.Page("pages/5_🤖_AI_Search.py", title="AI Search", icon="🤖"),
    ],
}

pg = st.navigation(pages)

# 페이지 설정
st.set_page_config(
    page_title="🎧 Music RAG",
    page_icon="🎧",
    layout="wide",
)

pg.run()

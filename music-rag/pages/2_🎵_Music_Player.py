import os
import dotenv
import streamlit as st
from supabase import create_client

# 환경변수 로드
dotenv.load_dotenv(".env.local", override=True)

# Supabase 클라이언트 초기화
@st.cache_resource
def get_supabase_client():
    return create_client(
        supabase_url=os.getenv("SUPABASE_URL"),
        supabase_key=os.getenv("SUPABASE_SECRET_KEY"),
    )

# 트랙 데이터 로드 (캐싱)
@st.cache_data(ttl=300)  # 5분 캐시
def load_all_tracks():
    client = get_supabase_client()
    page_size = 50
    offset = 0
    all_tracks = []

    while True:
        page_result = client.schema('public').table('tracks') \
            .select('*') \
            .range(offset, offset + page_size - 1) \
            .execute()
        data = page_result.data
        if not data:
            break
        all_tracks.extend(data)
        if len(data) < page_size:
            break
        offset += page_size

    return all_tracks

# 유사한 트랙 검색 (embedding 기반)
@st.cache_data(ttl=300)
def get_similar_tracks(track_embedding: list, exclude_track_id: int, match_count: int = 4):
    """
    주어진 트랙의 embedding을 기반으로 유사한 트랙을 검색합니다.
    자기 자신을 제외하기 위해 match_count + 1개를 가져온 후 필터링합니다.
    """
    if not track_embedding:
        return []
    
    client = get_supabase_client()
    try:
        response = client.rpc("search_tracks", {
            "query_embedding": track_embedding,
            "match_count": match_count
        }).execute()
        
        results = response.data if hasattr(response, 'data') else response
        # 자기 자신 제외
        similar_tracks = [t for t in results if t.get('id') != exclude_track_id]
        return similar_tracks[:3]  # 상위 3개만 반환
    except Exception as e:
        st.error(f"유사 트랙 검색 오류: {e}")
        return []

# 커스텀 CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .main-header h1 {
        color: #e94560;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 0 30px rgba(233, 69, 96, 0.5);
    }
    
    .main-header p {
        color: #a2d2ff;
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    .track-tags {
        color: #a2d2ff;
        font-size: 0.85rem;
        line-height: 1.6;
    }
    
    .tag-pill {
        background: rgba(233, 69, 96, 0.2);
        color: #e94560;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        display: inline-block;
        margin: 0.2rem;
    }
    
    .stats-container {
        background: linear-gradient(145deg, #0f3460, #1a1a2e);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin-bottom: 1.5rem;
        display: flex;
        justify-content: center;
        gap: 2rem;
    }
    
    .stat-item {
        text-align: center;
    }
    
    .stat-number {
        color: #e94560;
        font-size: 2rem;
        font-weight: 700;
    }
    
    .stat-label {
        color: #a2d2ff;
        font-size: 0.9rem;
    }
    
    /* 오디오 플레이어 스타일 */
    audio {
        width: 100%;
        border-radius: 8px;
    }
    
    /* Streamlit 기본 스타일 오버라이드 */
    .stAudio {
        background: transparent !important;
    }
    
    /* 비슷한 곡 카드 스타일 */
    .similar-track-card {
        background: linear-gradient(145deg, rgba(15, 52, 96, 0.5), rgba(26, 26, 46, 0.5));
        border: 1px solid rgba(233, 69, 96, 0.3);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    
    .similar-track-card:hover {
        border-color: #e94560;
        box-shadow: 0 4px 15px rgba(233, 69, 96, 0.2);
    }
    
    .similar-track-title {
        color: #fff;
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    
    .similar-track-similarity {
        color: #e94560;
        font-size: 0.8rem;
        font-weight: 500;
    }
    
    .similar-section-header {
        color: #a2d2ff;
        font-size: 0.95rem;
        font-weight: 500;
        margin: 1rem 0 0.5rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# 헤더
st.markdown("""
<div class="main-header">
    <h1>🎵 Music Player</h1>
    <p>Browse and play your music collection</p>
</div>
""", unsafe_allow_html=True)

# 데이터 로드
try:
    tracks = load_all_tracks()
except Exception as e:
    st.error(f"트랙을 불러오는 중 오류가 발생했습니다: {e}")
    tracks = []

if tracks:
    # 통계 표시
    st.markdown(f"""
    <div class="stats-container">
        <div class="stat-item">
            <div class="stat-number">{len(tracks)}</div>
            <div class="stat-label">Total Tracks</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 사이드바 - 검색 및 필터
    with st.sidebar:
        st.header("🔍 Search & Filter")
        
        search_query = st.text_input("Search by title or tags", placeholder="Enter keyword...")
        
        # 모든 태그 추출
        all_tags = set()
        for track in tracks:
            if track.get('tags'):
                tags = [t.strip() for t in track['tags'].split(',')]
                all_tags.update(tags)
        
        selected_tags = st.multiselect(
            "Filter by tags",
            options=sorted(all_tags),
            default=[]
        )
    
    # 트랙 필터링
    filtered_tracks = tracks
    
    if search_query:
        search_lower = search_query.lower()
        filtered_tracks = [
            t for t in filtered_tracks 
            if search_lower in t.get('title', '').lower() 
            or search_lower in t.get('tags', '').lower()
        ]
    
    if selected_tags:
        def has_tags(track):
            track_tags = [t.strip().lower() for t in track.get('tags', '').split(',')]
            return any(st.lower() in track_tags for st in selected_tags)
        filtered_tracks = [t for t in filtered_tracks if has_tags(t)]
    
    # 결과 표시
    st.subheader(f"🎧 {len(filtered_tracks)} tracks found")
    
    # 트랙 목록
    for idx, track in enumerate(filtered_tracks):
        with st.container():
            st.markdown(f"### {track.get('title', 'Untitled')}")
            
            # 태그 표시
            if track.get('tags'):
                tags = [t.strip() for t in track['tags'].split(',')]
                tags_html = ' '.join([f'<span class="tag-pill">{tag}</span>' for tag in tags[:5]])
                st.markdown(f'<div class="track-tags">{tags_html}</div>', unsafe_allow_html=True)
            
            # 아티스트 표시
            if track.get('artists'):
                st.caption(f"👤 {track['artists']}")
            
            # 트랙 ID 표시
            st.caption(f"ID: {track.get('id', 'N/A')}")
            
            # 오디오 플레이어
            if track.get('audioUrl'):
                st.audio(track['audioUrl'])
            else:
                st.warning("Audio URL not available")
            
            # 비슷한 곡 섹션
            if track.get('embeddings'):
                with st.expander("🎧 비슷한 곡 보기", expanded=False):
                    similar_tracks = get_similar_tracks(
                        track_embedding=track['embeddings'],
                        exclude_track_id=track.get('id'),
                        match_count=4
                    )
                    
                    if similar_tracks:
                        cols = st.columns(3)
                        for col_idx, similar_track in enumerate(similar_tracks):
                            with cols[col_idx]:
                                similarity = similar_track.get('similarity', 0)
                                similarity_pct = f"{similarity:.7f}" if similarity else "N/A"
                                
                                st.markdown(f"""
                                <div class="similar-track-card">
                                    <div class="similar-track-title">🎵 {similar_track.get('title', 'Untitled')}</div>
                                    <div class="similar-track-similarity">거리: {similarity_pct}</div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # 태그 표시 (간략하게)
                                if similar_track.get('tags'):
                                    sim_tags = [t.strip() for t in similar_track['tags'].split(',')]
                                    st.caption(f"🏷️ {', '.join(sim_tags[:3])}")
                                
                                # 오디오 플레이어
                                if similar_track.get('audioUrl'):
                                    st.audio(similar_track['audioUrl'])
                    else:
                        st.info("비슷한 곡을 찾을 수 없습니다.")
            
            st.divider()

else:
    st.info("🎵 No tracks found. Please check your database connection.")

# 푸터
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>Built with Streamlit & Supabase</div>",
    unsafe_allow_html=True
)

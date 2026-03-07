"""
음악 관련 텍스트 100건 임베딩 분포 분석 스크립트
"""

import json
import math
import sys
import urllib.request
import urllib.error

BASE_URL = "http://localhost:8000"

MUSIC_SAMPLES = [
    # 장르
    "classical music orchestra",
    "jazz saxophone improvisation",
    "blues guitar with harmonica",
    "hip hop beat with rap vocals",
    "electronic dance music with synthesizer",
    "country music with acoustic guitar",
    "reggae rhythm with bass",
    "folk music with banjo",
    "metal guitar with heavy distortion",
    "R&B smooth vocals",
    # 악기
    "grand piano solo performance",
    "violin playing a sad melody",
    "electric guitar riff",
    "acoustic guitar fingerpicking",
    "drum kit playing a fast beat",
    "bass guitar groove",
    "cello playing classical piece",
    "flute melody in the wind",
    "trumpet jazz solo",
    "organ church music",
    # 분위기/감정
    "upbeat happy dance music",
    "sad melancholic piano ballad",
    "intense dramatic orchestral music",
    "relaxing ambient background music",
    "romantic slow love song",
    "energetic workout pump up music",
    "peaceful meditation music",
    "dark mysterious atmospheric music",
    "joyful children's music",
    "nostalgic vintage music",
    # 보컬
    "female soprano opera singing",
    "male baritone classical singing",
    "choir singing in harmony",
    "soulful gospel singing",
    "boy band pop singing",
    "raspy rock vocal screaming",
    "whispering intimate vocal",
    "a cappella group vocals",
    "autotune electronic vocal effect",
    "vocal warm-up exercises",
    # 리듬/템포
    "fast tempo allegro music",
    "slow tempo adagio music",
    "syncopated rhythm jazz swing",
    "waltz three four time signature",
    "heavy metal double bass drum",
    "bossa nova Brazilian rhythm",
    "African drumming polyrhythm",
    "electronic music four on the floor beat",
    "slow ballad music",
    "accelerating music crescendo",
    # 장소/상황
    "live concert crowd cheering",
    "street busker playing guitar",
    "music played in a cathedral",
    "jazz played in a smoky bar",
    "outdoor festival music stage",
    "music in a recording studio",
    "lullaby for sleeping baby",
    "wedding ceremony music march",
    "music at a funeral",
    "birthday party celebration music",
    # 시대/스타일
    "1950s rock and roll music",
    "1970s disco funk music",
    "1980s synth pop music",
    "1990s grunge alternative rock",
    "2000s indie pop music",
    "baroque harpsichord music",
    "Renaissance madrigal music",
    "Romantic era symphony",
    "modern minimalist music",
    "avant garde experimental music",
    # 지역/문화
    "Indian classical sitar music",
    "Flamenco Spanish guitar music",
    "Celtic Irish fiddle music",
    "African traditional drum music",
    "Japanese koto music",
    "Middle Eastern oud music",
    "Latin salsa music",
    "Brazilian samba music",
    "Korean traditional gayageum",
    "Chinese erhu string music",
    # 음향 특성
    "music with heavy reverb",
    "dry close-up recorded music",
    "music with echo in large hall",
    "distorted lo-fi music",
    "crystal clear high fidelity music",
    "music with vinyl record crackle",
    "music with wind noise outdoors",
    "music played through old radio",
    "underwater muffled music",
    "music with audience applause",
    # 기타
    "music box playing a tune",
    "whistling a melody",
    "humming a soft tune",
    "beatboxing rhythmic sounds",
    "music played backwards",
    "silence between musical notes",
    "musical scale ascending",
    "dissonant chord music",
    "consonant harmony music",
    "music fading away into silence",
]


def embed_texts(texts: list[str]) -> list[list[float]]:
    payload = json.dumps({"texts": texts}).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/embed/text",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())["embeddings"]


def cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x**2 for x in a))
    nb = math.sqrt(sum(x**2 for x in b))
    return dot / (na * nb) if na and nb else 0.0


def stats(values: list[float]) -> dict:
    n = len(values)
    mean = sum(values) / n
    std = math.sqrt(sum((x - mean) ** 2 for x in values) / n)
    sorted_v = sorted(values)
    return {
        "min": sorted_v[0],
        "max": sorted_v[-1],
        "mean": mean,
        "std": std,
        "median": sorted_v[n // 2],
        "p25": sorted_v[n // 4],
        "p75": sorted_v[3 * n // 4],
    }


def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    assert len(MUSIC_SAMPLES) == 100, f"샘플 수 오류: {len(MUSIC_SAMPLES)}"

    print(f"총 {len(MUSIC_SAMPLES)}개 텍스트 임베딩 요청 중...")

    # 50개씩 나눠서 요청 (서버 부하 분산)
    batch1 = MUSIC_SAMPLES[:50]
    batch2 = MUSIC_SAMPLES[50:]
    embeddings = embed_texts(batch1) + embed_texts(batch2)

    dim = len(embeddings[0])
    print(f"완료: {len(embeddings)}개 임베딩, 차원={dim}")

    # ── 1. 개별 임베딩 통계 ──────────────────────────────────
    print_section("1. 개별 임베딩 통계 (mean / std / L2 norm)")
    print(f"{'#':>3}  {'텍스트':<45}  {'mean':>7}  {'std':>6}  {'L2':>6}")
    print("-" * 72)
    for i, (text, emb) in enumerate(zip(MUSIC_SAMPLES, embeddings)):
        mean = sum(emb) / dim
        std = math.sqrt(sum((x - mean) ** 2 for x in emb) / dim)
        norm = math.sqrt(sum(x**2 for x in emb))
        print(f"{i+1:>3}. {text[:44]:<44}  {mean:+.4f}  {std:.4f}  {norm:.4f}")

    # ── 2. 쌍별 코사인 유사도 분포 ───────────────────────────
    print_section("2. 쌍별 코사인 유사도 분포 (4950쌍)")
    all_sims = [
        cosine_sim(embeddings[i], embeddings[j])
        for i in range(len(embeddings))
        for j in range(i + 1, len(embeddings))
    ]
    s = stats(all_sims)
    print(f"  총 쌍 수 : {len(all_sims)}")
    print(f"  최솟값   : {s['min']:+.4f}")
    print(f"  P25      : {s['p25']:+.4f}")
    print(f"  중앙값   : {s['median']:+.4f}")
    print(f"  P75      : {s['p75']:+.4f}")
    print(f"  최댓값   : {s['max']:+.4f}")
    print(f"  평균     : {s['mean']:+.4f}")
    print(f"  표준편차 : {s['std']:.4f}")

    # 히스토그램
    bins = 10
    bin_min, bin_max = -0.2, 1.0
    bin_w = (bin_max - bin_min) / bins
    counts = [0] * bins
    for v in all_sims:
        idx = min(int((v - bin_min) / bin_w), bins - 1)
        if 0 <= idx < bins:
            counts[idx] += 1
    total = len(all_sims)
    print("\n  히스토그램:")
    for i, c in enumerate(counts):
        lo = bin_min + i * bin_w
        hi = lo + bin_w
        bar = "█" * int(c / total * 60)
        print(f"  [{lo:+.2f}, {hi:+.2f})  {bar:<60} {c:4d} ({c/total*100:.1f}%)")

    # ── 3. 상위/하위 유사쌍 ────────────────────────────────
    print_section("3. 유사도 TOP 15 쌍 (가장 유사)")
    pairs = [
        (cosine_sim(embeddings[i], embeddings[j]), i, j)
        for i in range(len(embeddings))
        for j in range(i + 1, len(embeddings))
    ]
    pairs.sort(reverse=True)
    print(f"  {'유사도':>7}  텍스트 A  ↔  텍스트 B")
    print(f"  {'-'*7}  {'-'*55}")
    for sim, i, j in pairs[:15]:
        print(f"  {sim:+.4f}  {MUSIC_SAMPLES[i][:26]:<26}  ↔  {MUSIC_SAMPLES[j][:26]}")

    print_section("4. 유사도 BOTTOM 15 쌍 (가장 상이)")
    print(f"  {'유사도':>7}  텍스트 A  ↔  텍스트 B")
    print(f"  {'-'*7}  {'-'*55}")
    for sim, i, j in pairs[-15:][::-1]:
        print(f"  {sim:+.4f}  {MUSIC_SAMPLES[i][:26]:<26}  ↔  {MUSIC_SAMPLES[j][:26]}")

    # ── 5. 카테고리 간 평균 유사도 ────────────────────────────
    print_section("5. 카테고리 내 vs 카테고리 간 평균 유사도")
    categories = {
        "장르(0-9)":    list(range(0, 10)),
        "악기(10-19)":  list(range(10, 20)),
        "분위기(20-29)":list(range(20, 30)),
        "보컬(30-39)":  list(range(30, 40)),
        "리듬(40-49)":  list(range(40, 50)),
        "상황(50-59)":  list(range(50, 60)),
        "시대(60-69)":  list(range(60, 70)),
        "지역(70-79)":  list(range(70, 80)),
        "음향(80-89)":  list(range(80, 90)),
        "기타(90-99)":  list(range(90, 100)),
    }
    cat_names = list(categories.keys())
    cat_indices = list(categories.values())

    def cat_avg_sim(idx_a, idx_b):
        sims = []
        for i in idx_a:
            for j in idx_b:
                if i != j:
                    sims.append(cosine_sim(embeddings[i], embeddings[j]))
        return sum(sims) / len(sims) if sims else 0.0

    # 헤더
    header = f"  {'':12}" + "".join(f"{n[:6]:>8}" for n in cat_names)
    print(header)
    for a_name, a_idx in zip(cat_names, cat_indices):
        row = f"  {a_name[:12]:<12}"
        for b_idx in cat_indices:
            row += f"  {cat_avg_sim(a_idx, b_idx):+.3f}"
        print(row)

    print("\n분석 완료.")


if __name__ == "__main__":
    try:
        main()
    except urllib.error.URLError as e:
        print(f"서버 연결 실패: {e}\n서버가 실행 중인지 확인하세요.")
        sys.exit(1)

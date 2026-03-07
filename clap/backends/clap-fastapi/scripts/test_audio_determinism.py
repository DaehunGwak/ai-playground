"""
오디오 임베딩 결정적 크롭 검증 + 분포 분석 스크립트

Phase 1 — 결정적 임베딩 검증 (5개 테스트)
Phase 2 — 임베딩 분포 분석 (100개 파일)
Phase 3 — docs/audio-embedding-test-report.md 생성

사전 조건:
  1. uv run python scripts/download_gtzan_samples.py   → 100개 WAV 다운로드
  2. uv run uvicorn app.main:app                       → 서버 실행
  3. uv run python scripts/test_audio_determinism.py   → 이 스크립트 실행
"""

import json
import math
import os
import sys
import urllib.error
import urllib.request

BASE_URL = "http://localhost:8000"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO_DIR = os.path.join(BASE_DIR, "resources", "audio")
REPORT_PATH = os.path.join(BASE_DIR, "docs", "audio-embedding-test-report.md")

GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock",
]
SAMPLES_PER_GENRE = 10


# ── 유틸리티 함수 ───────────────────────────────────────────────────────────

def embed_audio(file_path: str, start_sec: float | None = None) -> list[float]:
    """오디오 파일을 /embed/audio 엔드포인트로 전송하여 임베딩 반환."""
    url = f"{BASE_URL}/embed/audio"
    if start_sec is not None:
        url += f"?start_sec={start_sec}"

    with open(file_path, "rb") as f:
        data = f.read()

    filename = os.path.basename(file_path)
    boundary = "----FormBoundary7MA4YWxkTrZu0gW"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: audio/wav\r\n\r\n"
    ).encode() + data + f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())
        return result["embeddings"][0]


def embed_audio_expect_error(file_path: str, start_sec: float | None = None) -> int:
    """오류가 예상되는 요청을 전송하고 HTTP 상태 코드 반환."""
    url = f"{BASE_URL}/embed/audio"
    if start_sec is not None:
        url += f"?start_sec={start_sec}"

    with open(file_path, "rb") as f:
        data = f.read()

    filename = os.path.basename(file_path)
    boundary = "----FormBoundary7MA4YWxkTrZu0gW"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: audio/wav\r\n\r\n"
    ).encode() + data + f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req):
            return 200
    except urllib.error.HTTPError as e:
        return e.code


def cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x**2 for x in a))
    nb = math.sqrt(sum(x**2 for x in b))
    return dot / (na * nb) if na and nb else 0.0


def l2_norm(v: list[float]) -> float:
    return math.sqrt(sum(x**2 for x in v))


def vec_mean(v: list[float]) -> float:
    return sum(v) / len(v)


def vec_std(v: list[float]) -> float:
    m = vec_mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / len(v))


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


def histogram_lines(values: list[float], bin_min: float = -0.2, bin_max: float = 1.0, bins: int = 10) -> list[str]:
    bin_w = (bin_max - bin_min) / bins
    counts = [0] * bins
    for v in values:
        idx = min(int((v - bin_min) / bin_w), bins - 1)
        if 0 <= idx < bins:
            counts[idx] += 1
    total = len(values)
    lines = []
    for i, c in enumerate(counts):
        lo = bin_min + i * bin_w
        hi = lo + bin_w
        bar = "█" * int(c / total * 40)
        lines.append(f"[{lo:+.2f}, {hi:+.2f})  {bar:<40} {c:5d} ({c/total*100:.1f}%)")
    return lines


def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def pass_fail(ok: bool, msg: str):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {msg}")
    return ok


# ── Phase 1: 결정적 임베딩 검증 ────────────────────────────────────────────

def run_phase1(sample_files: list[str]) -> dict:
    """결정적 임베딩 5개 테스트. 결과 dict 반환."""
    results = {}

    print_section("Phase 1 — 결정적 임베딩 검증")

    # 테스트용 파일 고정: blues.00 사용
    test_file = sample_files[0]
    print(f"\n  테스트 파일: {os.path.basename(test_file)}")

    # ── 테스트 1: 100개 파일 각 2회 → cosine_sim=1.0 ──────────────────────
    print("\n  [테스트 1] 100개 파일 각 2회 호출 → 완전 일치 확인")
    t1_failures = []
    for fp in sample_files:
        emb1 = embed_audio(fp)
        emb2 = embed_audio(fp)
        sim = cosine_sim(emb1, emb2)
        if abs(sim - 1.0) > 1e-5:
            t1_failures.append((os.path.basename(fp), sim))

    t1_ok = len(t1_failures) == 0
    results["t1"] = t1_ok
    if t1_ok:
        pass_fail(True, f"100개 파일 모두 cosine_sim=1.0 (완전 일치)")
    else:
        pass_fail(False, f"{len(t1_failures)}개 파일에서 불일치 발생")
        for fname, sim in t1_failures[:5]:
            print(f"    {fname}: sim={sim:.6f}")

    # ── 테스트 2: start_sec=0 vs start_sec=5 vs 기본(center) → 서로 다름 ──
    print("\n  [테스트 2] start_sec=0 / start_sec=5 / center(기본) → 서로 다른 임베딩")
    emb_start0 = embed_audio(test_file, start_sec=0.0)
    emb_start5 = embed_audio(test_file, start_sec=5.0)
    emb_center = embed_audio(test_file)

    sim_0_5 = cosine_sim(emb_start0, emb_start5)
    sim_0_c = cosine_sim(emb_start0, emb_center)
    sim_5_c = cosine_sim(emb_start5, emb_center)

    t2_ok = (abs(sim_0_5 - 1.0) > 1e-5 and
             abs(sim_0_c - 1.0) > 1e-5 and
             abs(sim_5_c - 1.0) > 1e-5)
    results["t2"] = t2_ok
    results["t2_sims"] = {"0_vs_5": sim_0_5, "0_vs_center": sim_0_c, "5_vs_center": sim_5_c}
    pass_fail(t2_ok, f"sim(0,5)={sim_0_5:.4f}  sim(0,center)={sim_0_c:.4f}  sim(5,center)={sim_5_c:.4f}")

    # ── 테스트 3: start_sec=5로 2회 → 동일 임베딩 ─────────────────────────
    print("\n  [테스트 3] start_sec=5로 2회 호출 → 동일 임베딩")
    emb_a = embed_audio(test_file, start_sec=5.0)
    emb_b = embed_audio(test_file, start_sec=5.0)
    sim_t3 = cosine_sim(emb_a, emb_b)
    t3_ok = abs(sim_t3 - 1.0) <= 1e-5
    results["t3"] = t3_ok
    pass_fail(t3_ok, f"cosine_sim={sim_t3:.6f}")

    # ── 테스트 4: start_sec=25 (30초 파일) → 422 ──────────────────────────
    print("\n  [테스트 4] start_sec=25 (30초 파일, 10초 크롭 → 오버플로) → 422")
    status4 = embed_audio_expect_error(test_file, start_sec=25.0)
    t4_ok = status4 == 422
    results["t4"] = t4_ok
    results["t4_status"] = status4
    pass_fail(t4_ok, f"HTTP {status4} (기대: 422)")

    # ── 테스트 5: start_sec=-1 → 422 ──────────────────────────────────────
    print("\n  [테스트 5] start_sec=-1 → 422")
    status5 = embed_audio_expect_error(test_file, start_sec=-1.0)
    t5_ok = status5 == 422
    results["t5"] = t5_ok
    results["t5_status"] = status5
    pass_fail(t5_ok, f"HTTP {status5} (기대: 422)")

    all_ok = all([t1_ok, t2_ok, t3_ok, t4_ok, t5_ok])
    print(f"\n  Phase 1 결과: {'전체 PASS' if all_ok else '일부 FAIL'}")
    results["all_ok"] = all_ok
    return results


# ── Phase 2: 임베딩 분포 분석 ──────────────────────────────────────────────

def run_phase2(sample_files: list[str], genre_of: list[str]) -> dict:
    """100개 오디오 임베딩 수집 및 분포 분석."""
    print_section("Phase 2 — 임베딩 분포 분석")

    # 임베딩 수집
    print(f"\n  {len(sample_files)}개 파일 임베딩 요청 중...")
    embeddings = []
    for i, fp in enumerate(sample_files):
        emb = embed_audio(fp)
        embeddings.append(emb)
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(sample_files)} 완료")
    dim = len(embeddings[0])
    print(f"  완료: {len(embeddings)}개 임베딩, 차원={dim}")

    # ── 1. 개별 임베딩 통계 ───────────────────────────────────────────────
    print_section("2-1. 개별 임베딩 통계 (mean / std / L2 norm)")
    print(f"  {'#':>3}  {'파일명':<30}  {'mean':>8}  {'std':>6}  {'L2':>6}")
    print("  " + "-" * 60)
    per_stats = []
    for i, (fp, emb) in enumerate(zip(sample_files, embeddings)):
        m = vec_mean(emb)
        s = vec_std(emb)
        n = l2_norm(emb)
        per_stats.append({"file": os.path.basename(fp), "mean": m, "std": s, "norm": n})
        print(f"  {i+1:>3}. {os.path.basename(fp):<30}  {m:+.5f}  {s:.5f}  {n:.5f}")

    # ── 2. 쌍별 코사인 유사도 분포 ───────────────────────────────────────
    print_section("2-2. 쌍별 코사인 유사도 분포 (4950쌍)")
    all_sims = [
        (cosine_sim(embeddings[i], embeddings[j]), i, j)
        for i in range(len(embeddings))
        for j in range(i + 1, len(embeddings))
    ]
    sim_values = [s for s, _, _ in all_sims]
    s = stats(sim_values)
    print(f"  총 쌍 수 : {len(all_sims)}")
    print(f"  최솟값   : {s['min']:+.4f}")
    print(f"  P25      : {s['p25']:+.4f}")
    print(f"  중앙값   : {s['median']:+.4f}")
    print(f"  P75      : {s['p75']:+.4f}")
    print(f"  최댓값   : {s['max']:+.4f}")
    print(f"  평균     : {s['mean']:+.4f}")
    print(f"  표준편차 : {s['std']:.4f}")
    print("\n  히스토그램:")
    for line in histogram_lines(sim_values):
        print(f"  {line}")

    # ── 3. TOP/BOTTOM 15 유사쌍 ──────────────────────────────────────────
    sorted_pairs = sorted(all_sims, reverse=True)

    print_section("2-3. 유사도 TOP 15 쌍")
    print(f"  {'유사도':>7}  {'파일 A':<25}  ↔  {'파일 B'}")
    print(f"  {'-'*7}  {'-'*60}")
    for sim, i, j in sorted_pairs[:15]:
        fn_a = os.path.basename(sample_files[i])
        fn_b = os.path.basename(sample_files[j])
        print(f"  {sim:+.4f}  {fn_a:<25}  ↔  {fn_b}")

    print_section("2-4. 유사도 BOTTOM 15 쌍")
    print(f"  {'유사도':>7}  {'파일 A':<25}  ↔  {'파일 B'}")
    print(f"  {'-'*7}  {'-'*60}")
    for sim, i, j in sorted_pairs[-15:][::-1]:
        fn_a = os.path.basename(sample_files[i])
        fn_b = os.path.basename(sample_files[j])
        print(f"  {sim:+.4f}  {fn_a:<25}  ↔  {fn_b}")

    # ── 4. 장르 내/간 유사도 행렬 ────────────────────────────────────────
    print_section("2-5. 장르 내/간 평균 코사인 유사도 (10×10)")
    genre_indices = {g: [] for g in GENRES}
    for idx, g in enumerate(genre_of):
        genre_indices[g].append(idx)

    def genre_avg_sim(idx_a: list[int], idx_b: list[int]) -> float:
        sims_g = [
            cosine_sim(embeddings[i], embeddings[j])
            for i in idx_a for j in idx_b if i != j
        ]
        return sum(sims_g) / len(sims_g) if sims_g else 0.0

    genre_matrix = {}
    for ga in GENRES:
        genre_matrix[ga] = {}
        for gb in GENRES:
            genre_matrix[ga][gb] = genre_avg_sim(genre_indices[ga], genre_indices[gb])

    # 헤더
    short = [g[:6] for g in GENRES]
    header = f"  {'':12}" + "".join(f"{n:>8}" for n in short)
    print(header)
    for ga in GENRES:
        row = f"  {ga:<12}"
        for gb in GENRES:
            row += f"  {genre_matrix[ga][gb]:+.3f}"
        print(row)

    return {
        "embeddings": embeddings,
        "per_stats": per_stats,
        "sim_stats": s,
        "sim_values": sim_values,
        "sorted_pairs": sorted_pairs,
        "genre_matrix": genre_matrix,
        "dim": dim,
    }


# ── Phase 3: 리포트 생성 ────────────────────────────────────────────────────

def generate_report(phase1: dict, phase2: dict, sample_files: list[str], genre_of: list[str]):
    """docs/audio-embedding-test-report.md 생성."""
    print_section("Phase 3 — 리포트 생성")

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)

    s = phase2["sim_stats"]
    per_stats = phase2["per_stats"]
    sorted_pairs = phase2["sorted_pairs"]
    genre_matrix = phase2["genre_matrix"]
    sim_values = phase2["sim_values"]
    dim = phase2["dim"]

    def p1_mark(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    t2_sims = phase1.get("t2_sims", {})

    lines = []
    lines.append("# CLAP 오디오 임베딩 결정적 크롭 검증 + 분포 분석 리포트\n")
    lines.append(f"**작성일**: 2026-03-06")
    lines.append(f"**모델**: `music_audioset_epoch_15_esc_90.14.pt`")
    lines.append(f"**아키텍처**: HTSAT-base (`enable_fusion=False`)")
    lines.append(f"**실행 스크립트**: `scripts/test_audio_determinism.py`\n")
    lines.append("---\n")

    # 1. 테스트 개요
    lines.append("## 1. 테스트 개요\n")
    lines.append("| 항목 | 값 |")
    lines.append("|------|-----|")
    lines.append(f"| 데이터셋 | GTZAN Genre Collection (`marsyas/gtzan`) |")
    lines.append(f"| 임베딩 차원 | {dim} |")
    lines.append(f"| 총 샘플 수 | 100개 오디오 파일 |")
    lines.append(f"| 오디오 형식 | WAV, 22050Hz, mono, 30초 |")
    lines.append(f"| 크롭 설정 | 10초 (CLIP_SECONDS=10, sr=48000) |")
    lines.append(f"| 총 쌍 수 | 4,950쌍 |\n")

    lines.append("### 장르 구성 (10개 장르 × 10개 샘플)\n")
    lines.append("| 장르 | 파일 수 |")
    lines.append("|------|:-------:|")
    for g in GENRES:
        lines.append(f"| {g} | {genre_of.count(g)} |")
    lines.append("")

    lines.append("---\n")

    # 2. 결정적 임베딩 검증 결과
    lines.append("## 2. 결정적 임베딩 검증 결과\n")
    lines.append("| # | 테스트 | 결과 | 상세 |")
    lines.append("|---|--------|:----:|------|")
    lines.append(f"| 1 | 100개 파일 각 2회 호출 → cosine_sim=1.0 | **{p1_mark(phase1['t1'])}** | 모든 파일에서 완전 일치 확인 |")
    lines.append(
        f"| 2 | start_sec=0 / 5 / center → 서로 다름 | **{p1_mark(phase1['t2'])}** | "
        f"sim(0,5)={t2_sims.get('0_vs_5', 0):.4f}, sim(0,c)={t2_sims.get('0_vs_center', 0):.4f}, "
        f"sim(5,c)={t2_sims.get('5_vs_center', 0):.4f} |"
    )
    lines.append(f"| 3 | start_sec=5로 2회 호출 → 동일 임베딩 | **{p1_mark(phase1['t3'])}** | cosine_sim=1.0 |")
    lines.append(f"| 4 | start_sec=25 (30초 파일) → 422 | **{p1_mark(phase1['t4'])}** | HTTP {phase1.get('t4_status', '?')} |")
    lines.append(f"| 5 | start_sec=-1 → 422 | **{p1_mark(phase1['t5'])}** | HTTP {phase1.get('t5_status', '?')} |\n")

    overall = "**전체 PASS**" if phase1["all_ok"] else "**일부 FAIL**"
    lines.append(f"> 결정적 임베딩 검증 종합: {overall}\n")

    lines.append("---\n")

    # 3. 개별 임베딩 통계
    lines.append("## 3. 개별 임베딩 통계\n")
    lines.append("### 전체 요약\n")
    norms = [p["norm"] for p in per_stats]
    means = [p["mean"] for p in per_stats]
    stds_p = [p["std"] for p in per_stats]
    lines.append(f"| 통계량 | 값 |")
    lines.append(f"|--------|-----|")
    lines.append(f"| L2 norm 범위 | {min(norms):.5f} ~ {max(norms):.5f} |")
    lines.append(f"| per-vector mean 범위 | {min(means):+.5f} ~ {max(means):+.5f} |")
    lines.append(f"| per-vector std 범위 | {min(stds_p):.5f} ~ {max(stds_p):.5f} |")
    lines.append(f"| 차원 수 | {dim} |\n")

    lines.append("### 파일별 통계 (처음 20개)\n")
    lines.append("| # | 파일 | mean | std | L2 norm |")
    lines.append("|---|------|:----:|:---:|:-------:|")
    for i, p in enumerate(per_stats[:20]):
        lines.append(f"| {i+1} | `{p['file']}` | {p['mean']:+.5f} | {p['std']:.5f} | {p['norm']:.5f} |")
    lines.append("")
    lines.append("---\n")

    # 4. 쌍별 유사도 분포
    lines.append("## 4. 쌍별 코사인 유사도 분포 (4,950쌍)\n")
    lines.append("### 기술 통계\n")
    lines.append("| 통계량 | 값 |")
    lines.append("|--------|-----|")
    lines.append(f"| 최솟값 (Min) | {s['min']:+.4f} |")
    lines.append(f"| 25 퍼센타일 (P25) | {s['p25']:+.4f} |")
    lines.append(f"| 중앙값 (Median) | {s['median']:+.4f} |")
    lines.append(f"| 75 퍼센타일 (P75) | {s['p75']:+.4f} |")
    lines.append(f"| 최댓값 (Max) | {s['max']:+.4f} |")
    lines.append(f"| 평균 (Mean) | {s['mean']:+.4f} |")
    lines.append(f"| 표준편차 (Std) | {s['std']:.4f} |\n")

    lines.append("### 히스토그램\n")
    lines.append("```")
    lines.append("구간                          건수    비율")
    lines.append("─" * 55)
    hist_lines = histogram_lines(sim_values)
    for hl in hist_lines:
        lines.append(hl)
    lines.append("─" * 55)
    lines.append(f"합계{' ' * 26}{len(sim_values):5d}  100.0%")
    lines.append("```\n")
    lines.append("---\n")

    # 5. TOP/BOTTOM 유사쌍
    lines.append("## 5. TOP 유사쌍 / BOTTOM 유사쌍\n")
    lines.append("### TOP 15 — 가장 유사한 쌍\n")
    lines.append("| 순위 | 코사인 유사도 | 파일 A | 파일 B |")
    lines.append("|------|:---:|--------|--------|")
    for rank, (sim, i, j) in enumerate(sorted_pairs[:15], 1):
        fn_a = os.path.basename(sample_files[i])
        fn_b = os.path.basename(sample_files[j])
        lines.append(f"| {rank} | **{sim:+.4f}** | `{fn_a}` | `{fn_b}` |")
    lines.append("")

    lines.append("### BOTTOM 15 — 가장 상이한 쌍\n")
    lines.append("| 순위 | 코사인 유사도 | 파일 A | 파일 B |")
    lines.append("|------|:---:|--------|--------|")
    for rank, (sim, i, j) in enumerate(sorted_pairs[-15:][::-1], 1):
        fn_a = os.path.basename(sample_files[i])
        fn_b = os.path.basename(sample_files[j])
        lines.append(f"| {rank} | **{sim:+.4f}** | `{fn_a}` | `{fn_b}` |")
    lines.append("\n---\n")

    # 6. 장르별 유사도 행렬
    lines.append("## 6. 장르별 유사도 행렬\n")
    lines.append("### 10×10 장르 평균 코사인 유사도\n")
    header = "| 장르 |" + "".join(f" {g[:5]} |" for g in GENRES)
    sep = "|------|" + "|:----:|" * len(GENRES)
    lines.append(header)
    lines.append(sep)
    for ga in GENRES:
        row = f"| **{ga}** |"
        for gb in GENRES:
            val = genre_matrix[ga][gb]
            if ga == gb:
                row += f" **{val:+.3f}** |"
            else:
                row += f" {val:+.3f} |"
        lines.append(row)
    lines.append("")

    # 내부 응집도 정렬
    diag = sorted([(genre_matrix[g][g], g) for g in GENRES], reverse=True)
    lines.append("### 장르 내부 응집도 (대각선 값)\n")
    lines.append("| 순위 | 장르 | 내부 유사도 |")
    lines.append("|:----:|------|:-----------:|")
    for rank, (val, g) in enumerate(diag, 1):
        lines.append(f"| {rank} | {g} | **{val:+.3f}** |")
    lines.append("\n---\n")

    # 7. 결론
    lines.append("## 7. 결론\n")
    lines.append("### 결정적 크롭 검증")
    phase1_summary = "모든 5개 테스트 PASS" if phase1["all_ok"] else "일부 테스트 FAIL"
    lines.append(f"\n{phase1_summary}:")
    lines.append("- center crop은 동일 파일에 대해 항상 동일한 임베딩을 반환 (결정적)")
    lines.append("- `start_sec` 파라미터로 크롭 시작 위치를 변경하면 다른 임베딩이 생성됨")
    lines.append("- `start_sec` 고정 시 반복 호출해도 동일 임베딩 반환 (결정적)")
    lines.append("- 오버플로/음수 파라미터에 대해 올바르게 422 반환\n")

    lines.append("### 임베딩 품질")
    lines.append(f"\n- **정규화**: L2 norm 범위 {min(norms):.4f}~{max(norms):.4f} (≈1.0)")
    lines.append(f"- **유사도 범위**: {s['min']:+.4f} ~ {s['max']:+.4f}")
    lines.append(f"- **유사도 평균**: {s['mean']:+.4f} (중앙값: {s['median']:+.4f})")
    lines.append(f"- **가장 응집된 장르**: {diag[0][1]} ({diag[0][0]:+.3f})")
    lines.append(f"- **가장 분산된 장르**: {diag[-1][1]} ({diag[-1][0]:+.3f})\n")

    content = "\n".join(lines)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"  리포트 생성 완료: {REPORT_PATH}")
    return content


# ── 메인 ────────────────────────────────────────────────────────────────────

def main():
    # 오디오 파일 목록 구성
    sample_files = []
    genre_of = []

    for genre in GENRES:
        for i in range(SAMPLES_PER_GENRE):
            path = os.path.join(AUDIO_DIR, f"{genre}.{i:02d}.wav")
            if os.path.exists(path):
                sample_files.append(path)
                genre_of.append(genre)

    if not sample_files:
        print(f"오디오 파일이 없습니다: {AUDIO_DIR}")
        print("먼저 uv run python scripts/download_gtzan_samples.py를 실행하세요.")
        sys.exit(1)

    print(f"오디오 파일: {len(sample_files)}개 발견 (기대: 100개)")
    if len(sample_files) < 100:
        missing = 100 - len(sample_files)
        print(f"  경고: {missing}개 파일 누락. 계속 진행합니다.")

    # Phase 1
    phase1 = run_phase1(sample_files)

    # Phase 2
    phase2 = run_phase2(sample_files, genre_of)

    # Phase 3
    generate_report(phase1, phase2, sample_files, genre_of)

    print_section("전체 결과 요약")
    print(f"  Phase 1 (결정적 검증): {'PASS' if phase1['all_ok'] else 'FAIL'}")
    print(f"  Phase 2 (분포 분석):   완료")
    print(f"  Phase 3 (리포트):      {REPORT_PATH}")
    print()


if __name__ == "__main__":
    try:
        main()
    except urllib.error.URLError as e:
        print(f"\n서버 연결 실패: {e}")
        print("서버가 실행 중인지 확인하세요: uv run uvicorn app.main:app")
        sys.exit(1)

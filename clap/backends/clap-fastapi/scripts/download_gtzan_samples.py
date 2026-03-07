"""
GTZAN Genre Collection 샘플 다운로더
marsyas/gtzan 레포의 data/genres.tar.gz를 다운로드하고
장르별 10개씩 100개 오디오 파일을 resources/audio/에 WAV로 저장.

레포 구조:
  data/genres.tar.gz → genres/{genre}/{genre}.{number:05d}.au

huggingface_hub, librosa, soundfile 모두 이미 설치된 패키지.
"""

import os
import sys
import tarfile

GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock",
]
SAMPLES_PER_GENRE = 10
REPO_ID = "marsyas/gtzan"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "resources", "audio")


def main():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("huggingface_hub 패키지가 필요합니다.")
        sys.exit(1)

    try:
        import librosa
        import soundfile as sf
    except ImportError:
        print("librosa, soundfile 패키지가 필요합니다.")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 이미 100개 모두 있으면 스킵
    existing = [
        os.path.join(OUTPUT_DIR, f"{g}.{i:02d}.wav")
        for g in GENRES for i in range(SAMPLES_PER_GENRE)
    ]
    if all(os.path.exists(p) for p in existing):
        print("모든 파일이 이미 존재합니다. 스킵.")
        return

    # genres.tar.gz 다운로드
    print("data/genres.tar.gz 다운로드 중 (약 1.2GB)...")
    tgz_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="data/genres.tar.gz",
        repo_type="dataset",
    )
    print(f"다운로드 완료: {tgz_path}")

    # tar.gz에서 장르별 10개씩 추출
    total_saved = 0
    total_skipped = 0

    print("\ntar.gz 파싱 및 변환 중...")
    with tarfile.open(tgz_path, "r:gz") as tar:
        members = tar.getmembers()
        print(f"  총 멤버 수: {len(members)}")

        for genre in GENRES:
            # 해당 장르 파일 필터링 및 정렬 (macOS 메타데이터 ._* 제외)
            genre_members = sorted(
                [
                    m for m in members
                    if f"/{genre}/" in m.name
                    and m.isfile()
                    and not os.path.basename(m.name).startswith("._")
                ],
                key=lambda m: m.name,
            )

            if not genre_members:
                print(f"  [{genre}] 파일 없음 — 건너뜀")
                continue

            selected = genre_members[:SAMPLES_PER_GENRE]
            print(f"  [{genre}] {len(selected)}개 처리")

            for count, member in enumerate(selected):
                out_path = os.path.join(OUTPUT_DIR, f"{genre}.{count:02d}.wav")

                if os.path.exists(out_path):
                    total_skipped += 1
                    continue

                try:
                    f_obj = tar.extractfile(member)
                    if f_obj is None:
                        continue

                    import io
                    audio_bytes = f_obj.read()
                    audio_array, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050, mono=True)
                    sf.write(out_path, audio_array, sr)
                    duration = len(audio_array) / sr
                    print(f"    저장: {os.path.basename(out_path)} (dur={duration:.1f}s)")
                    total_saved += 1
                except Exception as e:
                    print(f"    오류: {member.name} — {e}")

    print(f"\n완료: {total_saved}개 저장, {total_skipped}개 스킵")
    print(f"저장 경로: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

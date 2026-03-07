# CLAP 텍스트 임베딩 분포 테스트 리포트

**작성일**: 2026-03-06
**모델**: `music_audioset_epoch_15_esc_90.14.pt`
**아키텍처**: HTSAT-base (`enable_fusion=False`)
**실행 스크립트**: `scripts/test_music_embeddings.py`

---

## 1. 테스트 개요

| 항목 | 값 |
|------|-----|
| 체크포인트 | `music_audioset_epoch_15_esc_90.14.pt` (2.2 GB) |
| 임베딩 차원 | 512 |
| 총 샘플 수 | 100개 텍스트 |
| 배치 크기 | 50개씩 2배치 |
| 총 쌍 수 | 4,950쌍 |

### 텍스트 카테고리 구성 (10개 카테고리 × 10개 샘플)

| 카테고리 | 예시 |
|----------|------|
| 장르 (Genre) | classical music orchestra, jazz saxophone improvisation, blues guitar, hip hop beat, EDM, country, reggae, folk, metal, R&B |
| 악기 (Instrument) | grand piano solo, violin, electric guitar, acoustic guitar, drum kit, bass guitar, cello, flute, trumpet, organ |
| 분위기 (Mood) | upbeat happy dance, sad melancholic piano, intense dramatic orchestral, relaxing ambient, romantic slow love song, energetic workout, peaceful meditation, dark mysterious, joyful children's, nostalgic vintage |
| 보컬 (Vocal) | female soprano opera, male baritone classical, choir harmony, soulful gospel, boy band pop, raspy rock vocal, whispering intimate, a cappella, autotune electronic, vocal warm-up |
| 리듬 (Rhythm) | fast tempo allegro, slow tempo adagio, syncopated jazz swing, waltz 3/4, heavy metal double bass drum, bossa nova, African drumming polyrhythm, electronic 4-on-the-floor, slow ballad, accelerating crescendo |
| 상황 (Situation) | live concert crowd, street busker, cathedral music, smoky bar jazz, outdoor festival, recording studio, lullaby for baby, wedding march, funeral music, birthday party |
| 시대 (Era) | 1950s rock and roll, 1970s disco funk, 1980s synth pop, 1990s grunge, 2000s indie pop, baroque harpsichord, Renaissance madrigal, Romantic symphony, modern minimalist, avant garde experimental |
| 지역 (Region) | Indian classical sitar, Flamenco Spanish guitar, Celtic Irish fiddle, African traditional drum, Japanese koto, Middle Eastern oud, Latin salsa, Brazilian samba, Korean gayageum, Chinese erhu |
| 음향 (Acoustic) | heavy reverb, dry close-up recorded, echo in large hall, distorted lo-fi, crystal clear hi-fi, vinyl record crackle, wind noise outdoors, old radio, underwater muffled, audience applause |
| 기타 (Misc) | music box, whistling, humming, beatboxing, music played backwards, silence between notes, musical scale ascending, dissonant chord, consonant harmony, music fading away |

---

## 2. 개별 임베딩 통계

### 전체 요약

모든 100개 임베딩은 **L2 정규화** 처리됨 (L2 norm = 1.0000).

| 통계량 | 값 |
|--------|-----|
| L2 norm (전체) | 1.0000 (균일) |
| per-vector mean 범위 | -0.0052 ~ +0.0043 |
| per-vector std 범위 | 0.0439 ~ 0.0442 |
| 차원 수 | 512 |

- **mean ≈ 0**: 임베딩 벡터가 원점 주변에 분포
- **std ≈ 0.044**: 512차원 단위구면에서 예상되는 값 (이론값: `1/√512 ≈ 0.0442`)
- 모든 샘플이 균일하게 정규화되어 있어 코사인 유사도 비교에 적합

### 주목할 이상치

`#32 male baritone classical singing`: mean = **-0.0052**, std = **0.0439** — 전체 중 가장 큰 음의 평균과 가장 낮은 분산으로, 임베딩 공간에서 구조적으로 가장 이질적인 벡터.

---

## 3. 쌍별 코사인 유사도 분포 (4,950쌍)

### 기술 통계

| 통계량 | 값 |
|--------|-----|
| 최솟값 (Min) | -0.2057 |
| 25 퍼센타일 (P25) | +0.1192 |
| 중앙값 (Median) | +0.2378 |
| 75 퍼센타일 (P75) | +0.3782 |
| 최댓값 (Max) | +0.9132 |
| 평균 (Mean) | +0.2586 |
| 표준편차 (Std) | 0.1899 |

### 히스토그램

```
구간               건수    비율
───────────────────────────────
[-0.20, -0.08)      60    1.2%  ▌
[-0.08, +0.04)     492    9.9%  ██████
[+0.04, +0.16)   1,085   21.9%  █████████████
[+0.16, +0.28)   1,280   25.9%  ████████████████  ← 최빈 구간
[+0.28, +0.40)     936   18.9%  ████████████
[+0.40, +0.52)     608   12.3%  ████████
[+0.52, +0.64)     305    6.2%  ████
[+0.64, +0.76)     130    2.6%  ██
[+0.76, +0.88)      47    0.9%  ▌
[+0.88, +1.00)       7    0.1%  ▌
───────────────────────────────
합계              4,950  100.0%
```

**분포 특성**: 양의 유사도 쪽으로 치우친 우편향(right-skewed) 분포. 대부분(약 88%)의 쌍이 0 이상의 유사도를 가지며, 이는 동일 도메인(음악) 텍스트 간 일반적인 의미 공유를 반영한다.

---

## 4. TOP 유사쌍 / BOTTOM 유사쌍

### TOP 15 — 가장 유사한 쌍

| 순위 | 코사인 유사도 | 텍스트 A | 텍스트 B |
|------|:---:|----------|----------|
| 1 | **+0.9132** | music played in a cathedral | music at a funeral |
| 2 | +0.9067 | music played in a cathedral | music with echo in large hall |
| 3 | +0.9003 | music in a recording studio | music at a funeral |
| 4 | +0.8986 | music played in a cathedral | music fading away into silence |
| 5 | +0.8922 | music played in a cathedral | music in a recording studio |
| 6 | +0.8848 | silence between musical notes | music fading away into silence |
| 7 | +0.8809 | music in a recording studio | music fading away into silence |
| 8 | +0.8765 | music at a funeral | music fading away into silence |
| 9 | +0.8640 | music with wind noise outdoors | music fading away into silence |
| 10 | +0.8609 | romantic slow love song | slow ballad music |
| 11 | +0.8477 | Japanese koto music | Chinese erhu string music |
| 12 | +0.8465 | autotune electronic vocal effect | Japanese koto music |
| 13 | +0.8275 | distorted lo-fi music | music fading away into silence |
| 14 | +0.8266 | music in a recording studio | music with echo in large hall |
| 15 | +0.8260 | electric guitar riff | Japanese koto music |

> **관찰**: TOP 쌍의 대부분이 `상황(Situation)` 또는 `음향(Acoustic)` 카테고리 내부 혹은 이 둘 사이에서 형성됨. 특히 "cathedral", "recording studio", "funeral", "fading away" 관련 텍스트들이 클러스터를 형성. "Japanese koto"가 여러 카테고리의 텍스트와 높은 유사도를 보이는 것은 주목할 점.

### BOTTOM 15 — 가장 상이한 쌍

| 순위 | 코사인 유사도 | 텍스트 A | 텍스트 B |
|------|:---:|----------|----------|
| 1 | **-0.2057** | heavy metal double bass drum | music box playing a tune |
| 2 | -0.1914 | 1970s disco funk music | consonant harmony music |
| 3 | -0.1888 | male baritone classical singing | 1980s synth pop music |
| 4 | -0.1866 | male baritone classical singing | electronic music four on the floor beat |
| 5 | -0.1849 | organ church music | 1970s disco funk music |
| 6 | -0.1842 | male baritone classical singing | music with vinyl record crackle |
| 7 | -0.1731 | electronic dance music with synthesizer | male baritone classical singing |
| 8 | -0.1504 | choir singing in harmony | 1970s disco funk music |
| 9 | -0.1477 | 2000s indie pop music | African traditional drum music |
| 10 | -0.1456 | male baritone classical singing | 2000s indie pop music |
| 11 | -0.1393 | vocal warm-up exercises | underwater muffled music |
| 12 | -0.1390 | female soprano opera singing | heavy metal double bass drum |
| 13 | -0.1388 | male baritone classical singing | crystal clear high fidelity music |
| 14 | -0.1333 | 1980s synth pop music | underwater muffled music |
| 15 | -0.1306 | male baritone classical singing | heavy metal double bass drum |

> **관찰**: `male baritone classical singing` (#32)이 BOTTOM 15 중 6회 등장 — 개별 임베딩 이상치로 확인된 결과와 일치. `1970s disco funk music`과 `organ church music` / `choir` 계열의 낮은 유사도는 시대적·음향적 특성의 극단적 차이를 반영.

---

## 5. 카테고리 내/간 평균 코사인 유사도 행렬

### 10×10 유사도 행렬

|  | 장르 | 악기 | 분위기 | 보컬 | 리듬 | 상황 | 시대 | 지역 | 음향 | 기타 |
|--|:----:|:----:|:------:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| **장르** | **0.195** | 0.234 | 0.278 | 0.172 | 0.242 | 0.289 | 0.194 | 0.203 | 0.244 | 0.226 |
| **악기** | 0.234 | **0.306** | 0.304 | 0.188 | 0.261 | 0.318 | 0.181 | 0.243 | 0.298 | 0.291 |
| **분위기** | 0.278 | 0.304 | **0.450** | 0.201 | 0.314 | 0.360 | 0.288 | 0.236 | 0.355 | 0.335 |
| **보컬** | 0.172 | 0.188 | 0.201 | **0.249** | 0.146 | 0.242 | 0.117 | 0.141 | 0.215 | 0.233 |
| **리듬** | 0.242 | 0.261 | 0.314 | 0.146 | **0.273** | 0.290 | 0.200 | 0.268 | 0.221 | 0.220 |
| **상황** | 0.289 | 0.318 | 0.360 | 0.242 | 0.290 | **0.500** | 0.211 | 0.226 | 0.509 | 0.425 |
| **시대** | 0.194 | 0.181 | 0.288 | 0.117 | 0.200 | 0.211 | **0.281** | 0.199 | 0.178 | 0.151 |
| **지역** | 0.203 | 0.243 | 0.236 | 0.141 | 0.268 | 0.226 | 0.199 | **0.355** | 0.131 | 0.132 |
| **음향** | 0.244 | 0.298 | 0.355 | 0.215 | 0.221 | 0.509 | 0.178 | 0.131 | **0.557** | 0.470 |
| **기타** | 0.226 | 0.291 | 0.335 | 0.233 | 0.220 | 0.425 | 0.151 | 0.132 | 0.470 | **0.428** |

### 카테고리 내부 유사도 (대각선)

| 카테고리 | 내부 유사도 | 해석 |
|----------|:-----------:|------|
| 음향 | **0.557** | 가장 응집된 클러스터 |
| 상황 | 0.500 | 높은 응집도 |
| 기타 | 0.428 | 중간-높음 |
| 분위기 | 0.450 | 중간-높음 |
| 지역 | 0.355 | 중간 |
| 악기 | 0.306 | 중간 |
| 시대 | 0.281 | 중간-낮음 |
| 리듬 | 0.273 | 중간-낮음 |
| 보컬 | 0.249 | 낮음 |
| 장르 | 0.195 | 가장 낮음 |

### 주목할 교차 카테고리 관계

| 카테고리 쌍 | 유사도 | 설명 |
|-------------|:------:|------|
| 상황 ↔ 음향 | 0.509 | 상황의 내부유사도(0.500)보다 높음 — 두 카테고리가 사실상 혼합 |
| 음향 ↔ 기타 | 0.470 | 강한 교차 유사도 |
| 분위기 ↔ 상황 | 0.360 | 맥락과 감정이 연결됨 |
| 보컬 ↔ 시대 | 0.117 | 전체 최저 교차 유사도 |
| 지역 ↔ 음향 | 0.131 | 지역성과 음향처리는 독립적 |

---

## 6. 결론 및 시사점

### 임베딩 품질

1. **정규화 일관성**: 모든 임베딩이 완벽한 L2=1.0 정규화를 보여 코사인 유사도 기반 검색에 바로 사용 가능.
2. **이론적 분포 일치**: per-vector std ≈ 0.0442 ≈ `1/√512`, 512차원 단위구면에서 기대되는 분포와 정확히 일치.
3. **의미 있는 구분**: 동일 도메인 텍스트 간에도 넓은 유사도 범위(-0.21 ~ +0.91)를 보여 fine-grained 검색 가능성 확인.

### 카테고리별 특성

- **음향(Acoustic) 카테고리**가 가장 강한 내부 응집도(0.557)를 보임 — CLAP 모델이 음향 처리 방식의 차이를 잘 포착
- **장르(Genre) 카테고리**가 가장 낮은 응집도(0.195) — "classical", "jazz", "hip hop"처럼 서로 다른 음악 장르 간 임베딩 분산이 큼
- **상황↔음향** 카테고리가 강하게 혼합(0.509 > 상황 내부 0.500) — 녹음 환경이 장소/상황과 밀접하게 연관

### 이상 패턴

- `male baritone classical singing`: 개별 통계 이상치이자 BOTTOM 유사도 쌍에서 반복 등장 — 모델이 이 표현을 다른 음악 텍스트와 구별되게 인코딩
- `Japanese koto music`: TOP 유사도 쌍에서 예상 밖의 타 카테고리 텍스트와 높은 유사도 — 모델의 동아시아 음악 표현 특성 확인 필요

### API 활용 권장사항

- 텍스트-텍스트 유사도 임계값: **≥ 0.50** (상위 약 2.5% 해당)를 "강한 의미 유사"로 설정 권장
- 코사인 유사도 평균(0.259)을 기준으로 임계값 조정 가능
- 카테고리 혼재 검색 시 상황/음향 카테고리 혼합에 주의

---

## 부록: 검증용 10-샘플 예비 테스트

체크포인트 로딩 확인을 위한 사전 테스트 (일반 오디오 텍스트 10개).

| 쌍 | 코사인 유사도 |
|----|:---:|
| rain falling on a rooftop ↔ thunder and lightning storm | **+0.6623** (최고) |
| birds singing in the forest ↔ a dog barking | +0.6180 |
| a dog barking ↔ piano music playing | +0.5830 |
| birds singing in the forest ↔ jazz trumpet melody | **-0.0300** (최저) |

45쌍 평균: +0.3223, 중앙값: +0.2908

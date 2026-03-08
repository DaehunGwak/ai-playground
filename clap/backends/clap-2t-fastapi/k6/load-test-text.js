import http from 'k6/http';
import { check } from 'k6';

// BASE_URL 환경변수로 오버라이드 가능 (기본값: localhost:8105)
const BASE_URL = __ENV.BASE_URL || 'http://localhost:8105';

// test_music_embeddings.py에서 검증된 100개 텍스트 쿼리
const TEXTS = [
  // 장르
  'classical music orchestra',
  'jazz saxophone improvisation',
  'blues guitar with harmonica',
  'hip hop beat with rap vocals',
  'electronic dance music with synthesizer',
  'country music with acoustic guitar',
  'reggae rhythm with bass',
  'folk music with banjo',
  'metal guitar with heavy distortion',
  'R&B smooth vocals',
  // 악기
  'grand piano solo performance',
  'violin playing a sad melody',
  'electric guitar riff',
  'acoustic guitar fingerpicking',
  'drum kit playing a fast beat',
  'bass guitar groove',
  'cello playing classical piece',
  'flute melody in the wind',
  'trumpet jazz solo',
  'organ church music',
  // 분위기/감정
  'upbeat happy dance music',
  'sad melancholic piano ballad',
  'intense dramatic orchestral music',
  'relaxing ambient background music',
  'romantic slow love song',
  'energetic workout pump up music',
  'peaceful meditation music',
  'dark mysterious atmospheric music',
  "joyful children's music",
  'nostalgic vintage music',
  // 보컬
  'female soprano opera singing',
  'male baritone classical singing',
  'choir singing in harmony',
  'soulful gospel singing',
  'boy band pop singing',
  'raspy rock vocal screaming',
  'whispering intimate vocal',
  'a cappella group vocals',
  'autotune electronic vocal effect',
  'vocal warm-up exercises',
  // 리듬/템포
  'fast tempo allegro music',
  'slow tempo adagio music',
  'syncopated rhythm jazz swing',
  'waltz three four time signature',
  'heavy metal double bass drum',
  'bossa nova Brazilian rhythm',
  'African drumming polyrhythm',
  'electronic music four on the floor beat',
  'slow ballad music',
  'accelerating music crescendo',
  // 장소/상황
  'live concert crowd cheering',
  'street busker playing guitar',
  'music played in a cathedral',
  'jazz played in a smoky bar',
  'outdoor festival music stage',
  'music in a recording studio',
  'lullaby for sleeping baby',
  'wedding ceremony music march',
  'music at a funeral',
  'birthday party celebration music',
  // 시대/스타일
  '1950s rock and roll music',
  '1970s disco funk music',
  '1980s synth pop music',
  '1990s grunge alternative rock',
  '2000s indie pop music',
  'baroque harpsichord music',
  'Renaissance madrigal music',
  'Romantic era symphony',
  'modern minimalist music',
  'avant garde experimental music',
  // 지역/문화
  'Indian classical sitar music',
  'Flamenco Spanish guitar music',
  'Celtic Irish fiddle music',
  'African traditional drum music',
  'Japanese koto music',
  'Middle Eastern oud music',
  'Latin salsa music',
  'Brazilian samba music',
  'Korean traditional gayageum',
  'Chinese erhu string music',
  // 음향 특성
  'music with heavy reverb',
  'dry close-up recorded music',
  'music with echo in large hall',
  'distorted lo-fi music',
  'crystal clear high fidelity music',
  'music with vinyl record crackle',
  'music with wind noise outdoors',
  'music played through old radio',
  'underwater muffled music',
  'music with audience applause',
  // 기타
  'music box playing a tune',
  'whistling a melody',
  'humming a soft tune',
  'beatboxing rhythmic sounds',
  'music played backwards',
  'silence between musical notes',
  'musical scale ascending',
  'dissonant chord music',
  'consonant harmony music',
  'music fading away into silence',
];

// 스트레스 테스트: semaphore=2 환경에서 30 VU로 포화 탐색
// clap-fastapi(60 VU) 대비 절반 — Triton 2-tier 구조 고려
export const options = {
  stages: [
    { duration: '10s', target: 5 },   // warm-up
    { duration: '20s', target: 15 },  // 압박 시작
    { duration: '30s', target: 30 },  // 피크 스트레스
    { duration: '20s', target: 15 },  // 회복 구간
    { duration: '10s', target: 0 },   // ramp-down
  ],
  thresholds: {
    http_req_duration: ['p(95)<5000'],  // 스트레스 테스트용 완화 임계값
    http_req_failed: ['rate<0.05'],     // 에러율 5% 이하
  },
};

export default function () {
  // 단건 요청
  const text = TEXTS[Math.floor(Math.random() * TEXTS.length)];
  const payload = JSON.stringify({ texts: [text] });

  const res = http.post(`${BASE_URL}/embed/text`, payload, {
    headers: { 'Content-Type': 'application/json' },
  });

  check(res, {
    'status 200': (r) => r.status === 200,
    'embeddings length 1': (r) => {
      const body = JSON.parse(r.body);
      return body.embeddings && body.embeddings.length === 1;
    },
    'dimension 512': (r) => {
      const body = JSON.parse(r.body);
      return body.dimension === 512;
    },
  });
}

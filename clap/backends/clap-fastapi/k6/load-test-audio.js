import http from 'k6/http';
import { check } from 'k6';

// BASE_URL 환경변수로 오버라이드 가능 (기본값: clap-fastapi.local)
const BASE_URL = __ENV.BASE_URL || 'http://clap-fastapi.local';

// 장르당 1개씩 10개 WAV — init 단계에서 로드 (~13MB)
const AUDIO_FILES = [
  { name: 'blues.00.wav',     data: open('../resources/audio/blues.00.wav', 'b') },
  { name: 'classical.00.wav', data: open('../resources/audio/classical.00.wav', 'b') },
  { name: 'country.00.wav',   data: open('../resources/audio/country.00.wav', 'b') },
  { name: 'disco.00.wav',     data: open('../resources/audio/disco.00.wav', 'b') },
  { name: 'hiphop.00.wav',    data: open('../resources/audio/hiphop.00.wav', 'b') },
  { name: 'jazz.00.wav',      data: open('../resources/audio/jazz.00.wav', 'b') },
  { name: 'metal.00.wav',     data: open('../resources/audio/metal.00.wav', 'b') },
  { name: 'pop.00.wav',       data: open('../resources/audio/pop.00.wav', 'b') },
  { name: 'reggae.00.wav',    data: open('../resources/audio/reggae.00.wav', 'b') },
  { name: 'rock.00.wav',      data: open('../resources/audio/rock.00.wav', 'b') },
];

// VU 증설 테스트: 3 → 5 → 10 → 20으로 단계 확장
export const options = {
  stages: [
    { duration: '10s', target: 5 },   // warm-up
    { duration: '20s', target: 5 },   // 5 VUs 안정 구간
    { duration: '10s', target: 10 },  // 10 VUs로 증설
    { duration: '20s', target: 10 },  // 10 VUs 안정 구간
    { duration: '10s', target: 20 },  // 20 VUs로 증설
    { duration: '20s', target: 20 },  // 20 VUs 안정 구간
    { duration: '10s', target: 0 },   // ramp-down
  ],
  thresholds: {
    http_req_duration: ['p(95)<15000'],
    http_req_failed: ['rate<0.05'],
  },
};

export default function () {
  // 10개 중 랜덤 1개 파일 선택
  const file = AUDIO_FILES[Math.floor(Math.random() * AUDIO_FILES.length)];

  const formData = {
    file: http.file(file.data, file.name, 'audio/wav'),
  };

  const res = http.post(`${BASE_URL}/embed/audio`, formData);

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

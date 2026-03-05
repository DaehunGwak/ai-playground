import http from 'k6/http';
import { check } from 'k6';

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8003';

export const options = {
  stages: [
    { duration: '10s', target: 300 },   // 빠른 램프업
    { duration: '50s', target: 300 },   // 300 VU 유지
    { duration: '15s', target: 0 },     // 쿨다운
  ],
  thresholds: {
    http_req_duration: [{ threshold: 'p(95)<500', abortOnFail: false }],
    http_req_failed: [{ threshold: 'rate<0.1', abortOnFail: false }],
  },
};

export default function () {
  check(http.get(`${BASE_URL}/heavy`), {
    'status 200': (r) => r.status === 200,
  });
  // sleep 없음 — 최대 부하
}

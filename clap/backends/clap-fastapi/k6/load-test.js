import http from 'k6/http';
import { check, sleep } from 'k6';

// BASE_URL 환경변수로 오버라이드 가능 (기본값: clap-fastapi.local)
const BASE_URL = __ENV.BASE_URL || 'http://clap-fastapi.local';

export const options = {
  stages: [
    { duration: '10s', target: 20 },   // ramp-up
    { duration: '30s', target: 50 },   // sustained load
    { duration: '10s', target: 0 },    // ramp-down
  ],
  thresholds: {
    http_req_duration: ['p(95)<2000'],  // 95th percentile < 2000ms (모델 추론 포함)
    http_req_failed: ['rate<0.01'],     // 에러율 < 1%
  },
};

export default function () {
  check(http.get(`${BASE_URL}/health`), {
    'health status 200': (r) => r.status === 200,
  });

  check(http.get(`${BASE_URL}/echo/hello`), {
    'echo status 200': (r) => r.status === 200,
  });

  check(http.get(`${BASE_URL}/heavy`), {
    'heavy status 200': (r) => r.status === 200,
  });

  const embedPayload = JSON.stringify({ texts: ['a dog barking', 'piano music'] });
  check(
    http.post(`${BASE_URL}/embed/text`, embedPayload, {
      headers: { 'Content-Type': 'application/json' },
    }),
    {
      'embed/text status 200': (r) => r.status === 200,
      'embed/text has embeddings': (r) => {
        const body = JSON.parse(r.body);
        return body.embeddings && body.embeddings.length === 2 && body.dimension === 512;
      },
    }
  );

  sleep(0.5);
}

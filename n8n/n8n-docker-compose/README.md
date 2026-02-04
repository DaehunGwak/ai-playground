# n8n docker compose

## steps

1. `.env` 파일 환경 변수들 적절하게 변경
2. 시작: `docker-compose up -d`
3. 종료: `docker-compose stop`

## Ollama 연동

Docker 컨테이너에서 로컬 Ollama를 사용하려면 추가 설정이 필요합니다.

### 빠른 설정 (자동화 스크립트)

```bash
# 실행 권한 부여 (최초 1회)
chmod +x update-ollama-host.sh

# 스크립트 실행
./update-ollama-host.sh
```

스크립트가 자동으로:
- 현재 호스트 IP를 감지
- `.env` 파일 생성/업데이트
- Ollama 재시작
- Docker 컨테이너 재시작
- 연결 테스트

### 수동 설정

1. **Ollama를 0.0.0.0에서 리스닝하도록 설정:**
```bash
pkill -9 ollama
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

2. **n8n에서 Ollama 노드 설정:**
```
Base URL: http://host.docker.internal:11434
```

### 문제 해결

연결 문제가 발생하면 [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) 문서를 참고하세요.

주요 내용:
- 네트워크 진단 방법
- 여러 가지 해결 방법 (launchd, IP 직접 사용 등)
- 자주 발생하는 문제와 해결책
- 보안 고려사항

## References

- https://docs.n8n.io/hosting/installation/docker/#using-with-postgresql
- https://github.com/n8n-io/n8n-hosting/tree/main/docker-compose
- https://github.com/n8n-io/n8n-hosting/tree/main/docker-compose/withPostgres

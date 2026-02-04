# 🔧 트러블슈팅 가이드

## Docker 컨테이너에서 로컬 Ollama 접속 문제

### 📋 문제 증상

- `curl http://localhost:11434` ✅ 작동함
- Docker 컨테이너 내에서 `http://host.docker.internal:11434` ❌ Connection Refused

### 🔍 근본 원인

Ollama가 기본적으로 `127.0.0.1:11434`에서만 리스닝하기 때문에, Docker 컨테이너에서 `host.docker.internal` (실제로는 `172.17.0.1` 등의 Docker 게이트웨이 IP)을 통해 접근할 수 없습니다.

**왜 이런 일이 발생하나요?**
- `localhost`나 `127.0.0.1`은 호스트 머신 내부에서만 접근 가능한 루프백 주소입니다
- Docker 컨테이너는 별도의 네트워크 네임스페이스에서 실행되므로, 호스트의 `127.0.0.1`에 직접 접근할 수 없습니다
- `host.docker.internal`은 Docker가 제공하는 특수 DNS 이름으로, 호스트 머신의 실제 네트워크 인터페이스를 가리킵니다
- Ollama가 `0.0.0.0`에서 리스닝해야 모든 네트워크 인터페이스에서 접근 가능합니다

---

## ✅ 해결 방법

### 1단계: 현재 상태 진단

Ollama가 어느 주소에서 리스닝하는지 확인합니다:

```bash
lsof -i :11434
```

**출력 해석:**
- ❌ `127.0.0.1:11434` → 문제 있음 (로컬에서만 접근 가능)
- ✅ `*:11434` 또는 `0.0.0.0:11434` → 정상 (모든 인터페이스에서 접근 가능)

대안 명령어:
```bash
netstat -an | grep 11434
# 또는
sudo lsof -iTCP:11434 -sTCP:LISTEN
```

---

### 2단계: Ollama 설정 변경

#### 🔹 방법 A: 임시 실행 (테스트용)

가장 빠르게 테스트할 수 있는 방법입니다:

```bash
# 1. 모든 Ollama 프로세스 종료
pkill -9 ollama

# 2. 잠시 대기 (프로세스가 완전히 종료될 때까지)
sleep 2

# 3. 환경 변수와 함께 Ollama 실행 (이 터미널은 열어두어야 함)
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

⚠️ **주의**: 이 터미널을 닫으면 Ollama도 종료됩니다.

---

#### 🔹 방법 B: launchd 서비스 설정 (영구 설정 - 추천)

Mac에서 Ollama를 백그라운드 서비스로 실행하는 방법입니다:

```bash
# 1. 기존 Ollama 프로세스 완전 종료
pkill -9 ollama

# 2. launchd 설정 파일 생성
mkdir -p ~/Library/LaunchAgents

cat > ~/Library/LaunchAgents/com.ollama.plist << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.ollama</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/local/bin/ollama</string>
        <string>serve</string>
    </array>
    <key>EnvironmentVariables</key>
    <dict>
        <key>OLLAMA_HOST</key>
        <string>0.0.0.0:11434</string>
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/tmp/ollama.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/ollama.error.log</string>
</dict>
</plist>
EOF

# 3. launchd 서비스 로드
launchctl unload ~/Library/LaunchAgents/com.ollama.plist 2>/dev/null
launchctl load ~/Library/LaunchAgents/com.ollama.plist

# 4. 서비스 상태 확인
launchctl list | grep ollama
```

**서비스 관리 명령어:**
```bash
# 서비스 중지
launchctl unload ~/Library/LaunchAgents/com.ollama.plist

# 서비스 시작
launchctl load ~/Library/LaunchAgents/com.ollama.plist

# 서비스 재시작
launchctl unload ~/Library/LaunchAgents/com.ollama.plist
launchctl load ~/Library/LaunchAgents/com.ollama.plist

# 로그 확인
tail -f /tmp/ollama.log
tail -f /tmp/ollama.error.log
```

---

#### 🔹 방법 C: .zshrc 환경 변수 설정

셸 환경 변수로 설정하는 방법입니다 (Ollama를 직접 실행할 때 유용):

```bash
# 1. .zshrc에 환경 변수 추가
echo 'export OLLAMA_HOST=0.0.0.0:11434' >> ~/.zshrc

# 2. 현재 셸에 적용
source ~/.zshrc

# 3. Ollama 재시작
pkill -9 ollama
sleep 2
ollama serve
```

---

#### 🔹 방법 D: 호스트 IP 직접 사용 (최후의 방법)

`host.docker.internal`이 작동하지 않거나 `0.0.0.0` 설정이 불가능한 경우, 호스트의 실제 IP 주소를 사용하는 방법입니다.

**⚠️ 이 방법은 다음과 같은 경우에 사용하세요:**
- `host.docker.internal`이 제대로 작동하지 않을 때
- Docker Desktop for Mac이 아닌 다른 Docker 환경을 사용할 때
- 보안상의 이유로 `0.0.0.0` 바인딩을 피하고 싶을 때
- 네트워크가 변경될 때마다 재설정이 필요함 (Wi-Fi → 이더넷 전환 시)

##### Step 1: 호스트 IP 주소 확인

여러 방법으로 현재 사용 중인 네트워크의 IP 주소를 확인할 수 있습니다:

**방법 1: ifconfig 사용 (가장 확실한 방법)**

```bash
# Wi-Fi 사용 시
ipconfig getifaddr en0

# 이더넷 사용 시
ipconfig getifaddr en1

# 모든 IP 주소 확인
ifconfig | grep "inet " | grep -v 127.0.0.1

# 더 깔끔한 출력
ifconfig | grep "inet " | grep -v 127.0.0.1 | awk '{print $2}'
```

**방법 2: 시스템 환경설정에서 확인**

```bash
# 시스템 환경설정을 명령어로 열기
open "x-apple.systempreferences:com.apple.preference.network"

# 또는 간단하게
ipconfig getifaddr en0 || ipconfig getifaddr en1
```

**방법 3: 현재 활성 네트워크 인터페이스 자동 감지**

```bash
# 활성 네트워크 인터페이스의 IP 가져오기
DEFAULT_ROUTE=$(route -n get default | grep interface | awk '{print $2}')
HOST_IP=$(ipconfig getifaddr $DEFAULT_ROUTE)
echo "현재 호스트 IP: $HOST_IP"
```

**출력 예시:**
```
192.168.1.100  # 또는
10.0.0.50      # 또는
172.16.0.10    # 등
```

##### Step 2: Ollama를 특정 IP로 바인딩

**옵션 A: 특정 IP로만 바인딩 (더 안전)**

```bash
# 1. 호스트 IP 확인 (예: 192.168.1.100)
HOST_IP=$(ipconfig getifaddr en0 || ipconfig getifaddr en1)
echo "호스트 IP: $HOST_IP"

# 2. Ollama 종료
pkill -9 ollama
sleep 2

# 3. 특정 IP로 Ollama 실행
OLLAMA_HOST=${HOST_IP}:11434 ollama serve
```

**옵션 B: 여러 인터페이스 동시 바인딩 (유연성)**

만약 Wi-Fi와 이더넷을 모두 사용하거나 여러 네트워크를 전환하는 경우:

```bash
# 0.0.0.0 사용 (모든 인터페이스)
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

##### Step 3: Docker Compose에 고정 IP 설정

docker-compose.yml을 수정하여 환경 변수로 IP를 관리합니다:

**1. docker-compose.yml 파일 수정**

```yaml
services:
  n8n:
    image: docker.n8n.io/n8nio/n8n
    restart: always
    environment:
      - DB_TYPE=postgresdb
      - DB_POSTGRESDB_HOST=postgres
      - DB_POSTGRESDB_PORT=5432
      - DB_POSTGRESDB_DATABASE=${POSTGRES_DB}
      - DB_POSTGRESDB_USER=${POSTGRES_NON_ROOT_USER}
      - DB_POSTGRESDB_PASSWORD=${POSTGRES_NON_ROOT_PASSWORD}
      - WEBHOOK_URL=http://127.0.0.1:5678/
      - N8N_HOST=127.0.0.1
      - N8N_PORT=5678
      - N8N_PROTOCOL=http
      # Ollama 호스트 IP를 환경 변수로 설정
      - OLLAMA_HOST=${OLLAMA_HOST:-host.docker.internal}
    ports:
      - 5678:5678
    links:
      - postgres
    volumes:
      - n8n_storage:/home/node/.n8n
    depends_on:
      postgres:
        condition: service_healthy
    extra_hosts:
      - "host.docker.internal:host-gateway"
      # 호스트 IP를 직접 추가 (선택 사항)
      - "ollama.host:${HOST_IP:-172.17.0.1}"
```

**2. .env 파일 생성**

```bash
# 호스트 IP를 .env 파일에 저장
cat > .env << EOF
# 기존 데이터베이스 설정
POSTGRES_USER=changeme
POSTGRES_PASSWORD=changeme
POSTGRES_DB=n8n
POSTGRES_NON_ROOT_USER=n8n
POSTGRES_NON_ROOT_PASSWORD=changeme

# Ollama 설정
HOST_IP=$(ipconfig getifaddr en0 || ipconfig getifaddr en1 || echo "192.168.1.100")
OLLAMA_HOST=${HOST_IP}
EOF

# .env 파일 확인
cat .env
```

##### Step 4: 연결 테스트

```bash
# 1. 호스트 IP 확인
HOST_IP=$(ipconfig getifaddr en0 || ipconfig getifaddr en1)
echo "테스트할 IP: $HOST_IP"

# 2. 로컬에서 테스트
curl http://${HOST_IP}:11434/api/tags

# 3. Docker 컨테이너 재시작
docker-compose down
docker-compose up -d

# 4. Docker 컨테이너에서 테스트
docker exec -it n8n-docker-compose-n8n-1 wget -qO- http://${HOST_IP}:11434/api/tags

# 또는 extra_hosts에 추가한 경우
docker exec -it n8n-docker-compose-n8n-1 wget -qO- http://ollama.host:11434/api/tags
```

##### Step 5: n8n에서 사용

n8n의 Ollama 노드 설정:

**옵션 1: 직접 IP 사용**
```
Base URL: http://192.168.1.100:11434
```
(실제 IP로 교체)

**옵션 2: extra_hosts 사용**
```
Base URL: http://ollama.host:11434
```

##### 🔄 네트워크 변경 시 자동 업데이트 스크립트

네트워크가 자주 변경되는 경우 자동화 스크립트를 만들 수 있습니다:

**update-ollama-host.sh 생성:**

```bash
cat > update-ollama-host.sh << 'SCRIPT'
#!/bin/bash

# 현재 활성 네트워크 인터페이스의 IP 가져오기
DEFAULT_ROUTE=$(route -n get default 2>/dev/null | grep interface | awk '{print $2}')
if [ -z "$DEFAULT_ROUTE" ]; then
    echo "❌ 활성 네트워크 인터페이스를 찾을 수 없습니다."
    exit 1
fi

HOST_IP=$(ipconfig getifaddr $DEFAULT_ROUTE)
if [ -z "$HOST_IP" ]; then
    echo "❌ IP 주소를 가져올 수 없습니다."
    exit 1
fi

echo "✅ 현재 호스트 IP: $HOST_IP"

# .env 파일 업데이트
if [ -f .env ]; then
    # 기존 HOST_IP 라인이 있으면 업데이트
    if grep -q "^HOST_IP=" .env; then
        sed -i '' "s/^HOST_IP=.*/HOST_IP=${HOST_IP}/" .env
        echo "✅ .env 파일 업데이트 완료"
    else
        echo "HOST_IP=${HOST_IP}" >> .env
        echo "✅ .env 파일에 HOST_IP 추가 완료"
    fi
    
    # OLLAMA_HOST도 업데이트
    if grep -q "^OLLAMA_HOST=" .env; then
        sed -i '' "s/^OLLAMA_HOST=.*/OLLAMA_HOST=${HOST_IP}/" .env
    else
        echo "OLLAMA_HOST=${HOST_IP}" >> .env
    fi
else
    echo "⚠️  .env 파일이 없습니다. 생성합니다..."
    cat > .env << EOF
HOST_IP=${HOST_IP}
OLLAMA_HOST=${HOST_IP}
EOF
fi

# Ollama 재시작
echo "🔄 Ollama 재시작 중..."
pkill -9 ollama
sleep 2
OLLAMA_HOST=${HOST_IP}:11434 ollama serve &
echo "✅ Ollama가 ${HOST_IP}:11434에서 실행 중입니다."

# Docker 컨테이너 재시작 (선택 사항)
read -p "Docker 컨테이너를 재시작하시겠습니까? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🔄 Docker 컨테이너 재시작 중..."
    docker-compose restart n8n
    echo "✅ 완료!"
fi

echo ""
echo "📝 n8n에서 다음 URL을 사용하세요:"
echo "   Base URL: http://${HOST_IP}:11434"
SCRIPT

# 실행 권한 부여
chmod +x update-ollama-host.sh

# 실행
./update-ollama-host.sh
```

**사용 방법:**
```bash
# 네트워크 변경 시 실행
./update-ollama-host.sh
```

##### 📱 자동 감지 및 알림 (선택 사항)

네트워크 변경을 자동으로 감지하는 macOS LaunchAgent:

```bash
# 네트워크 변경 감지 스크립트 생성
cat > ~/watch-network-change.sh << 'SCRIPT'
#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}/dev/personal/ai-playground/n8n/n8n-docker-compose"

OLD_IP=""
while true; do
    CURRENT_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null)
    
    if [ -n "$CURRENT_IP" ] && [ "$CURRENT_IP" != "$OLD_IP" ]; then
        echo "[$(date)] 네트워크 변경 감지: $OLD_IP -> $CURRENT_IP"
        
        # update 스크립트가 있으면 실행
        if [ -f "./update-ollama-host.sh" ]; then
            ./update-ollama-host.sh
        fi
        
        OLD_IP=$CURRENT_IP
    fi
    
    sleep 30  # 30초마다 체크
done
SCRIPT

chmod +x ~/watch-network-change.sh
```

##### ⚠️ 주의사항

1. **IP 변경 문제:**
   - DHCP를 사용하는 경우 IP가 변경될 수 있습니다
   - 네트워크 변경 시 (Wi-Fi ↔ 이더넷) IP가 달라집니다
   - 해결: 고정 IP를 설정하거나 위의 자동화 스크립트를 사용하세요

2. **보안:**
   - 특정 IP로 바인딩하면 해당 네트워크에서만 접근 가능합니다
   - 공용 Wi-Fi에서는 다른 사용자가 접근할 수 있으니 주의하세요
   - 방화벽 규칙을 설정하는 것을 권장합니다

3. **방화벽 설정:**
```bash
# macOS 방화벽에서 Ollama 허용
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add /usr/local/bin/ollama
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --unblock /usr/local/bin/ollama
```

##### ✅ 장단점 비교

| 방법 | 장점 | 단점 |
|------|------|------|
| `0.0.0.0` | 모든 네트워크에서 작동, 자동 적응 | 보안 위험 증가 |
| `host.docker.internal` | 표준 방법, 간단 | macOS에서 가끔 문제 발생 |
| **특정 IP** | 보안성 향상, 명확한 제어 | IP 변경 시 재설정 필요 |

---

### 3단계: 연결 테스트

#### 📍 테스트 1: 리스닝 상태 재확인

```bash
lsof -i :11434
```

다음과 같이 출력되어야 합니다:
```
COMMAND   PID        USER   FD   TYPE             DEVICE SIZE/OFF NODE NAME
ollama  12345 tt    8u  IPv4 0x123456789abcdef0      0t0  TCP *:11434 (LISTEN)
```

`*:11434` 또는 `0.0.0.0:11434`로 표시되면 성공입니다.

---

#### 📍 테스트 2: 로컬 호스트에서 접근

```bash
curl http://localhost:11434/api/tags
```

예상 출력: JSON 형식의 모델 리스트

---

#### 📍 테스트 3: Docker 컨테이너에서 접근

```bash
# n8n 컨테이너가 실행 중인지 확인
docker ps | grep n8n

# 컨테이너 내부에서 Ollama API 호출
docker exec -it n8n-docker-compose-n8n-1 wget -qO- http://host.docker.internal:11434/api/tags
```

✅ **성공**: JSON 형식의 모델 리스트가 반환됨
❌ **실패**: `Connection refused` 에러

---

#### 📍 테스트 4: curl을 사용한 테스트 (wget이 없는 경우)

```bash
docker exec -it n8n-docker-compose-n8n-1 sh -c "command -v curl && curl http://host.docker.internal:11434/api/tags"
```

---

### 4단계: Docker 컨테이너 재시작

설정 변경 후 n8n 컨테이너를 재시작합니다:

```bash
cd /Users/gwagdaehun/dev/personal/ai-playground/n8n/n8n-docker-compose
docker-compose restart n8n

# 또는 전체 재시작
docker-compose down
docker-compose up -d
```

---

### 5단계: n8n에서 Ollama 사용

n8n의 Ollama 노드 설정에서 다음 URL을 사용합니다:

```
Base URL: http://host.docker.internal:11434
```

**주의사항:**
- `localhost` ❌ 사용 불가
- `127.0.0.1` ❌ 사용 불가
- `host.docker.internal` ✅ 사용해야 함

---

## 🔧 추가 디버깅 방법

### 방화벽 확인

macOS 방화벽이 Ollama 연결을 차단하고 있을 수 있습니다:

```bash
# 방화벽 상태 확인
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --getglobalstate

# 방화벽 규칙 확인
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --listapps | grep ollama
```

**해결 방법:**
1. 시스템 환경설정 → 보안 및 개인정보 보호 → 방화벽
2. 방화벽 옵션 클릭
3. Ollama 앱 추가 또는 허용

---

### 호스트 IP 직접 사용

`host.docker.internal`이 작동하지 않는 경우, 호스트의 실제 IP를 사용할 수 있습니다:

```bash
# 호스트의 실제 IP 주소 확인 (Wi-Fi)
ipconfig getifaddr en0

# 또는 (이더넷)
ipconfig getifaddr en1

# 모든 네트워크 인터페이스 확인
ifconfig | grep "inet " | grep -v 127.0.0.1
```

예를 들어 IP가 `192.168.1.100`이라면:
```bash
# Docker 컨테이너에서 테스트
docker exec -it n8n-docker-compose-n8n-1 wget -qO- http://192.168.1.100:11434/api/tags

# n8n 설정
Base URL: http://192.168.1.100:11434
```

---

### Docker Compose 설정 확인

`docker-compose.yml` 파일에 `extra_hosts`가 올바르게 설정되어 있는지 확인:

```yaml
services:
  n8n:
    # ... 기타 설정 ...
    extra_hosts:
      - "host.docker.internal:host-gateway"  # 이 줄이 있어야 함
```

---

### 네트워크 연결 테스트

```bash
# Docker 컨테이너의 네트워크 설정 확인
docker exec -it n8n-docker-compose-n8n-1 cat /etc/hosts

# host.docker.internal이 올바른 IP로 매핑되어 있는지 확인
# 출력 예시:
# 172.17.0.1      host.docker.internal

# ping 테스트
docker exec -it n8n-docker-compose-n8n-1 ping -c 3 host.docker.internal

# 포트 연결 테스트
docker exec -it n8n-docker-compose-n8n-1 nc -zv host.docker.internal 11434
```

---

### Ollama 프로세스 확인

```bash
# Ollama 프로세스가 실행 중인지 확인
ps aux | grep ollama

# Ollama가 사용하는 포트 확인
lsof -p $(pgrep ollama)

# Ollama 버전 확인
ollama --version
```

---

## 🚨 자주 발생하는 문제

### 문제 1: "Connection refused" 지속

**원인:** Ollama가 여전히 `127.0.0.1`에서만 리스닝
**해결:** 
```bash
# 확인
lsof -i :11434

# 모든 Ollama 프로세스 강제 종료
pkill -9 ollama
pgrep ollama  # 출력이 없어야 함

# 환경 변수 확인
echo $OLLAMA_HOST

# 재시작
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

---

### 문제 2: "Address already in use"

**원인:** 이미 다른 Ollama 프로세스가 실행 중
**해결:**
```bash
# 포트 사용 중인 프로세스 찾기
lsof -i :11434

# 해당 프로세스 종료
kill -9 <PID>

# 또는 모든 Ollama 프로세스 종료
pkill -9 ollama
```

---

### 문제 3: Docker 컨테이너에서 "wget: not found"

**원인:** n8n 이미지에 wget이 없을 수 있음
**해결:**
```bash
# curl 사용
docker exec -it n8n-docker-compose-n8n-1 sh -c "curl http://host.docker.internal:11434/api/tags"

# 또는 nc(netcat) 사용
docker exec -it n8n-docker-compose-n8n-1 nc -zv host.docker.internal 11434
```

---

### 문제 4: launchd 서비스가 시작되지 않음

**확인:**
```bash
# 서비스 상태 확인
launchctl list | grep ollama

# 로그 확인
cat /tmp/ollama.log
cat /tmp/ollama.error.log

# Ollama 실행 파일 경로 확인
which ollama

# plist 파일의 경로가 올바른지 확인
cat ~/Library/LaunchAgents/com.ollama.plist
```

**Ollama 경로가 다른 경우:**
```bash
# Homebrew로 설치한 경우
/opt/homebrew/bin/ollama

# 또는
/usr/local/bin/ollama
```

plist 파일의 `<string>/usr/local/bin/ollama</string>`를 실제 경로로 수정하세요.

---

## 🔐 보안 고려사항

### ⚠️ 주의: 0.0.0.0 리스닝의 의미

`OLLAMA_HOST=0.0.0.0:11434`로 설정하면 Ollama가 **모든 네트워크 인터페이스**에서 접근 가능해집니다.

**이것이 의미하는 것:**
- ✅ 로컬 Docker 컨테이너에서 접근 가능
- ⚠️ 같은 네트워크의 다른 기기에서도 접근 가능
- 🚨 공용 네트워크에서는 인터넷을 통해 접근 가능할 수 있음

### 보안 강화 방법

#### 1. 방화벽 규칙 설정

```bash
# macOS 방화벽 활성화
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --setglobalstate on

# 특정 앱에 대한 연결만 허용
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add /usr/local/bin/ollama
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --blockapp /usr/local/bin/ollama
```

#### 2. 특정 네트워크 인터페이스만 바인딩

Docker 게이트웨이 IP만 허용하고 싶다면:

```bash
# Docker 네트워크의 게이트웨이 IP 확인
docker network inspect bridge | grep Gateway

# 예: 172.17.0.1
OLLAMA_HOST=172.17.0.1:11434 ollama serve
```

#### 3. 리버스 프록시 사용

nginx 등을 사용하여 접근 제어를 추가할 수 있습니다.

---

## 📚 참고 자료

### Ollama 환경 변수

- `OLLAMA_HOST`: Ollama 서버가 바인딩할 주소 (기본값: `127.0.0.1:11434`)
- `OLLAMA_MODELS`: 모델 저장 경로
- `OLLAMA_ORIGINS`: CORS 허용 origin 설정

### Docker 네트워킹

- `host.docker.internal`: macOS/Windows Docker에서 호스트를 가리키는 특수 DNS
- `extra_hosts`: 컨테이너의 `/etc/hosts`에 사용자 정의 호스트 추가
- `host-gateway`: Docker가 자동으로 호스트 게이트웨이 IP로 해석

---

## ✅ 체크리스트

문제 해결 시 다음 사항을 순서대로 확인하세요:

- [ ] Ollama가 실행 중인가? (`ps aux | grep ollama`)
- [ ] Ollama가 0.0.0.0에서 리스닝하는가? (`lsof -i :11434`)
- [ ] 로컬에서 접근 가능한가? (`curl http://localhost:11434/api/tags`)
- [ ] Docker 컨테이너가 실행 중인가? (`docker ps`)
- [ ] `host.docker.internal`이 올바르게 설정되었나? (`docker exec ... cat /etc/hosts`)
- [ ] 컨테이너에서 호스트로 네트워크 연결이 가능한가? (`docker exec ... ping host.docker.internal`)
- [ ] 컨테이너에서 Ollama 포트 접근이 가능한가? (`docker exec ... nc -zv host.docker.internal 11434`)
- [ ] 방화벽이 연결을 차단하지 않는가?

---

## 💡 빠른 해결 요약

```bash
# 1. Ollama 종료
pkill -9 ollama

# 2. 0.0.0.0에서 리스닝하도록 재시작
OLLAMA_HOST=0.0.0.0:11434 ollama serve &

# 3. 확인
lsof -i :11434

# 4. Docker 컨테이너 재시작
docker-compose restart n8n

# 5. 테스트
docker exec -it n8n-docker-compose-n8n-1 wget -qO- http://host.docker.internal:11434/api/tags

# 6. n8n에서 사용
# Base URL: http://host.docker.internal:11434
```

---

**문제가 해결되지 않으면 다음 정보와 함께 이슈를 제기하세요:**
- `lsof -i :11434` 출력
- `docker exec -it n8n-docker-compose-n8n-1 cat /etc/hosts` 출력
- `docker network inspect bridge` 출력
- Ollama 버전 (`ollama --version`)
- macOS 버전

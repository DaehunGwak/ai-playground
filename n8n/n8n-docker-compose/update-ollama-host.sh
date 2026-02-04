#!/bin/bash

# Ollama 호스트 IP 자동 업데이트 스크립트
# Docker 컨테이너에서 로컬 Ollama에 접속하기 위한 호스트 IP를 자동으로 감지하고 설정합니다.

set -e

echo "🔍 호스트 IP 주소 확인 중..."

# 현재 활성 네트워크 인터페이스의 IP 가져오기
DEFAULT_ROUTE=$(route -n get default 2>/dev/null | grep interface | awk '{print $2}')
if [ -z "$DEFAULT_ROUTE" ]; then
    echo "❌ 활성 네트워크 인터페이스를 찾을 수 없습니다."
    echo "   네트워크에 연결되어 있는지 확인해주세요."
    exit 1
fi

HOST_IP=$(ipconfig getifaddr $DEFAULT_ROUTE)
if [ -z "$HOST_IP" ]; then
    echo "❌ IP 주소를 가져올 수 없습니다."
    echo "   인터페이스: $DEFAULT_ROUTE"
    exit 1
fi

echo "✅ 현재 호스트 IP: $HOST_IP (인터페이스: $DEFAULT_ROUTE)"

# 스크립트 디렉토리로 이동
cd "$(dirname "$0")"

# .env 파일 업데이트
echo ""
echo "📝 .env 파일 업데이트 중..."

if [ -f .env ]; then
    # 기존 HOST_IP 라인이 있으면 업데이트
    if grep -q "^HOST_IP=" .env; then
        if [[ "$OSTYPE" == "darwin"* ]]; then
            sed -i '' "s/^HOST_IP=.*/HOST_IP=${HOST_IP}/" .env
        else
            sed -i "s/^HOST_IP=.*/HOST_IP=${HOST_IP}/" .env
        fi
        echo "✅ .env 파일의 HOST_IP 업데이트 완료"
    else
        echo "HOST_IP=${HOST_IP}" >> .env
        echo "✅ .env 파일에 HOST_IP 추가 완료"
    fi
    
    # OLLAMA_HOST도 업데이트
    if grep -q "^OLLAMA_HOST=" .env; then
        if [[ "$OSTYPE" == "darwin"* ]]; then
            sed -i '' "s/^OLLAMA_HOST=.*/OLLAMA_HOST=${HOST_IP}/" .env
        else
            sed -i "s/^OLLAMA_HOST=.*/OLLAMA_HOST=${HOST_IP}/" .env
        fi
    else
        echo "OLLAMA_HOST=${HOST_IP}" >> .env
    fi
else
    echo "⚠️  .env 파일이 없습니다. 생성합니다..."
    cat > .env << EOF
# Ollama Host Configuration
HOST_IP=${HOST_IP}
OLLAMA_HOST=${HOST_IP}

# PostgreSQL Configuration (필요 시 설정)
# POSTGRES_USER=changeme
# POSTGRES_PASSWORD=changeme
# POSTGRES_DB=n8n
# POSTGRES_NON_ROOT_USER=n8n
# POSTGRES_NON_ROOT_PASSWORD=changeme
EOF
    echo "✅ .env 파일 생성 완료"
fi

# 현재 Ollama 상태 확인
echo ""
echo "🔍 Ollama 상태 확인 중..."

OLLAMA_PID=$(pgrep ollama || echo "")
if [ -n "$OLLAMA_PID" ]; then
    echo "✅ Ollama가 실행 중입니다 (PID: $OLLAMA_PID)"
    
    # 현재 리스닝 주소 확인
    LISTENING=$(lsof -i :11434 -sTCP:LISTEN 2>/dev/null | tail -n +2 || echo "")
    if [ -n "$LISTENING" ]; then
        echo "   리스닝 주소:"
        echo "$LISTENING" | awk '{print "   - " $9}'
    fi
    
    # Ollama 재시작 여부 묻기
    echo ""
    read -p "Ollama를 재시작하시겠습니까? (y/n): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🔄 Ollama 재시작 중..."
        pkill -9 ollama
        sleep 2
        
        # Ollama를 백그라운드로 실행
        OLLAMA_HOST=${HOST_IP}:11434 ollama serve > /tmp/ollama.log 2>&1 &
        sleep 2
        
        # 재시작 확인
        if pgrep ollama > /dev/null; then
            echo "✅ Ollama가 ${HOST_IP}:11434에서 실행 중입니다."
        else
            echo "❌ Ollama 재시작 실패. 로그를 확인하세요:"
            echo "   tail -f /tmp/ollama.log"
            exit 1
        fi
    fi
else
    echo "⚠️  Ollama가 실행되고 있지 않습니다."
    
    read -p "Ollama를 시작하시겠습니까? (y/n): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🚀 Ollama 시작 중..."
        OLLAMA_HOST=${HOST_IP}:11434 ollama serve > /tmp/ollama.log 2>&1 &
        sleep 2
        
        if pgrep ollama > /dev/null; then
            echo "✅ Ollama가 ${HOST_IP}:11434에서 실행 중입니다."
        else
            echo "❌ Ollama 시작 실패. 로그를 확인하세요:"
            echo "   tail -f /tmp/ollama.log"
            exit 1
        fi
    fi
fi

# 연결 테스트
echo ""
echo "🔗 연결 테스트 중..."

if curl -s --max-time 5 http://${HOST_IP}:11434/api/tags > /dev/null 2>&1; then
    echo "✅ 로컬에서 Ollama 접속 성공"
else
    echo "❌ 로컬에서 Ollama 접속 실패"
    echo "   URL: http://${HOST_IP}:11434"
    echo "   다음을 확인해주세요:"
    echo "   1. Ollama가 실행 중인지"
    echo "   2. 방화벽 설정"
    echo "   3. Ollama 로그: tail -f /tmp/ollama.log"
fi

# Docker 컨테이너 재시작 (선택 사항)
if [ -f docker-compose.yml ]; then
    echo ""
    read -p "Docker 컨테이너를 재시작하시겠습니까? (y/n): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🔄 Docker 컨테이너 재시작 중..."
        
        if docker-compose restart n8n 2>/dev/null; then
            echo "✅ n8n 컨테이너 재시작 완료!"
            
            # Docker 컨테이너에서 연결 테스트
            echo ""
            echo "🔗 Docker 컨테이너에서 연결 테스트 중..."
            sleep 3
            
            if docker exec -it n8n-docker-compose-n8n-1 wget -qO- http://${HOST_IP}:11434/api/tags > /dev/null 2>&1; then
                echo "✅ Docker 컨테이너에서 Ollama 접속 성공!"
            else
                echo "⚠️  Docker 컨테이너에서 Ollama 접속 실패"
                echo "   다음 명령어로 수동 테스트:"
                echo "   docker exec -it n8n-docker-compose-n8n-1 wget -qO- http://${HOST_IP}:11434/api/tags"
            fi
        else
            echo "⚠️  컨테이너 재시작 실패 또는 컨테이너가 실행 중이지 않습니다."
        fi
    fi
fi

# 최종 안내
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ 설정 완료!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📝 n8n Ollama 노드 설정:"
echo "   Base URL: http://${HOST_IP}:11434"
echo ""
echo "🔧 유용한 명령어:"
echo "   • Ollama 상태 확인: lsof -i :11434"
echo "   • Ollama 로그 확인: tail -f /tmp/ollama.log"
echo "   • 로컬 테스트: curl http://${HOST_IP}:11434/api/tags"
echo "   • Docker 테스트: docker exec -it n8n-docker-compose-n8n-1 wget -qO- http://${HOST_IP}:11434/api/tags"
echo ""
echo "💡 네트워크 변경 시 이 스크립트를 다시 실행하세요:"
echo "   ./update-ollama-host.sh"
echo ""

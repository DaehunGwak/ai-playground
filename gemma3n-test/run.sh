#!/bin/bash

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

check_ollama() {
    log_info "Ollama 서버 상태 확인 중..."
    
    if ! command_exists ollama; then
        log_error "Ollama가 설치되어 있지 않습니다."
        log_info "설치: brew install ollama"
        return 1
    fi
    
    if ! ollama list >/dev/null 2>&1; then
        log_warning "Ollama 서버가 실행되지 않았습니다."
        log_info "다른 터미널에서 'ollama serve' 명령을 실행하세요."
        return 1
    fi
    
    if ! ollama list | grep -q "gemma3n"; then
        log_warning "Gemma3N 모델이 설치되지 않았습니다."
        log_info "'ollama pull gemma3n:e4b' 명령으로 모델을 다운로드하세요."
        return 1
    fi
    
    log_success "Ollama 서버와 Gemma3N 모델이 준비되었습니다."
    return 0
}

setup_backend() {
    log_info "백엔드 설정 중..."
    
    if ! command_exists uv; then
        log_error "uv가 설치되어 있지 않습니다."
        log_info "설치: curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
    
    if [ ! -d ".venv" ]; then
        log_info "가상환경 생성 중..."
        uv venv
    fi
    
    log_info "Python 의존성 설치 중..."
    uv pip install fastapi uvicorn python-multipart ollama pydantic python-dotenv aiofiles
    
    log_success "백엔드 설정 완료"
}

run_backend() {
    setup_backend
    
    log_info "백엔드 서버 시작 중..."
    log_info "서버 주소: http://localhost:8000"
    log_info "API 문서: http://localhost:8000/docs"
    
    source .venv/bin/activate
    uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
}

setup_frontend() {
    log_info "프론트엔드 설정 중..."
    
    if ! command_exists node; then
        log_error "Node.js가 설치되어 있지 않습니다."
        log_info "설치: brew install node"
        exit 1
    fi
    
    cd frontend
    
    if [ ! -d "node_modules" ]; then
        log_info "npm 의존성 설치 중..."
        npm install
    fi
    
    cd ..
    log_success "프론트엔드 설정 완료"
}

run_frontend() {
    setup_frontend
    
    log_info "프론트엔드 서버 시작 중..."
    log_info "서버 주소: http://localhost:5173"
    
    cd frontend
    npm run dev
}

run_all() {
    log_info "백엔드와 프론트엔드를 모두 실행합니다..."
    
    check_ollama || log_warning "Ollama를 먼저 설정해주세요."
    
    setup_backend
    setup_frontend
    
    log_success "설정 완료! 서버들을 시작합니다..."
    echo ""
    log_info "백엔드: http://localhost:8000"
    log_info "프론트엔드: http://localhost:5173"
    log_info "종료하려면 Ctrl+C를 두 번 누르세요."
    echo ""
    
    source .venv/bin/activate
    uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload &
    BACKEND_PID=$!
    
    sleep 3
    
    cd frontend
    npm run dev &
    FRONTEND_PID=$!
    
    trap "log_info 'Shutting down...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM
    
    wait
}

main() {
    echo ""
    echo "================================================"
    echo "   🎵 Gemma3N 음악 분석기 실행 스크립트"
    echo "================================================"
    echo ""
    
    case "${1:-all}" in
        backend)
            check_ollama
            run_backend
            ;;
        frontend)
            run_frontend
            ;;
        all)
            run_all
            ;;
        check)
            check_ollama
            ;;
        *)
            echo "사용법: $0 {backend|frontend|all|check}"
            echo ""
            echo "  backend  - 백엔드만 실행"
            echo "  frontend - 프론트엔드만 실행"
            echo "  all      - 백엔드 + 프론트엔드 실행 (기본값)"
            echo "  check    - Ollama 상태 확인"
            echo ""
            exit 1
            ;;
    esac
}

main "$@"


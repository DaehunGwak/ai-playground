"""
Gemma3N 음악 분석 백엔드 API
FastAPI를 사용한 RESTful API 서버
"""

import os
from pathlib import Path
from typing import Optional

import aiofiles
import ollama
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(
    title="Gemma3N Music Analyzer API",
    description="AI 기반 음악 특징 분석 API",
    version="0.3.2",
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 업로드 디렉토리 설정
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# 지원하는 오디오 포맷
SUPPORTED_FORMATS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"}


class AnalysisRequest(BaseModel):
    """음악 분석 요청 모델"""

    audio_filename: str
    user_message: str


class AnalysisResponse(BaseModel):
    """음악 분석 응답 모델"""

    analysis: str
    audio_filename: str


class ChatRequest(BaseModel):
    """일반 채팅 요청 모델"""

    message: str


class ChatResponse(BaseModel):
    """일반 채팅 응답 모델"""

    response: str


@app.get("/")
async def root():
    """헬스 체크 엔드포인트"""
    return {
        "status": "ok",
        "message": "Gemma3N Music Analyzer API",
        "ollama_status": await check_ollama_status(),
    }


async def check_ollama_status() -> dict:
    """Ollama 서버 상태 확인"""
    try:
        models = ollama.list()
        gemma3n_available = any("gemma3n" in model.get("name", "") for model in models.get("models", []))
        return {
            "available": True,
            "gemma3n_installed": gemma3n_available,
        }
    except Exception as e:
        return {
            "available": False,
            "error": str(e),
        }


@app.post("/api/upload-audio")
async def upload_audio(file: UploadFile = File(...)):
    """음악 파일 업로드 엔드포인트"""
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 파일 형식입니다. 지원 형식: {', '.join(SUPPORTED_FORMATS)}",
        )

    content = await file.read()
    if len(content) > 200 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="파일 크기는 200MB를 초과할 수 없습니다.")

    file_path = UPLOAD_DIR / file.filename
    async with aiofiles.open(file_path, "wb") as f:
        await f.write(content)

    return {
        "success": True,
        "filename": file.filename,
        "size": len(content),
        "path": str(file_path),
    }


@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_music(audio_filename: str = Form(...), user_message: str = Form(...)):
    """음악 분석 엔드포인트"""
    file_path = UPLOAD_DIR / audio_filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="음악 파일을 찾을 수 없습니다.")

    try:
        response = ollama.chat(
            model="gemma3n:e4b",
            messages=[
                {
                    "role": "user",
                    "content": f"{user_message}\n\n음악 파일: {audio_filename}",
                    "images": [str(file_path.absolute())],
                }
            ],
        )

        return AnalysisResponse(
            analysis=response["message"]["content"],
            audio_filename=audio_filename,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"음악 분석 중 오류가 발생했습니다: {str(e)}",
        )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """일반 Gemma3N 채팅 엔드포인트"""
    try:
        response = ollama.chat(
            model="gemma3n:e4b",
            messages=[
                {
                    "role": "user",
                    "content": request.message,
                }
            ],
        )

        return ChatResponse(response=response["message"]["content"])

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"채팅 중 오류가 발생했습니다: {str(e)}",
        )


@app.delete("/api/audio/{filename}")
async def delete_audio(filename: str):
    """업로드된 음악 파일 삭제"""
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")

    try:
        file_path.unlink()
        return {"success": True, "message": f"{filename}이(가) 삭제되었습니다."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"파일 삭제 중 오류 발생: {str(e)}")


@app.get("/api/uploaded-files")
async def list_uploaded_files():
    """업로드된 모든 음악 파일 목록 조회"""
    files = []
    for file_path in UPLOAD_DIR.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_FORMATS:
            stat = file_path.stat()
            files.append(
                {
                    "filename": file_path.name,
                    "size": stat.st_size,
                    "modified": stat.st_mtime,
                }
            )
    return {"files": files}


@app.get("/api/audio/{filename}")
async def get_audio(filename: str):
    """오디오 파일 스트리밍"""
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")
    
    return FileResponse(
        path=file_path,
        media_type="audio/mpeg",
        headers={"Accept-Ranges": "bytes"}
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)


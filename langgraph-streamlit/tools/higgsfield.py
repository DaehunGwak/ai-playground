"""
Higgsfield API를 이용한 이미지 생성 Tool
"""
import time
import json
import requests

from langchain_core.tools import tool


# Higgsfield API Base URL
HIGGSFIELD_BASE_URL = "https://platform.higgsfield.ai"


def _extract_status_from_response(response_data: dict) -> tuple[str, list | None]:
    """
    API 응답에서 상태와 결과를 추출합니다.
    
    Higgsfield API 응답 구조:
    {
        "id": "...",
        "jobs": [
            {
                "status": "queued|in_progress|completed|failed|nsfw|canceled",
                "results": [...] or null
            }
        ]
    }
    
    또는 단순 구조:
    {
        "status": "...",
        "images": [...]
    }
    """
    # 먼저 jobs 배열에서 상태 확인 (실제 Higgsfield API 구조)
    jobs = response_data.get("jobs", [])
    if jobs and len(jobs) > 0:
        job = jobs[0]
        status = job.get("status", "unknown")
        results = job.get("results")
        return status, results
    
    # jobs가 없으면 최상위 레벨에서 확인 (대체 구조)
    status = response_data.get("status", "unknown")
    results = (
        response_data.get("images") or 
        response_data.get("outputs") or 
        response_data.get("results")
    )
    return status, results


def _extract_image_urls_from_results(results) -> list[str]:
    """결과에서 이미지 URL들을 추출합니다."""
    if not results:
        return []
    
    urls = []
    for item in results:
        if isinstance(item, str):
            urls.append(item)
        elif isinstance(item, dict):
            # 다양한 필드명 시도
            url = (
                item.get("url") or 
                item.get("image_url") or 
                item.get("output_url") or
                item.get("result")
            )
            if url:
                urls.append(url)
    
    return urls


def create_higgsfield_tools(api_key: str, api_secret: str):
    """
    Higgsfield API를 사용하는 이미지 생성 관련 tools 생성
    
    Args:
        api_key: Higgsfield API 키 (hf-api-key)
        api_secret: Higgsfield API Secret (hf-secret)
    
    Returns:
        이미지 생성 관련 tool 함수들의 리스트
    """
    
    def _get_headers():
        """공통 헤더 반환"""
        return {
            "hf-api-key": api_key,
            "hf-secret": api_secret,
            "Content-Type": "application/json"
        }
    
    def _check_generation_status(request_id: str) -> dict:
        """생성 상태 확인 (내부 함수)"""
        url = f"{HIGGSFIELD_BASE_URL}/requests/{request_id}/status"
        try:
            response = requests.get(url, headers=_get_headers(), timeout=30)
            if response.status_code == 200:
                return response.json()
            return {"status": "error", "message": response.text, "status_code": response.status_code}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    @tool
    def generate_image(
        prompt: str, 
        aspect_ratio: str = "4:3",
        num_images: int = 1,
        output_format: str = "png"
    ) -> str:
        """
        Higgsfield API를 사용하여 텍스트 프롬프트 기반으로 이미지를 생성합니다.
        이 도구는 이미지 생성이 완료될 때까지 자동으로 대기하고 결과를 반환합니다.
        
        사용자가 "이미지 생성해줘", "그림 그려줘", "이미지 만들어줘" 등을 요청하면 이 도구를 사용하세요.
        
        Args:
            prompt: 이미지 생성을 위한 텍스트 프롬프트 (최소 2자 이상). 영어로 작성하면 더 좋은 결과를 얻습니다.
            aspect_ratio: 이미지 비율. 가능한 값: auto, 1:1, 4:3, 3:4, 3:2, 2:3, 16:9, 9:16 (기본값: 4:3)
            num_images: 생성할 이미지 수, 1-4 사이 (기본값: 1)
            output_format: 출력 포맷, jpeg 또는 png (기본값: png)
        
        Returns:
            생성된 이미지 URL과 관련 정보, 또는 에러 메시지
        """
        # 입력 검증
        if len(prompt) < 2:
            return "❌ 프롬프트는 최소 2자 이상이어야 합니다."
        
        valid_ratios = ["auto", "1:1", "4:3", "3:4", "3:2", "2:3", "16:9", "9:16"]
        if aspect_ratio not in valid_ratios:
            aspect_ratio = "4:3"
        
        num_images = max(1, min(4, num_images))
        
        if output_format not in ["jpeg", "png"]:
            output_format = "png"
        
        # API 요청
        url = f"{HIGGSFIELD_BASE_URL}/v1/text2image/nano-banana"
        data = {
            "params": {
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
                "num_images": num_images,
                "output_format": output_format,
                "input_images": []
            }
        }
        
        try:
            # 이미지 생성 요청
            response = requests.post(url, json=data, headers=_get_headers(), timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                
                # request_id 추출 (id 또는 request_id 필드)
                request_id = result.get("id") or result.get("request_id")
                
                if not request_id:
                    return f"❌ 요청 ID를 받지 못했습니다: {json.dumps(result, ensure_ascii=False)}"
                
                # 초기 상태 확인
                initial_status, initial_results = _extract_status_from_response(result)
                
                # 이미 완료된 경우 (동기 응답)
                if initial_status == "completed" and initial_results:
                    image_urls = _extract_image_urls_from_results(initial_results)
                    if image_urls:
                        image_list = "\n".join([f"- {img_url}" for img_url in image_urls])
                        return f"✅ 이미지 생성 완료!\n\n🖼️ 생성된 이미지:\n{image_list}\n\n📋 Request ID: {request_id}"
                
                # 상태 폴링 (최대 180초 대기)
                max_wait = 180
                poll_interval = 3
                elapsed = 0
                
                while elapsed < max_wait:
                    status_result = _check_generation_status(request_id)
                    current_status, results = _extract_status_from_response(status_result)
                    
                    if current_status == "completed":
                        # 완료됨 - 결과 반환
                        image_urls = _extract_image_urls_from_results(results)
                        
                        if image_urls:
                            image_list = "\n".join([f"- {img_url}" for img_url in image_urls])
                            return f"✅ 이미지 생성 완료!\n\n🖼️ 생성된 이미지:\n{image_list}\n\n📋 Request ID: {request_id}"
                        else:
                            # 이미지 URL을 찾지 못한 경우 전체 응답 반환
                            return f"✅ 이미지 생성 완료!\n\n📋 Request ID: {request_id}\n\n전체 응답:\n{json.dumps(status_result, ensure_ascii=False, indent=2)}"
                    
                    elif current_status == "failed":
                        # 실패 원인 추출
                        jobs = status_result.get("jobs", [])
                        error_msg = "알 수 없는 오류"
                        if jobs:
                            error_msg = jobs[0].get("error") or jobs[0].get("message") or error_msg
                        return f"❌ 이미지 생성 실패: {error_msg}\n\n📋 Request ID: {request_id}"
                    
                    elif current_status == "nsfw":
                        return f"❌ NSFW 컨텐츠가 감지되어 이미지 생성이 거부되었습니다.\n다른 프롬프트로 시도해주세요.\n\n📋 Request ID: {request_id}"
                    
                    elif current_status == "canceled":
                        return f"❌ 이미지 생성이 취소되었습니다.\n\n📋 Request ID: {request_id}"
                    
                    elif current_status == "error":
                        error_msg = status_result.get("message", "상태 확인 중 오류 발생")
                        return f"❌ 상태 확인 오류: {error_msg}\n\n📋 Request ID: {request_id}"
                    
                    elif current_status in ["queued", "in_progress"]:
                        time.sleep(poll_interval)
                        elapsed += poll_interval
                    
                    else:
                        # 알 수 없는 상태도 대기
                        time.sleep(poll_interval)
                        elapsed += poll_interval
                
                # 타임아웃
                return f"⏰ 이미지 생성 시간 초과 (180초).\n나중에 check_image_status 도구로 상태를 확인해주세요.\n\n📋 Request ID: {request_id}"
            
            elif response.status_code == 401:
                return "❌ 인증 실패: Higgsfield API 키 또는 Secret이 유효하지 않습니다."
            elif response.status_code == 422:
                try:
                    error_detail = response.json()
                    return f"❌ 유효성 검증 오류: {json.dumps(error_detail, ensure_ascii=False)}"
                except:
                    return f"❌ 유효성 검증 오류: {response.text}"
            elif response.status_code == 429:
                return "❌ 요청 한도 초과: 잠시 후 다시 시도해주세요."
            else:
                return f"❌ 이미지 생성 요청 실패 (상태 코드: {response.status_code}): {response.text}"
                
        except requests.exceptions.Timeout:
            return "❌ 요청 시간 초과: 서버 응답이 너무 오래 걸립니다."
        except requests.exceptions.RequestException as e:
            return f"❌ 네트워크 오류: {str(e)}"
        except Exception as e:
            return f"❌ 예상치 못한 오류: {str(e)}"
    
    @tool
    def check_image_status(request_id: str) -> str:
        """
        이전에 요청한 이미지 생성의 현재 상태를 확인합니다.
        
        Args:
            request_id: 이미지 생성 요청 시 받은 Request ID (UUID 형식)
        
        Returns:
            현재 생성 상태와 관련 정보
        """
        try:
            status_result = _check_generation_status(request_id)
            current_status, results = _extract_status_from_response(status_result)
            
            status_emoji = {
                "queued": "🕐",
                "in_progress": "⏳",
                "completed": "✅",
                "failed": "❌",
                "nsfw": "🚫",
                "canceled": "🚫",
                "error": "❌"
            }
            
            status_text = {
                "queued": "대기 중",
                "in_progress": "생성 중",
                "completed": "완료",
                "failed": "실패",
                "nsfw": "NSFW 차단됨",
                "canceled": "취소됨",
                "error": "오류"
            }
            
            emoji = status_emoji.get(current_status, "❓")
            text = status_text.get(current_status, current_status)
            
            response_text = f"{emoji} 상태: {text}\n\n📋 Request ID: {request_id}"
            
            if current_status == "completed":
                image_urls = _extract_image_urls_from_results(results)
                if image_urls:
                    image_list = "\n".join([f"- {img_url}" for img_url in image_urls])
                    response_text += f"\n\n🖼️ 생성된 이미지:\n{image_list}"
                else:
                    response_text += f"\n\n전체 응답:\n{json.dumps(status_result, ensure_ascii=False, indent=2)}"
            
            elif current_status == "error":
                error_msg = status_result.get("message", "알 수 없는 오류")
                response_text += f"\n\n오류 내용: {error_msg}"
            
            return response_text
                
        except Exception as e:
            return f"❌ 상태 확인 중 오류 발생: {str(e)}"
    
    @tool
    def cancel_image_generation(request_id: str) -> str:
        """
        진행 중인 이미지 생성 요청을 취소합니다.
        
        Args:
            request_id: 취소할 이미지 생성 요청의 Request ID (UUID 형식)
        
        Returns:
            취소 결과 메시지
        """
        try:
            cancel_url = f"{HIGGSFIELD_BASE_URL}/requests/{request_id}/cancel"
            response = requests.post(cancel_url, headers=_get_headers(), timeout=30)
            
            if response.status_code == 202:
                return f"✅ 이미지 생성 요청이 성공적으로 취소되었습니다.\n\n📋 Request ID: {request_id}"
            elif response.status_code == 404:
                return f"❌ 해당 요청을 찾을 수 없습니다. Request ID를 확인해주세요.\n\n📋 Request ID: {request_id}"
            else:
                return f"❌ 취소 요청 실패 (상태 코드: {response.status_code}): {response.text}"
                
        except requests.exceptions.RequestException as e:
            return f"❌ 네트워크 오류: {str(e)}"
        except Exception as e:
            return f"❌ 예상치 못한 오류: {str(e)}"
    
    return [generate_image, check_image_status, cancel_image_generation]

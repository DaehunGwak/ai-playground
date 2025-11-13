import axios from 'axios'

const API_BASE_URL = '/api'

export interface UploadResponse {
  success: boolean
  filename: string
  size: number
  path: string
}

export interface AnalysisResponse {
  analysis: string
  audio_filename: string
}

export interface ChatRequest {
  message: string
}

export interface ChatResponse {
  response: string
}

export async function uploadAudio(file: File): Promise<UploadResponse> {
  const formData = new FormData()
  formData.append('file', file)

  try {
    const response = await axios.post<UploadResponse>(
      `${API_BASE_URL}/upload-audio`,
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    )
    return response.data
  } catch (error: any) {
    if (error.response?.data?.detail) {
      throw new Error(error.response.data.detail)
    }
    throw new Error('파일 업로드에 실패했습니다.')
  }
}

export async function analyzeMusic(
  audioFilename: string,
  userMessage: string
): Promise<AnalysisResponse> {
  const formData = new FormData()
  formData.append('audio_filename', audioFilename)
  formData.append('user_message', userMessage)

  try {
    const response = await axios.post<AnalysisResponse>(
      `${API_BASE_URL}/analyze`,
      formData
    )
    return response.data
  } catch (error: any) {
    if (error.response?.data?.detail) {
      throw new Error(error.response.data.detail)
    }
    throw new Error('음악 분석에 실패했습니다.')
  }
}

export async function chat(message: string): Promise<ChatResponse> {
  try {
    const response = await axios.post<ChatResponse>(
      `${API_BASE_URL}/chat`,
      { message }
    )
    return response.data
  } catch (error: any) {
    if (error.response?.data?.detail) {
      throw new Error(error.response.data.detail)
    }
    throw new Error('채팅 중 오류가 발생했습니다.')
  }
}


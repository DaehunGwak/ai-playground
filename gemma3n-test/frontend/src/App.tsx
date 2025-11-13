import { useState, useRef, useEffect } from 'react'
import { Music, Send, Paperclip, Loader2, AlertCircle, X } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { uploadAudio, analyzeMusic, chat } from './api'
import './App.css'

interface Message {
  role: 'user' | 'assistant'
  content: string
  audioFile?: string
}

function App() {
  const [audioFile, setAudioFile] = useState<File | null>(null)
  const [uploadedFileName, setUploadedFileName] = useState<string>('')
  const [messages, setMessages] = useState<Message[]>([])
  const [inputMessage, setInputMessage] = useState('')
  const [isUploading, setIsUploading] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [error, setError] = useState<string>('')
  const fileInputRef = useRef<HTMLInputElement>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  // 메시지가 추가될 때마다 스크롤을 맨 아래로
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isProcessing])

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    setError('')
    setAudioFile(file)
    setIsUploading(true)

    try {
      const result = await uploadAudio(file)
      setUploadedFileName(result.filename)
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: `🎵 "${result.filename}" 파일이 준비되었습니다. (${(result.size / 1024 / 1024).toFixed(2)}MB)\n\n이 음악에 대해 무엇이든 물어보세요!`,
        },
      ])
    } catch (err: any) {
      setError(err.message || '파일 업로드에 실패했습니다.')
      setAudioFile(null)
    } finally {
      setIsUploading(false)
    }
  }

  const handleRemoveAudioFile = () => {
    setAudioFile(null)
    setUploadedFileName('')
  }

  const handleSendMessage = async () => {
    if (!inputMessage.trim()) return

    const userMessage = inputMessage.trim()
    const hasAudio = !!uploadedFileName

    setInputMessage('')
    setMessages((prev) => [
      ...prev,
      {
        role: 'user',
        content: userMessage,
        audioFile: hasAudio ? uploadedFileName : undefined,
      },
    ])
    setIsProcessing(true)
    setError('')

    try {
      let responseContent: string

      if (hasAudio) {
        const result = await analyzeMusic(uploadedFileName, userMessage)
        responseContent = result.analysis
      } else {
        const result = await chat(userMessage)
        responseContent = result.response
      }

      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: responseContent },
      ])
    } catch (err: any) {
      setError(err.message || '처리 중 오류가 발생했습니다.')
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: '❌ 죄송합니다. 오류가 발생했습니다. 다시 시도해주세요.',
        },
      ])
    } finally {
      setIsProcessing(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  return (
    <div className="app">
      <header className="header">
        <div className="header-content">
          <Music className="header-icon" size={32} />
          <div>
            <h1>Gemma3N AI Assistant</h1>
            <p>음악 분석 & 대화형 AI</p>
          </div>
        </div>
      </header>

      <main className="main">
        {error && (
          <div className="error-message">
            <AlertCircle size={18} />
            <span>{error}</span>
          </div>
        )}

        <div className="chat-container">
          <div className="messages">
            {messages.length === 0 && (
              <div className="welcome-message">
                <Music size={48} />
                <h2>Gemma3N AI Assistant에 오신 것을 환영합니다</h2>
                <p>
                  💬 무엇이든 질문하거나<br />
                  🎵 음악 파일을 첨부하여 분석 요청하세요
                </p>
                <div className="welcome-examples">
                  <span>예시: "Python 코드 작성 도와줘"</span>
                  <span>예시: 🎵 + "이 곡의 장르는?"</span>
                </div>
              </div>
            )}
            {messages.map((message, index) => (
              <div key={index} className={`message ${message.role}`}>
                {message.audioFile && message.role === 'user' && (
                  <div className="message-audio-badge">
                    <Music size={14} />
                    <span>{message.audioFile}</span>
                  </div>
                )}
                <div className="message-content">
                  {message.role === 'assistant' ? (
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                      {message.content}
                    </ReactMarkdown>
                  ) : (
                    message.content
                  )}
                </div>
              </div>
            ))}
            {isProcessing && (
              <div className="message assistant">
                <div className="message-content loading">
                  <Loader2 className="spin" size={20} />
                  {uploadedFileName ? '음악을 분석하고 있습니다...' : '생각 중입니다...'}
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <div className="input-area">
            {audioFile && uploadedFileName && (
              <div className="audio-player-section">
                <div className="attached-file-header">
                  <div className="attached-file">
                    <Music size={16} />
                    <span>{audioFile.name}</span>
                    <button onClick={handleRemoveAudioFile} className="remove-file-btn">
                      <X size={16} />
                    </button>
                  </div>
                </div>
                <audio 
                  controls 
                  className="audio-player"
                  src={`/api/audio/${uploadedFileName}`}
                >
                  브라우저가 오디오를 지원하지 않습니다.
                </audio>
              </div>
            )}
            <div className="input-container">
              <button
                className="attach-button"
                onClick={() => fileInputRef.current?.click()}
                disabled={isUploading || isProcessing}
                title="음악 파일 첨부"
              >
                {isUploading ? (
                  <Loader2 className="spin" size={20} />
                ) : (
                  <Paperclip size={20} />
                )}
              </button>
              <input
                ref={fileInputRef}
                type="file"
                accept=".mp3,.wav,.m4a,.flac,.ogg,.aac"
                onChange={handleFileSelect}
                style={{ display: 'none' }}
              />
              <textarea
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder={
                  uploadedFileName
                    ? '음악에 대해 질문하세요...'
                    : '메시지를 입력하세요... (🎵 파일 첨부 가능)'
                }
                disabled={isProcessing}
                rows={1}
              />
              <button
                onClick={handleSendMessage}
                disabled={!inputMessage.trim() || isProcessing}
                className="send-button"
              >
                {isProcessing ? <Loader2 className="spin" size={20} /> : <Send size={20} />}
              </button>
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}

export default App


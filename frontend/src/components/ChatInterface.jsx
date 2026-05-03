import React, { useState, useRef, useEffect, useCallback } from 'react'
import {
  Box,
  TextField,
  IconButton,
  Paper,
  Typography,
  Chip,
  Alert,
  CircularProgress,
  LinearProgress,
  Divider
} from '@mui/material'
import SendIcon from '@mui/icons-material/Send'
import PsychologyIcon from '@mui/icons-material/Psychology'
import WarningIcon from '@mui/icons-material/Warning'
import ArticleIcon from '@mui/icons-material/Article'
import { sendChatMessage } from '../services/api'

const ChatInterface = () => {
  const [messages, setMessages] = useState([])
  const [inputValue, setInputValue] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)
  const [conversationId, setConversationId] = useState(null)
  const [streamingContent, setStreamingContent] = useState('')
  const [currentSeverity, setCurrentSeverity] = useState(null)
  const [currentSources, setCurrentSources] = useState([])
  
  const messagesEndRef = useRef(null)
  const chatContainerRef = useRef(null)

  // Scroll to bottom when messages change
  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [])

  useEffect(() => {
    scrollToBottom()
  }, [messages, streamingContent, scrollToBottom])

  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!inputValue.trim() || isLoading) return

    const userMessage = inputValue.trim()
    setInputValue('')
    setError(null)
    setStreamingContent('')

    // Add user message to chat
    const newMessages = [...messages, { role: 'user', content: userMessage }]
    setMessages(newMessages)

    setIsLoading(true)

    try {
      // Send request with streaming
      await sendChatMessage(
        userMessage,
        conversationId,
        newMessages.slice(0, -1), // history excluding current
        {
          onChunk: (chunk) => {
            setStreamingContent(prev => prev + chunk)
          },
          onComplete: (response) => {
            const assistantMessage = {
              role: 'assistant',
              content: response.message,
              severity: response.severity_level,
              isCrisis: response.is_crisis,
              sources: response.sources,
              timestamp: new Date().toISOString()
            }
            setMessages(prev => [...prev, assistantMessage])
            setConversationId(response.conversation_id)
            setCurrentSeverity(response.severity_level)
            setCurrentSources(response.sources)
            setStreamingContent('')
            setIsLoading(false)
          },
          onError: (error) => {
            setError(error.message)
            setStreamingContent('')
            setIsLoading(false)
          }
        }
      )
    } catch (err) {
      setError(err.message || 'Đã xảy ra lỗi khi gửi tin nhắn')
      setIsLoading(false)
    }
  }

  // Handle keyboard shortcuts
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  // Format time
  const formatTime = (timestamp) => {
    if (!timestamp) return ''
    const date = new Date(timestamp)
    return date.toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' })
  }

  // Render message content with line breaks
  const renderMessageContent = (content) => {
    return content.split('\n').map((line, index) => (
      <React.Fragment key={index}>
        {line}
        {index < content.split('\n').length - 1 && <br />}
      </React.Fragment>
    ))
  }

  // Get severity label and color
  const getSeverityInfo = (severity) => {
    if (!severity) return { label: '', color: 'default' }
    switch (severity) {
      case 5:
        return { label: 'KHẨN CẤP', color: 'error' }
      case 4:
        return { label: 'NGHIÊM TRỌNG', color: 'error' }
      case 3:
        return { label: 'TRUNG BÌNH', color: 'warning' }
      case 2:
        return { label: 'NHẸ', color: 'info' }
      case 1:
        return { label: 'THÔNG THƯỜNG', color: 'success' }
      default:
        return { label: '', color: 'default' }
    }
  }

  const severityInfo = getSeverityInfo(currentSeverity)

  return (
    <Box className="chat-container" ref={chatContainerRef}>
      {/* Crisis Banner */}
      {currentSeverity >= 4 && (
        <Alert severity="error" icon={<WarningIcon />} className="crisis-banner" sx={{ mb: 2 }}>
          <strong>⚠️ CẢNH BÁO:</strong> Hệ thống đã phát hiện dấu hiệu khẩn cấp. Hãy liên hệ ngay với giáo viên, phụ huynh hoặc dịch vụ hỗ trợ tâm lý chuyên nghiệp.
        </Alert>
      )}

      {/* Messages */}
      <Box className="messages-container">
        {messages.length === 0 && (
          <Box textAlign="center" py={4}>
            <PsychologyIcon sx={{ fontSize: 64, color: 'primary.main', mb: 2 }} />
            <Typography variant="h6" color="textSecondary" gutterBottom>
              Chào bạn! Tôi là trợ lý tư vấn tâm lý học đường.
            </Typography>
            <Typography variant="body2" color="textSecondary">
              Tôi có thể giúp gì cho bạn hôm nay? Bạn có thể chia sẻ về áp lực học tập, mâu thuẫn bạn bè, hay bất kỳ vấn đề tâm lý nào khác.
            </Typography>
          </Box>
        )}

        {messages.map((msg, index) => (
          <Box key={index} className={`message ${msg.role}`}>
            <Paper
              elevation={1}
              className="message-content"
              sx={{
                bgcolor: msg.role === 'user' ? 'primary.main' : 'grey.200',
                color: msg.role === 'user' ? 'white' : 'text.primary'
              }}
            >
              <Typography variant="body1" style={{ whiteSpace: 'pre-wrap' }}>
                {renderMessageContent(msg.content)}
              </Typography>
              
              {/* Show severity and sources for assistant messages */}
              {msg.role === 'assistant' && (
                <Box mt={1}>
                  {msg.severity && (
                    <Chip
                      label={getSeverityInfo(msg.severity).label}
                      color={getSeverityInfo(msg.severity).color}
                      size="small"
                      sx={{ mr: 1, fontSize: '0.7rem' }}
                    />
                  )}
                  {msg.isCrisis && (
                    <Chip
                      icon={<WarningIcon />}
                      label="Chế độ sơ cứu"
                      color="error"
                      size="small"
                      sx={{ fontSize: '0.7rem' }}
                    />
                  )}
                </Box>
              )}
            </Paper>
            
            <Typography variant="caption" className="message-time">
              {formatTime(msg.timestamp)}
            </Typography>
          </Box>
        ))}

        {/* Streaming message */}
        {streamingContent && (
          <Box className="message assistant">
            <Paper elevation={1} className="message-content" sx={{ bgcolor: 'grey.200' }}>
              <Typography variant="body1" style={{ whiteSpace: 'pre-wrap' }}>
                {renderMessageContent(streamingContent)}
                <CircularProgress size={16} sx={{ ml: 1, display: 'inline' }} />
              </Typography>
            </Paper>
          </Box>
        )}

        {/* Loading indicator */}
        {isLoading && !streamingContent && (
          <Box className="message assistant">
            <Paper elevation={1} className="message-content" sx={{ bgcolor: 'grey.200' }}>
              <Box display="flex" alignItems="center">
                <CircularProgress size={20} sx={{ mr: 2 }} />
                <Typography>Đang suy nghĩ...</Typography>
              </Box>
            </Paper>
          </Box>
        )}

        {/* Error */}
        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}

        <div ref={messagesEndRef} />
      </Box>

      {/* Sources display */}
      {currentSources.length > 0 && (
        <Box mt={2}>
          <Divider />
          <Typography variant="caption" color="textSecondary" display="block" sx={{ mt: 1 }}>
            <ArticleIcon fontSize="small" sx={{ verticalAlign: 'middle', mr: 0.5 }} />
            Nguồn tham khảo ({currentSources.length}):
          </Typography>
          <Box display="flex" flexWrap="wrap" gap={1} mt={0.5}>
            {currentSources.slice(0, 3).map((source, idx) => (
              <Chip
                key={idx}
                label={source.metadata?.doc_type || 'Nguồn'}
                size="small"
                variant="outlined"
                sx={{ fontSize: '0.7rem' }}
              />
            ))}
            {currentSources.length > 3 && (
              <Chip label={`+${currentSources.length - 3} khác`} size="small" variant="outlined" />
            )}
          </Box>
        </Box>
      )}

      {/* Input area */}
      <Divider sx={{ mt: 'auto' }} />
      <form onSubmit={handleSubmit} className="input-container">
        <TextField
          fullWidth
          variant="outlined"
          placeholder="Nhập tin nhắn của bạn... (Shift+Enter để xuống dòng)"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyPress={handleKeyPress}
          disabled={isLoading}
          multiline
          maxRows={4}
          className="message-input"
        />
        <IconButton
          type="submit"
          color="primary"
          disabled={!inputValue.trim() || isLoading}
          size="large"
        >
          <SendIcon />
        </IconButton>
      </form>
    </Box>
  )
}

export default ChatInterface
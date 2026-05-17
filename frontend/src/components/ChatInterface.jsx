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
  Divider,
  Avatar
} from '@mui/material'
import SendIcon from '@mui/icons-material/Send'
import PsychologyIcon from '@mui/icons-material/Psychology'
import WarningIcon from '@mui/icons-material/Warning'
import ArticleIcon from '@mui/icons-material/Article'
import PersonIcon from '@mui/icons-material/Person'
import ReactMarkdown from 'react-markdown'
import { sendChatMessage } from '../services/api'

const ChatInterface = () => {
  const [messages, setMessages] = useState([])
  const [inputValue, setInputValue] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)
  const [conversationId, setConversationId] = useState(null)
  const [streamingContent, setStreamingContent] = useState('')
  const [currentSeverity, setCurrentSeverity] = useState(null)
  
  const messagesEndRef = useRef(null)
  const chatContainerRef = useRef(null)

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [])

  useEffect(() => {
    scrollToBottom()
  }, [messages, streamingContent, scrollToBottom])

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!inputValue.trim() || isLoading) return

    const userMessage = inputValue.trim()
    setInputValue('')
    setError(null)
    setStreamingContent('')

    const newMessages = [...messages, { role: 'user', content: userMessage, timestamp: new Date().toISOString() }]
    setMessages(newMessages)
    setIsLoading(true)

    try {
      await sendChatMessage(
        userMessage,
        conversationId,
        newMessages.slice(0, -1),
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

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  const formatTime = (timestamp) => {
    if (!timestamp) return ''
    const date = new Date(timestamp)
    return date.toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' })
  }

  const getSeverityInfo = (severity) => {
    switch (severity) {
      case 5: return { label: 'KHẨN CẤP', color: 'error' }
      case 4: return { label: 'NGHIÊM TRỌNG', color: 'error' }
      case 3: return { label: 'TRUNG BÌNH', color: 'warning' }
      case 2: return { label: 'NHẸ', color: 'info' }
      case 1: return { label: 'THÔNG THƯỜNG', color: 'success' }
      default: return { label: '', color: 'default' }
    }
  }

  return (
    <Box className="chat-container" ref={chatContainerRef} sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      {/* Crisis Banner */}
      {currentSeverity >= 4 && (
        <Alert severity="error" icon={<WarningIcon />} sx={{ mb: 2, borderRadius: 2 }}>
          <strong>⚠️ CẢNH BÁO:</strong> Hệ thống phát hiện dấu hiệu cần hỗ trợ khẩn cấp. Hãy liên hệ chuyên gia hoặc người thân ngay.
        </Alert>
      )}

      {/* Messages List */}
      <Box className="messages-container" sx={{ flexGrow: 1, overflowY: 'auto', p: 2 }}>
        {messages.length === 0 && (
          <Box textAlign="center" py={8}>
            <PsychologyIcon sx={{ fontSize: 80, color: 'primary.light', mb: 2, opacity: 0.5 }} />
            <Typography variant="h5" gutterBottom sx={{ fontWeight: 'bold', color: 'primary.main' }}>
              Chào bạn, tôi có thể giúp gì cho bạn?
            </Typography>
            <Typography variant="body1" color="textSecondary">
              Hãy chia sẻ những lo lắng hoặc câu hỏi của bạn về tâm lý học đường.
            </Typography>
          </Box>
        )}

        {messages.map((msg, index) => (
          <Box key={index} sx={{ mb: 4, display: 'flex', flexDirection: 'column', alignItems: msg.role === 'user' ? 'flex-end' : 'flex-start' }}>
            <Box sx={{ display: 'flex', flexDirection: msg.role === 'user' ? 'row-reverse' : 'row', alignItems: 'flex-start', maxWidth: '85%' }}>
              <Avatar sx={{ bgcolor: msg.role === 'user' ? 'primary.main' : '#43766C', width: 35, height: 35, ml: msg.role === 'user' ? 1.5 : 0, mr: msg.role === 'assistant' ? 1.5 : 0 }}>
                {msg.role === 'user' ? <PersonIcon fontSize="small" /> : <PsychologyIcon fontSize="small" />}
              </Avatar>
              
              <Paper elevation={0} sx={{ 
                p: 2, 
                borderRadius: msg.role === 'user' ? '20px 20px 4px 20px' : '20px 20px 20px 4px', 
                bgcolor: msg.role === 'user' ? 'primary.main' : 'white', 
                color: msg.role === 'user' ? 'white' : 'text.primary', 
                border: msg.role === 'assistant' ? '1px solid #e0e0e0' : 'none',
                boxShadow: '0 2px 8px rgba(0,0,0,0.05)'
              }}>
                <Box sx={{ fontSize: '1rem', lineHeight: 1.6 }}>
                  <ReactMarkdown>{msg.content}</ReactMarkdown> 
                </Box>
              </Paper>
            </Box>

            {msg.role === 'assistant' && (
              <Box sx={{ mt: 1, ml: 6, display: 'flex', flexDirection: 'column', gap: 0.8 }}>
                <Box display="flex" alignItems="center" gap={1.5}>
                  {msg.severity && (
                    <Chip 
                      label={`Mức độ: ${getSeverityInfo(msg.severity).label}`} 
                      color={getSeverityInfo(msg.severity).color} 
                      size="small" 
                      sx={{ height: 20, fontSize: '0.65rem', fontWeight: 'bold' }} 
                    />
                  )}
                  <Typography variant="caption" sx={{ color: 'text.disabled' }}>
                    {formatTime(msg.timestamp)}
                  </Typography>
                </Box>

                {msg.sources && msg.sources.filter(s => s.title && s.title !== 'Unknown').length > 0 && (
                  <Box display="flex" flexWrap="wrap" gap={0.5}>
                    <ArticleIcon sx={{ fontSize: 14, color: 'text.secondary', mr: 0.5 }} />
                    {msg.sources
                      .filter(s => s.title && s.title !== 'Unknown') // Lọc bỏ Unknown
                      .map((source, idx) => (
                        <Chip 
                          key={idx} 
                          label={source.title} 
                          variant="outlined"
                          sx={{ fontSize: '0.6rem', height: 18, color: 'text.secondary' }} 
                        />
                      ))}
                  </Box>
                )}
              </Box>
            )}
          </Box>
        ))}

        {/* Streaming / Loading */}
        {(streamingContent || isLoading) && (
          <Box className="message assistant" sx={{ mb: 3, display: 'flex', alignItems: 'flex-end', maxWidth: '85%' }}>
            <Avatar sx={{ bgcolor: 'secondary.main', width: 32, height: 32, mr: 1 }}>
              <PsychologyIcon fontSize="small" />
            </Avatar>
            <Paper elevation={0} sx={{ p: 2, borderRadius: 3, bgcolor: 'background.paper', border: '1px solid #e0e0e0', minWidth: 80 }}>
              {streamingContent ? (
                <ReactMarkdown>{streamingContent}</ReactMarkdown>
              ) : (
                <Box display="flex" alignItems="center" gap={1}>
                  <CircularProgress size={16} thickness={5} />
                  <Typography variant="body2" color="textSecondary">Đang suy nghĩ...</Typography>
                </Box>
              )}
            </Paper>
          </Box>
        )}

        {error && <Alert severity="error" sx={{ mb: 2, borderRadius: 2 }}>{error}</Alert>}
        <div ref={messagesEndRef} />
      </Box>

      {/* Input Field */}
      <Box sx={{ p: 2, backgroundColor: 'background.paper', borderTop: '1px solid #eee' }}>
        <form onSubmit={handleSubmit} style={{ display: 'flex', gap: '8px', alignItems: 'flex-end' }}>
          <TextField
            fullWidth
            multiline
            maxRows={4}
            variant="outlined"
            placeholder="Nhập tin nhắn... (Shift+Enter để xuống dòng)"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={(e) => {
               if (e.key === 'Enter' && !e.shiftKey) {
                 e.preventDefault();
                 handleSubmit(e);
               }
            }}
            disabled={isLoading}
            sx={{ '& .MuiOutlinedInput-root': { borderRadius: 4 } }}
          />
          <IconButton 
            type="submit" 
            color="primary" 
            disabled={!inputValue.trim() || isLoading}
            sx={{ bgcolor: inputValue.trim() ? 'primary.main' : 'transparent', color: inputValue.trim() ? 'white' : 'inherit', '&:hover': { bgcolor: 'primary.dark' }, p: 1.5 }}
          >
            <SendIcon />
          </IconButton>
        </form>
      </Box>
    </Box>
  )
}

export default ChatInterface
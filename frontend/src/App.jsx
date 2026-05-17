import React, { useState, useCallback } from 'react'
import { Container, Box, Typography, Paper, Alert, CircularProgress } from '@mui/material'
import ChatInterface from './components/ChatInterface'
import { checkHealth } from './services/api'
import './App.css'

function App() {
  const [isLoading, setIsLoading] = useState(true)
  const [healthStatus, setHealthStatus] = useState(null)
  const [healthError, setHealthError] = useState(null)

  // Check backend health on mount
  React.useEffect(() => {
    const checkBackend = async () => {
      try {
        const status = await checkHealth()
        setHealthStatus(status)
      } catch (error) {
        setHealthError(error.message)
      } finally {
        setIsLoading(false)
      }
    }
    checkBackend()
  }, [])

  if (isLoading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="100vh">
        <CircularProgress />
        <Typography variant="body1" style={{ marginLeft: 16 }}>
          Đang kiểm tra kết nối đến hệ thống...
        </Typography>
      </Box>
    )
  }

  if (healthError || healthStatus?.status === 'unhealthy') {
    return (
      <Container maxWidth="md" style={{ marginTop: 50 }}>
        <Alert severity="error">
          <Typography variant="h6">Không thể kết nối đến hệ thống</Typography>
          <Typography variant="body2">
            {healthError || 'Một hoặc nhiều dịch vụ backend không hoạt động. Vui lòng kiểm tra:'}
          </Typography>
          <ul>
            <li>Ollama đang chạy và đã pull model qwen2.5:3b</li>
            <li>Qdrant và Neo4j đang chạy</li>
            <li>Backend API đang chạy tại port 8000</li>
          </ul>
        </Alert>
      </Container>
    )
  }

  return (
    <Container maxWidth="lg" sx={{ py: 3, height: '100vh', display: 'flex', flexDirection: 'column' }}>
      <Box textAlign="center" mb={4}>
        <Typography 
          variant="h2" 
          component="h1" 
          sx={{ 
            fontWeight: 900, 
            color: '#43766C', 
            textTransform: 'uppercase',
            textShadow: '2px 2px 4px rgba(0,0,0,0.1)',
            fontSize: { xs: '2.2rem', md: '3.8rem' },
            mb: 1
          }}
        >
          Chatbot Tư Vấn Tâm Lý Học Đường
        </Typography>
        <Typography variant="h6" color="textSecondary" sx={{ fontStyle: 'italic', opacity: 0.8 }}>
          Hệ thống hỗ trợ thông minh dựa trên mô hình GraphRAG
        </Typography>
      </Box>

      <Paper 
        elevation={4} 
        sx={{ 
          flex: 1, 
          display: 'flex', 
          flexDirection: 'column', 
          borderRadius: 4,
          overflow: 'hidden', 
          mb: 2
        }}
      >
        <ChatInterface />
      </Paper>

      <Box textAlign="center" pb={1}>
        <Typography variant="caption" color="textSecondary">
          ⚠️ Hệ thống chỉ cung cấp hỗ trợ tư vấn ban đầu. Trong trường hợp khẩn cấp, vui lòng liên hệ dịch vụ chuyên nghiệp.
        </Typography>
      </Box>
    </Container>
  )
}

export default App
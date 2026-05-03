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
    <Container maxWidth="lg" style={{ marginTop: 20, marginBottom: 20 }}>
      <Box textAlign="center" mb={3}>
        <Typography variant="h4" component="h1" gutterBottom color="primary">
          🤖 Chatbot Tư vấn Tâm lý Học đường
        </Typography>
        <Typography variant="subtitle1" color="textSecondary">
          Hệ thống hỗ trợ tư vấn tâm lý sử dụng GraphRAG
        </Typography>
        <Box mt={2}>
          <Typography variant="caption" color="textSecondary">
            LLM: Qwen 2.5-3B (Ollama) | Vector DB: Qdrant | Graph DB: Neo4j
          </Typography>
        </Box>
      </Box>

      <Paper elevation={3} style={{ padding: 20, minHeight: '70vh', display: 'flex', flexDirection: 'column' }}>
        <ChatInterface />
      </Paper>

      <Box mt={2} textAlign="center">
        <Typography variant="caption" color="textSecondary">
          ⚠️ Hệ thống chỉ cung cấp hỗ trợ tư vấn ban đầu. Trong trường hợp khẩn cấp, vui lòng liên hệ dịch vụ chuyên nghiệp.
        </Typography>
      </Box>
    </Container>
  )
}

export default App
import axios from 'axios'

// Create axios instance
const api = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8000',
  timeout: 60000, // 60 seconds for potentially long LLM responses
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor for logging
api.interceptors.request.use(
  (config) => {
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`)
    return config
  },
  (error) => {
    console.error('API Request Error:', error)
    return Promise.reject(error)
  }
)

// Response interceptor
api.interceptors.response.use(
  (response) => {
    console.log(`API Response: ${response.status} ${response.config.url}`)
    return response
  },
  (error) => {
    console.error('API Response Error:', error.response?.data || error.message)
    return Promise.reject(error)
  }
)

// Health check
export const checkHealth = async (includeServices = false) => {
  const response = await api.get(`/health?include_services=${includeServices}`)
  return response.data
}

// Chat completion with streaming
export const sendChatMessage = (
  message,
  conversationId = null,
  history = [],
  callbacks = {}
) => {
  return new Promise((resolve, reject) => {
    const payload = {
      message,
      conversation_id: conversationId,
      history: history.map(msg => ({
        role: msg.role,
        content: msg.content
      })),
      user_id: null // Could be added later
    }

    api.post('/chat/completion', payload, {
      responseType: 'stream',
      onUploadProgress: (progressEvent) => {
        // Not used for small payloads
      },
    })
      .then(async (response) => {
        const reader = response.data.getReader()
        const decoder = new TextDecoder()
        let buffer = ''

        const readStream = async () => {
          try {
            const { done, value } = await reader.read()
            if (done) {
              // Signal complete if we got a full response
              if (callbacks.onComplete) {
                // Parse the final message from accumulated data if needed
                callbacks.onComplete({
                  message: buffer,
                  conversation_id: conversationId,
                  severity_level: 1,
                  is_crisis: false,
                  sources: []
                })
              }
              return resolve()
            }

            buffer += decoder.decode(value, { stream: true })
            const lines = buffer.split('\n')
            buffer = lines.pop() // Keep incomplete line in buffer

            for (const line of lines) {
              if (line.trim()) {
                try {
                  // Parse SSE data if needed, but backend returns raw text chunks
                  // For now, treat each chunk as text
                  if (callbacks.onChunk) {
                    callbacks.onChunk(line)
                  }
                } catch (e) {
                  console.warn('Failed to parse chunk:', e)
                }
              }
            }

            readStream()
          } catch (error) {
            if (callbacks.onError) {
              callbacks.onError(error)
            }
            reject(error)
          }
        }

        readStream()
      })
      .catch((error) => {
        if (callbacks.onError) {
          callbacks.onError(error)
        }
        reject(error)
      })
  })
}

// Get conversation history
export const getConversation = async (conversationId) => {
  const response = await api.get(`/chat/conversation/${conversationId}`)
  return response.data
}

// Delete conversation
export const deleteConversation = async (conversationId) => {
  const response = await api.delete(`/chat/conversation/${conversationId}`)
  return response.data
}

// Clear all conversations
export const clearAllConversations = async () => {
  const response = await api.post('/chat/clear')
  return response.data
}

export default api
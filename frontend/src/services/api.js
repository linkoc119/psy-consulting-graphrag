import axios from 'axios'

// Create axios instance
const api = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8000',
  timeout: 300000, // 5 minutes for potentially long LLM responses
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

// Chat completion (non-streaming - returns full response)
export const sendChatMessage = async (message, conversationId, history, callbacks) => {
  try {
    const payload = {
      message,
      conversation_id: conversationId,
      history: history.map(msg => ({ role: msg.role, content: msg.content })),
      user_id: null
    };

    const response = await api.post('/chat/completion', payload);
    
    // Trả về object dữ liệu đã được parse sẵn
    if (callbacks.onComplete) {
      callbacks.onComplete(response.data);
    }
    return response.data;
  } catch (error) {
    if (callbacks.onError) {
      callbacks.onError(error);
    }
    throw error;
  }
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
import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import { ThemeProvider, createTheme } from '@mui/material/styles'
import CssBaseline from '@mui/material/CssBaseline'

// Create a theme with Vietnamese-friendly fonts
const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#43766C',
      light: '#7091F5',
    },
    secondary: {
      main: '#F8FAE5',
    },
    background: {
      default: '#F5F7F8',
      paper: '#ffffff',
    },
  },
  typography: {
    fontFamily: ['"Inter"', '"Noto Sans Vietnamese"', '"Roboto"', 'sans-serif'].join(','),
    h4: { fontWeight: 700, color: '#1A3C40' },
  },
  shape: {
    borderRadius: 12, 
  },
});

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <App />
    </ThemeProvider>
  </React.StrictMode>
)
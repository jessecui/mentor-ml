import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    proxy: {
      '/chat': 'http://localhost:8080',
      '/health': 'http://localhost:8080',
      '/benchmark': 'http://localhost:8080',
      '/validate-password': 'http://localhost:8080',
    },
  },
})

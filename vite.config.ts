import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  optimizeDeps: {
    exclude: ['@tensorflow/tfjs-tflite']
  },
  server: {
    fs: {
      allow: ['..']
    }
  },
  define: {
    global: 'globalThis',
  },
  resolve: {
    alias: {
      // TensorFlow Liteの依存関係問題を回避
      './tflite_web_api_client': false,
      '../tflite_web_api_client': false,
    }
  }
})

import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  base: './', // 相対パスを使用して本番環境でのパス問題を解決
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
    rollupOptions: {
      output: {
        manualChunks: {
          // TensorFlow.jsを別チャンクに分離してロード速度を改善
          tensorflow: ['@tensorflow/tfjs'],
        },
        // モデルファイルをassetsディレクトリに出力
        assetFileNames: (assetInfo) => {
          if (assetInfo.name && assetInfo.name.endsWith('.tflite')) {
            return 'models/[name][extname]';
          }
          return 'assets/[name]-[hash][extname]';
        }
      }
    }
  },
  // srcディレクトリ内のファイルを静的アセットとして扱う
  assetsInclude: ['**/*.tflite'],
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

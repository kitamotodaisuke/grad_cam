/// <reference types="vite/client" />

// TensorFlow Lite モデルファイルのモジュール型定義
declare module '*.tflite' {
  const src: string;
  export default src;
}

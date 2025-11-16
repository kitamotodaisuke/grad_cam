<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# TensorFlow Lite 推論 & Grad-CAM 可視化アプリ

このプロジェクトはTypeScript + ReactでTensorFlow Liteモデルをブラウザで実行し、Grad-CAM風の可視化を行うWebアプリです。

## プロジェクト構造
- `/src/App.tsx`: メインのReactコンポーネント
- `/src/App.css`: UIスタイリング
- TensorFlow.js + @tensorflow/tfjs-tflite を使用

## 主な機能
1. .tfliteモデルファイルのアップロード
2. 画像のドラッグ&ドロップアップロード
3. ブラウザでの推論実行
4. Grad-CAM風ヒートマップ可視化
5. 推論結果の表示

## 開発時の注意点
- TypeScript strict modeを使用
- TensorFlow.jsのメモリ管理（tf.tidy、dispose）を適切に行う
- Canvasを使ったヒートマップ描画
- レスポンシブデザイン対応
- エラーハンドリングを徹底

## コーディングガイドライン
- 関数コンポーネント + Hooks を使用
- useCallback でパフォーマンス最適化
- 型安全性を重視
- ユーザビリティを考慮したUI/UX
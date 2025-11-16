/**
 * TensorFlow Lite 推論 & Grad-CAM 可視化アプリ用型定義
 * @fileoverview アプリケーション全体で使用される型定義を集約
 */

import * as tf from '@tensorflow/tfjs';

/**
 * じゃんけん推論結果の型定義
 */
export interface JankenPrediction {
  /** 予測されたじゃんけんの手（グー、チョキ、パー） */
  label: string;
  /** 予測の信頼度（0-1） */
  confidence: number;
}

/**
 * カスタムTFLiteモデルの型定義
 * TensorFlow Liteモデルを抽象化したインターフェース
 */
export interface CustomTFLiteModel {
  /**
   * 推論実行メソッド
   * @param input - 入力テンソル（通常は前処理済み画像）
   * @returns 推論結果のテンソル
   */
  predict: (input: tf.Tensor) => Promise<tf.Tensor>;
  /** モデルのメモリを解放 */
  dispose: () => void;
  /** モデルの入力形状 [batch, height, width, channels] */
  inputShape: number[];
  /** モデルの出力形状 [classes] */
  outputShape: number[];
}

/**
 * モデルの読み込み状態管理用型定義
 */
export interface ModelState {
  /** 読み込まれたモデル（nullの場合は未読み込み） */
  model: CustomTFLiteModel | null;
  /** モデルが正常に読み込まれているかのフラグ */
  isLoaded: boolean;
  /** エラーメッセージ（エラーがない場合はnull） */
  error: string | null;
}

/**
 * プリセットモデルの定義型
 */
export interface PresetModel {
  /** モデルファイル名 */
  name: string;
  /** 公開フォルダからのパス */
  path: string;
  /** フォールバック用のパス（srcからimportしたもの） */
  fallbackPath?: string;
  /** ユーザー向けの説明文 */
  description: string;
}

/**
 * デバッグ情報管理の型定義
 */
export interface DebugInfo {
  /** デバッグメッセージの配列 */
  debugInfo: string[];
  /** デバッグメッセージを追加する関数 */
  addDebugInfo: (message: string) => void;
  /** デバッグ情報をクリアする関数 */
  clearDebugInfo: () => void;
}

/**
 * TensorFlow.js初期化状態の型定義
 */
export interface TensorFlowState {
  /** TensorFlow.jsの初期化完了フラグ */
  tfReady: boolean;
  /** アプリケーションレベルのエラー */
  appError: string | null;
  /** デバッグ情報 */
  debugInfo: string[];
  /** デバッグメッセージ追加関数 */
  addDebugInfo: (message: string) => void;
  /** エラーリセット関数 */
  resetError: () => void;
}

/**
 * 画像処理関連の型定義
 */
export interface ImageProcessing {
  /** 選択された画像のBase64データURL */
  selectedImage: string | null;
  /** 画像選択処理関数 */
  handleImageUpload: (event: React.ChangeEvent<HTMLInputElement>) => void;
  /** ドラッグオーバー処理関数 */
  handleDragOver: (event: React.DragEvent) => void;
  /** ドロップ処理関数 */
  handleDrop: (event: React.DragEvent) => void;
  /** 画像リセット関数 */
  resetImage: () => void;
}

/**
 * 推論結果の型定義
 */
export interface InferenceResult {
  /** じゃんけん推論結果の配列 */
  predictions: JankenPrediction[];
  /** Grad-CAM可視化データ */
  gradcamData: ImageData | null;
  /** 推論実行中のフラグ */
  isInferring: boolean;
}

/**
 * UIコンポーネントのプロパティ型定義
 */
export interface UIComponentProps {
  /** 共通のクラス名 */
  className?: string;
  /** 子要素 */
  children?: React.ReactNode;
}

/**
 * モデルアップロードコンポーネントのプロパティ
 */
export interface ModelUploadProps extends UIComponentProps {
  /** モデル状態 */
  modelState: ModelState;
  /** モデルアップロード処理関数 */
  onModelUpload: (event: React.ChangeEvent<HTMLInputElement>) => void;
  /** プリセットモデル読み込み処理関数 */
  onPresetModelLoad: (model: PresetModel) => void;
  /** プリセットモデル一覧 */
  presetModels: PresetModel[];
}

/**
 * 画像アップロードコンポーネントのプロパティ
 */
export interface ImageUploadProps extends UIComponentProps {
  /** 画像処理状態 */
  imageProcessing: ImageProcessing;
  /** 画像要素への参照 */
  imageRef: React.RefObject<HTMLImageElement>;
  /** 画像ロード時の処理関数 */
  onImageLoad: () => void;
}

/**
 * 推論結果表示コンポーネントのプロパティ
 */
export interface ResultDisplayProps extends UIComponentProps {
  /** 推論結果 */
  inferenceResult: InferenceResult;
  /** キャンバス要素への参照 */
  canvasRef: React.RefObject<HTMLCanvasElement>;
  /** じゃんけんラベル配列 */
  jankenLabels: string[];
}

/**
 * エラー表示コンポーネントのプロパティ
 */
export interface ErrorDisplayProps extends UIComponentProps {
  /** エラーメッセージ */
  error: string;
  /** デバッグ情報 */
  debugInfo: string[];
  /** エラーリセット関数 */
  onReset: () => void;
}

/**
 * ローディング表示コンポーネントのプロパティ
 */
export interface LoadingDisplayProps extends UIComponentProps {
  /** ローディングメッセージ */
  message: string;
  /** 詳細メッセージ */
  description?: string;
  /** デバッグ情報 */
  debugInfo: string[];
}

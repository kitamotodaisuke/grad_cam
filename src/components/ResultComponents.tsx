import React from 'react';
import { ResultDisplayProps } from '../types';

/**
 * 推論結果表示コンポーネント
 * じゃんけん分類結果とGrad-CAM可視化を表示
 */
export const ResultDisplay: React.FC<ResultDisplayProps> = ({
  inferenceResult,
  canvasRef,
  jankenLabels,
  className = '',
}) => {
  const { predictions, gradcamData } = inferenceResult;

  // 結果がない場合のプレースホルダー
  if (!predictions.length && !gradcamData) {
    return (
      <div className={`dark-card h-full flex items-center justify-center min-h-[400px] ${className}`}>
        <div className="text-center text-dark-muted">
          <div className="text-6xl mb-4">📊</div>
          <h3 className="text-xl font-semibold mb-2">推論結果がここに表示されます</h3>
          <p>モデルを選択し、画像をアップロードして推論を実行してください</p>
        </div>
      </div>
    );
  }

  return (
    <div className={`space-y-6 ${className}`}>
      {/* 推論結果 */}
      {predictions.length > 0 && (
        <div className="dark-card animate-fade-in-up">
          <h2 className="section-title-dark">📊 推論結果</h2>
          <div className="space-y-4">
            {predictions.map((prediction, index) => (
              <div key={index} className="prediction-dark">
                <div className="flex justify-between items-center mb-2">
                  <span className="font-semibold text-dark-primary text-lg">{prediction.label}</span>
                  <span className="font-bold text-blue-400 text-xl">{(prediction.confidence * 100).toFixed(2)}%</span>
                </div>
                <div className="confidence-bar-dark">
                  <div className="confidence-fill-dark" style={{ width: `${prediction.confidence * 100}%` }} />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Grad-CAM可視化 */}
      {gradcamData && (
        <div className="dark-card animate-fade-in-up">
          <h2 className="section-title-dark">🎯 注目部分可視化 (Grad-CAM風)</h2>
          <div className="text-center">
            <canvas ref={canvasRef} className="max-w-full h-auto rounded-lg shadow-lg mb-4 mx-auto" />
            <p className="gradcam-description-dark">赤い部分ほどモデルが注目している領域です</p>
          </div>
        </div>
      )}

      {/* 推論情報 */}
      {(predictions.length > 0 || gradcamData) && (
        <div className="dark-card">
          <h3 className="text-dark-primary font-semibold mb-3">📋 推論情報</h3>
          <div className="text-dark-secondary text-sm space-y-2">
            <div className="flex justify-between">
              <span>モデル形式:</span>
              <span className="text-blue-400">TensorFlow Lite</span>
            </div>
            <div className="flex justify-between">
              <span>入力サイズ:</span>
              <span className="text-blue-400">224×224×3</span>
            </div>
            <div className="flex justify-between">
              <span>出力クラス数:</span>
              <span className="text-blue-400">{jankenLabels.length}クラス</span>
            </div>
            <div className="flex justify-between">
              <span>処理時間:</span>
              <span className="text-green-400">リアルタイム</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

/**
 * 推論実行ボタンコンポーネント
 * モデルと画像が準備されている場合のみ表示
 */
export const InferenceButton: React.FC<{
  isModelReady: boolean;
  hasImage: boolean;
  isInferring: boolean;
  onRunInference: () => void;
  className?: string;
}> = ({ isModelReady, hasImage, isInferring, onRunInference, className = '' }) => {
  if (!isModelReady || !hasImage) {
    return null;
  }

  return (
    <div className={`text-center ${className}`}>
      <button
        onClick={onRunInference}
        disabled={isInferring}
        className={`btn-primary-dark w-full ${isInferring ? 'animate-pulse' : ''}`}
      >
        {isInferring ? (
          <span className="flex items-center justify-center">
            <svg
              className="animate-spin -ml-1 mr-3 h-5 w-5 text-white spinner-dark"
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
            >
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path
                className="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
              ></path>
            </svg>
            推論中...
          </span>
        ) : (
          '推論実行'
        )}
      </button>
    </div>
  );
};

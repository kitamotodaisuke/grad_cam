import React, { useRef } from 'react';
import { ModelUploadProps, ImageUploadProps } from '../types';

/**
 * モデルアップロードコンポーネント
 * TFLiteモデルファイルとプリセットモデルの選択UI
 */
export const ModelUpload: React.FC<ModelUploadProps> = ({
  modelState,
  onModelUpload,
  onPresetModelLoad,
  presetModels,
  className = '',
}) => {
  const modelInputRef = useRef<HTMLInputElement>(null);

  return (
    <div className={`space-y-6 ${className}`}>
      {/* モデルファイルアップロード */}
      <div className="dark-card group relative overflow-hidden">
        {/* 背景アニメーション */}
        <div className="absolute inset-0 bg-gradient-to-br from-blue-500/5 via-purple-500/5 to-cyan-500/5 animate-pulse"></div>

        <h2 className="section-title-dark flex items-center gap-3">
          <div className="relative">
            <div className="w-8 h-8 bg-gradient-to-r from-blue-400 to-purple-500 rounded-lg flex items-center justify-center">
              <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                />
              </svg>
            </div>
            <div className="absolute -top-1 -right-1 w-3 h-3 bg-green-400 rounded-full animate-pulse"></div>
          </div>
          1. AIモデルファイルをアップロード
        </h2>

        {/* カスタムファイル入力エリア */}
        <div className="relative">
          <input
            ref={modelInputRef}
            type="file"
            accept=".tflite, .json"
            onChange={onModelUpload}
            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
          />

          <div className="bg-gray-800/50 border-2 border-dashed border-gray-600 rounded-xl p-8 text-center transition-all duration-300 hover:border-blue-400 hover:bg-gray-700/30 group-hover:shadow-lg group-hover:shadow-blue-500/20">
            {/* アップロードアイコンとアニメーション */}
            <div className="relative mx-auto w-16 h-16 mb-4">
              <div className="absolute inset-0 bg-gradient-to-r from-blue-400 to-purple-500 rounded-full animate-spin opacity-20"></div>
              <div className="relative bg-gray-700 rounded-full w-full h-full flex items-center justify-center">
                <svg className="w-8 h-8 text-blue-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                  />
                </svg>
              </div>

              {/* パーティクル効果 */}
              <div className="absolute top-0 left-0 w-2 h-2 bg-blue-400 rounded-full animate-ping"></div>
              <div
                className="absolute bottom-0 right-0 w-1 h-1 bg-purple-400 rounded-full animate-ping"
                style={{ animationDelay: '0.5s' }}
              ></div>
              <div
                className="absolute top-1/2 left-0 w-1.5 h-1.5 bg-cyan-400 rounded-full animate-ping"
                style={{ animationDelay: '1s' }}
              ></div>
            </div>

            {/* メインテキスト */}
            <div className="space-y-2">
              <h3 className="text-xl font-semibold text-white group-hover:text-blue-300 transition-colors">
                TensorFlow Liteモデルをドロップ
              </h3>
              <p className="text-gray-400 text-sm">
                または<span className="text-blue-400 font-semibold">クリック</span>してファイルを選択
              </p>
            </div>

            {/* サポートフォーマット */}
            <div className="mt-4 flex justify-center gap-4 text-xs">
              <div className="bg-gray-700 px-3 py-1 rounded-full">
                <span className="text-blue-300">.tflite</span>
              </div>
              <div className="bg-gray-700 px-3 py-1 rounded-full">
                <span className="text-purple-300">.json</span>
              </div>
            </div>

            {/* プログレスバー風の装飾 */}
            <div className="mt-6 relative h-1 bg-gray-700 rounded-full overflow-hidden">
              <div className="absolute inset-y-0 left-0 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full w-0 group-hover:w-full transition-all duration-1000 ease-out"></div>
            </div>
          </div>

          {/* ホバー時のグロー効果 */}
          <div className="absolute inset-0 rounded-xl bg-gradient-to-r from-blue-500/10 to-purple-500/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none"></div>
        </div>

        {/* ステータス表示 */}
        <div className="mt-4 space-y-3">
          {modelState.error && (
            <div className="bg-red-900/30 border border-red-500/50 rounded-lg p-4 animate-fadeIn">
              <div className="flex items-center gap-3">
                <div className="w-6 h-6 bg-red-500 rounded-full flex items-center justify-center flex-shrink-0">
                  <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </div>
                <div>
                  <h4 className="text-red-300 font-semibold">エラーが発生しました</h4>
                  <p className="text-red-200 text-sm mt-1">{modelState.error}</p>
                </div>
              </div>
            </div>
          )}

          {modelState.isLoaded && (
            <div className="bg-green-900/30 border border-green-500/50 rounded-lg p-4 animate-fadeIn">
              <div className="flex items-center gap-3">
                <div className="relative">
                  <div className="w-6 h-6 bg-green-500 rounded-full flex items-center justify-center">
                    <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                  </div>
                  <div className="absolute -top-1 -right-1 w-3 h-3 bg-green-400 rounded-full animate-ping"></div>
                </div>
                <div>
                  <h4 className="text-green-300 font-semibold">AIモデル読み込み完了</h4>
                  <p className="text-green-200 text-sm mt-1">推論の準備が整いました</p>
                </div>
              </div>

              {/* AIモデル情報表示 */}
              <div className="mt-3 grid grid-cols-2 gap-4 text-xs">
                <div className="bg-gray-800/50 rounded-lg p-2">
                  <div className="text-gray-400">フォーマット</div>
                  <div className="text-green-300 font-semibold">TensorFlow Lite</div>
                </div>
                <div className="bg-gray-800/50 rounded-lg p-2">
                  <div className="text-gray-400">ステータス</div>
                  <div className="text-green-300 font-semibold flex items-center gap-1">
                    <span>Ready</span>
                    <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* プリセットモデル */}
      <div className="dark-card">
        <h2 className="section-title-dark">2. プリセットモデルを選択</h2>
        <div className="flex flex-col gap-4">
          {presetModels.map((presetModel, index) => (
            <button
              key={index}
              onClick={() => onPresetModelLoad(presetModel)}
              className="btn-success-dark group relative overflow-hidden"
            >
              <div className="flex items-center gap-3">
                {/* AIアイコンのアニメーション */}
                <div className="relative">
                  <div className="absolute inset-0 bg-gradient-to-r from-blue-400 to-purple-500 rounded-full animate-pulse opacity-60"></div>
                  <div className="relative bg-gray-800 rounded-full p-2">
                    <svg
                      className="w-6 h-6 text-blue-300 group-hover:text-white transition-colors duration-300"
                      fill="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path d="M12 2C8.13 2 5 5.13 5 9c0 1.74.68 3.31 1.78 4.5L12 19l5.22-5.5C18.32 12.31 19 10.74 19 9c0-3.87-3.13-7-7-7zm0 9.5c-1.38 0-2.5-1.12-2.5-2.5s1.12-2.5 2.5-2.5 2.5 1.12 2.5 2.5-1.12 2.5-2.5 2.5z" />
                      <circle cx="8" cy="7" r="1" opacity="0.7" />
                      <circle cx="16" cy="7" r="1" opacity="0.7" />
                      <circle cx="12" cy="5" r="0.8" opacity="0.5" />
                      <line x1="8" y1="7" x2="12" y2="9" stroke="currentColor" strokeWidth="0.5" opacity="0.4" />
                      <line x1="16" y1="7" x2="12" y2="9" stroke="currentColor" strokeWidth="0.5" opacity="0.4" />
                      <line x1="12" y1="5" x2="12" y2="9" stroke="currentColor" strokeWidth="0.5" opacity="0.4" />
                    </svg>
                  </div>
                </div>

                {/* テキスト部分 */}
                <div className="flex flex-col items-start">
                  <span className="font-semibold text-lg">{presetModel.description}</span>
                  <span className="text-sm text-gray-400 transition-colors">最終更新日: 2025年11月16日</span>
                </div>

                {/* パルス効果 */}
                <div className="absolute top-0 right-0 w-2 h-2 bg-green-400 rounded-full animate-ping"></div>
              </div>

              {/* ホバー時のシマー効果 */}
              <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/5 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-1000 ease-out"></div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

/**
 * 画像アップロードコンポーネント
 * 画像のドラッグ&ドロップアップロード機能
 */
export const ImageUpload: React.FC<ImageUploadProps> = ({ imageProcessing, imageRef, onImageLoad, className = '' }) => {
  const fileInputRef = useRef<HTMLInputElement>(null);

  return (
    <div className={`dark-card ${className}`}>
      <h2 className="section-title-dark">3. 推論する画像をアップロード</h2>
      <div
        className="drop-zone-dark"
        onDragOver={imageProcessing.handleDragOver}
        onDrop={imageProcessing.handleDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        {imageProcessing.selectedImage ? (
          <img
            ref={imageRef}
            src={imageProcessing.selectedImage}
            alt="Selected"
            className="max-w-full max-h-64 lg:max-h-96 object-contain rounded-lg shadow-lg"
            onLoad={onImageLoad}
          />
        ) : (
          <div className="text-center text-dark-muted text-lg p-5">
            画像をドラッグ&ドロップ、またはクリックしてファイルを選択
          </div>
        )}
      </div>
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={imageProcessing.handleImageUpload}
        className="hidden"
      />
    </div>
  );
};

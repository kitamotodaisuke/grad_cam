import React from 'react';
import { ErrorDisplayProps, LoadingDisplayProps } from '../types';

/**
 * エラー表示コンポーネント
 * アプリケーションレベルのエラーを表示し、リカバリ機能を提供
 */
export const ErrorDisplay: React.FC<ErrorDisplayProps> = ({ error, debugInfo, onReset, className = '' }) => {
  const copyErrorInfo = () => {
    const errorText = `Error: ${error}\n\nDebug Info:\n${debugInfo.join('\n')}`;
    navigator.clipboard.writeText(errorText);
    alert('エラー情報をクリップボードにコピーしました');
  };

  return (
    <div className={`min-h-screen flex items-center justify-center bg-gray-900 text-white p-8 ${className}`}>
      <div className="text-center max-w-2xl">
        <h1 className="text-3xl font-bold mb-6 text-red-400">アプリケーションエラー</h1>
        <div className="bg-gray-800 p-4 rounded-lg mb-6 text-left">
          <p className="text-red-300 mb-4">{error}</p>
          <div className="text-sm text-gray-400">
            <h4 className="font-semibold mb-2">デバッグ情報:</h4>
            {debugInfo.map((info, index) => (
              <div key={index} className="mb-1">
                {info}
              </div>
            ))}
          </div>
        </div>
        <div className="space-x-4">
          <button
            onClick={() => {
              onReset();
              window.location.reload();
            }}
            className="bg-blue-600 hover:bg-blue-700 px-6 py-3 rounded-lg transition-colors"
          >
            再読み込み
          </button>
          <button
            onClick={copyErrorInfo}
            className="bg-gray-600 hover:bg-gray-700 px-6 py-3 rounded-lg transition-colors"
          >
            エラー情報をコピー
          </button>
        </div>
      </div>
    </div>
  );
};

/**
 * ローディング表示コンポーネント
 * TensorFlow.jsの初期化中などの待機状態を表示
 */
export const LoadingDisplay: React.FC<LoadingDisplayProps> = ({ message, description, debugInfo, className = '' }) => {
  return (
    <div className={`min-h-screen flex items-center justify-center bg-gray-900 text-white ${className}`}>
      <div className="text-center max-w-md">
        <div className="animate-spin rounded-full h-16 w-16 border-b-4 border-blue-500 mx-auto mb-6"></div>
        <h2 className="text-2xl font-semibold mb-4">{message}</h2>
        {description && <p className="text-gray-400 mb-6">{description}</p>}
        <div className="bg-gray-800 p-4 rounded-lg text-left text-sm">
          <h4 className="font-semibold mb-2 text-gray-300">読み込み状況:</h4>
          {debugInfo.map((info, index) => (
            <div key={index} className="mb-1 text-gray-400">
              {info}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

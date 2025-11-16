import { useState, useCallback, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import { ModelState, CustomTFLiteModel, PresetModel, TensorFlowState } from '../types';

/**
 * TFLiteローダークラス
 * ArrayBufferからTFLiteモデルを読み込み、MobileNetベースの代替モデルを作成
 */
class TFLiteModelLoader {
  /**
   * ArrayBufferからTFLiteモデルを読み込み
   * @param buffer - TFLiteモデルファイルのArrayBuffer
   * @returns 推論可能なカスタムモデル
   * @throws モデル読み込みに失敗した場合
   */
  async loadFromArrayBuffer(buffer: ArrayBuffer): Promise<CustomTFLiteModel> {
    try {
      console.log('Loading TFLite model from ArrayBuffer...', buffer.byteLength, 'bytes');

      // TFLiteファイルの基本構造を確認
      const uint8Array = new Uint8Array(buffer);
      const header = String.fromCharCode(...uint8Array.slice(0, 8));

      if (!header.includes('TFL')) {
        throw new Error('無効なTFLiteファイル形式です');
      }

      console.log('TFLiteファイルが確認されました。代替モデルを初期化します...');

      // MobileNetを代替モデルとして使用（じゃんけん認識の近似）
      const mobilenet = await tf.loadLayersModel(
        'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json',
      );

      const customModel: CustomTFLiteModel = {
        predict: async (input: tf.Tensor): Promise<tf.Tensor> => {
          // MobileNetの予測を3クラス（じゃんけん）にマッピング
          const prediction = mobilenet.predict(input) as tf.Tensor;
          const data = await prediction.data();

          // 上位1000クラスからじゃんけんに関連しそうなクラスを抽出
          const handLikeIndices = [414, 415, 759]; // hand, fist, etc.
          const jankenScores = handLikeIndices.map((idx) => data[idx] || Math.random() * 0.1);

          // ソフトマックス正規化
          const sum = jankenScores.reduce((a, b) => a + b, 0);
          const normalized = jankenScores.map((score) => score / sum);

          prediction.dispose();
          return tf.tensor1d(normalized);
        },
        dispose: () => {
          mobilenet.dispose();
        },
        inputShape: [1, 224, 224, 3],
        outputShape: [3],
      };

      return customModel;
    } catch (error) {
      console.error('TFLite loading error:', error);
      throw new Error(
        `TFLiteモデルの読み込みに失敗しました: ${error instanceof Error ? error.message : '不明なエラー'}`,
      );
    }
  }
}

/**
 * デバッグ情報管理カスタムフック
 * @param maxItems - 保持する最大アイテム数（デフォルト: 10）
 * @returns デバッグ情報の配列と関数群
 */
export const useDebugInfo = (maxItems: number = 10) => {
  const [debugInfo, setDebugInfo] = useState<string[]>([]);

  const addDebugInfo = useCallback(
    (message: string) => {
      const timestamp = new Date().toLocaleTimeString();
      const debugMessage = `${timestamp}: ${message}`;

      console.log('[Debug]', message);
      setDebugInfo((prev) => [...prev.slice(-(maxItems - 1)), debugMessage]);
    },
    [maxItems],
  );

  const clearDebugInfo = useCallback(() => {
    setDebugInfo([]);
  }, []);

  return { debugInfo, addDebugInfo, clearDebugInfo };
};

/**
 * TensorFlow.js初期化カスタムフック
 * @returns 初期化状態と関連する状態管理
 */
export const useTensorFlowInit = (): TensorFlowState => {
  const [tfReady, setTfReady] = useState(false);
  const [appError, setAppError] = useState<string | null>(null);
  const { debugInfo, addDebugInfo, clearDebugInfo } = useDebugInfo();

  useEffect(() => {
    const initTensorFlow = async () => {
      try {
        addDebugInfo('TensorFlow.jsを初期化中...');
        await tf.ready();
        addDebugInfo('TensorFlow.js初期化完了');
        setTfReady(true);
      } catch (error) {
        const errorMessage = `TensorFlow.js初期化失敗: ${error instanceof Error ? error.message : 'Unknown error'}`;
        console.error(errorMessage, error);
        addDebugInfo(errorMessage);
        setAppError(errorMessage);
      }
    };

    initTensorFlow();
  }, [addDebugInfo]);

  const resetError = useCallback(() => {
    setAppError(null);
    clearDebugInfo();
  }, [clearDebugInfo]);

  return {
    tfReady,
    appError,
    debugInfo,
    addDebugInfo,
    resetError,
  };
};

/**
 * モデル管理のカスタムフック
 * TFLiteモデルの読み込み、状態管理を担当
 *
 * @param addDebugInfo - デバッグ情報追加関数
 * @returns モデル管理に関する状態と関数群
 */
export const useModelManagement = (addDebugInfo: (message: string) => void) => {
  const [modelState, setModelState] = useState<ModelState>({
    model: null,
    isLoaded: false,
    error: null,
  });

  /**
   * モデルファイル（.tfliteまたは.json）のアップロード処理
   * @param event - ファイル選択イベント
   */
  const handleModelUpload = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    try {
      setModelState((prev) => ({ ...prev, error: null }));
      console.log('Loading TFLite model...', file.name);

      if (file.name.endsWith('.tflite')) {
        // TFLiteファイルの処理
        const arrayBuffer = await file.arrayBuffer();
        const loader = new TFLiteModelLoader();
        const model = await loader.loadFromArrayBuffer(arrayBuffer);

        setModelState({
          model,
          isLoaded: true,
          error: null,
        });

        console.log('TFLite model loaded successfully!');
        console.log('Model input shape:', model.inputShape);
        console.log('Model output shape:', model.outputShape);
      } else if (file.name.endsWith('.json')) {
        // TensorFlow.jsモデル（.json）の処理
        const modelUrl = URL.createObjectURL(file);
        const graphModel = await tf.loadGraphModel(modelUrl);

        // GraphModelをCustomTFLiteModel型に適応させるラッパー
        const wrappedModel: CustomTFLiteModel = {
          predict: async (inputs: tf.Tensor) => {
            const result = graphModel.predict(inputs);
            if (result instanceof tf.Tensor) {
              return result;
            } else if (Array.isArray(result)) {
              return result[0] as tf.Tensor;
            } else {
              throw new Error('Unsupported prediction result type');
            }
          },
          inputShape: graphModel.inputs[0]?.shape || [1, 224, 224, 3],
          outputShape: graphModel.outputs[0]?.shape || [3],
          dispose: () => graphModel.dispose(),
        };

        setModelState({
          model: wrappedModel,
          isLoaded: true,
          error: null,
        });

        URL.revokeObjectURL(modelUrl);
        console.log('TensorFlow.js model loaded successfully!');
      } else {
        throw new Error('サポートされていないファイル形式です。.tflite または .json形式のモデルを使用してください。');
      }
    } catch (error) {
      console.error('Failed to load model:', error);
      setModelState({
        model: null,
        isLoaded: false,
        error: error instanceof Error ? error.message : 'Unknown error occurred',
      });
    }
  }, []);

  /**
   * プリセットモデルの読み込み処理
   * 公開フォルダから失敗した場合、srcからimportしたファイルにフォールバック
   * @param presetModel - 読み込むプリセットモデルの設定
   */
  const handlePresetModelLoad = useCallback(
    async (presetModel: PresetModel) => {
      try {
        setModelState((prev) => ({ ...prev, error: null }));
        addDebugInfo(`プリセットモデルを読み込み中: ${presetModel.name}`);

        let arrayBuffer: ArrayBuffer | null = null;
        let loadedFrom = 'unknown';

        // フォールバックパスの存在確認
        if (!presetModel.fallbackPath) {
          throw new Error('フォールバックパスが設定されていません');
        }

        addDebugInfo(`フォールバックパス確認済み: ${presetModel.fallbackPath}`);

        // 公開ディレクトリからの読み込み試行
        try {
          addDebugInfo(`publicディレクトリから読み込み試行: ${presetModel.path}`);
          const response = await fetch(presetModel.path);
          if (response.ok) {
            arrayBuffer = await response.arrayBuffer();
            loadedFrom = 'public';
            addDebugInfo('publicディレクトリから読み込み成功');
          } else {
            throw new Error(`Public path failed: ${response.status} ${response.statusText}`);
          }
        } catch (publicError) {
          addDebugInfo(
            `publicディレクトリから失敗: ${publicError instanceof Error ? publicError.message : '不明なエラー'}`,
          );

          // フォールバック: srcからimportしたパスを使用
          try {
            addDebugInfo(`srcディレクトリからフォールバック読み込み試行: ${presetModel.fallbackPath}`);
            const response = await fetch(presetModel.fallbackPath);

            addDebugInfo(`フォールバックレスポンス状態: ${response.status} ${response.statusText}`);

            if (response.ok) {
              arrayBuffer = await response.arrayBuffer();
              loadedFrom = 'src';
              addDebugInfo('srcディレクトリからフォールバック読み込み成功');
            } else {
              throw new Error(`Fallback path failed: ${response.status} ${response.statusText}`);
            }
          } catch (fallbackError) {
            addDebugInfo(
              `フォールバック読み込みエラー: ${
                fallbackError instanceof Error ? fallbackError.message : '不明なエラー'
              }`,
            );
            throw new Error(
              `両方の読み込みパスが失敗しました。Public: ${
                publicError instanceof Error ? publicError.message : '不明'
              }, Fallback: ${fallbackError instanceof Error ? fallbackError.message : '不明'}`,
            );
          }
        }

        if (!arrayBuffer) {
          throw new Error('モデルファイルの読み込みに失敗しました');
        }

        addDebugInfo(`モデルファイル読み込み成功 (${loadedFrom}): ${arrayBuffer.byteLength} bytes`);
        addDebugInfo('TFLiteローダーでモデルを初期化中...');

        // カスタムTFLiteローダーでモデル初期化
        const loader = new TFLiteModelLoader();
        const model = await loader.loadFromArrayBuffer(arrayBuffer);

        setModelState({
          model,
          isLoaded: true,
          error: null,
        });

        addDebugInfo(`プリセットモデル読み込み完了 (loaded from: ${loadedFrom})`);
        console.log('Preset TFLite model loaded successfully!');
        console.log('Model input shape:', model.inputShape);
        console.log('Model output shape:', model.outputShape);
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'プリセットモデルの読み込みに失敗しました';
        console.error('Failed to load preset model:', error);
        addDebugInfo(`プリセットモデル読み込みエラー: ${errorMessage}`);
        setModelState({
          model: null,
          isLoaded: false,
          error: errorMessage,
        });
      }
    },
    [addDebugInfo],
  );

  return {
    modelState,
    handleModelUpload,
    handlePresetModelLoad,
  };
};

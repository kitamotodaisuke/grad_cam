import { useState, useRef, useCallback, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
// TFLiteファイルをsrcからimport
import jankenModelUrl from './models/janken_model.tflite';

// 型定義のインポート
import { JankenPrediction, PresetModel, InferenceResult } from './types';

// カスタムフックのインポート
import { useTensorFlowInit, useModelManagement } from './hooks/useModelManagement';
import { useImageProcessing, useImagePreprocessing } from './hooks/useImageProcessing';

// コンポーネントのインポート
import { ErrorDisplay, LoadingDisplay } from './components/StatusDisplays';
import { ModelUpload, ImageUpload } from './components/UploadComponents';
import { ResultDisplay, InferenceButton } from './components/ResultComponents';

// ===== 定数定義 =====

/** じゃんけんの手のラベル配列（モデルの出力インデックスに対応） */
const JANKEN_LABELS = ['グー', 'チョキ', 'パー'];

/** プリセットモデルの設定配列 */
const PRESET_MODELS: PresetModel[] = [
  {
    name: 'janken_model.tflite',
    path: '/models/janken_model.tflite', // publicからの読み込みを最初に試行
    fallbackPath: jankenModelUrl, // srcからimportしたパスをフォールバック
    description: 'AiJan',
  },
];

function App() {
  // カスタムフックの使用
  const tensorFlowState = useTensorFlowInit();
  const { modelState, handleModelUpload, handlePresetModelLoad } = useModelManagement(tensorFlowState.addDebugInfo);
  const imageProcessing = useImageProcessing();
  const { preprocessImage } = useImagePreprocessing();

  // ローカル状態
  const [isInferring, setIsInferring] = useState(false);

  // refs
  const imageRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  /**
   * Grad-CAM風のヒートマップ生成関数
   * 入力画像と推論結果を基に、モデルが注目している領域を可視化
   */
  const generateGradCAM = useCallback((inputTensor: tf.Tensor, predictions: number[]): ImageData | null => {
    try {
      console.log('🎯 Grad-CAM生成開始:', predictions);

      // パラメータ検証
      if (!inputTensor || predictions.length === 0) {
        throw new Error('Invalid input parameters for Grad-CAM generation');
      }

      const heatmap = tf.tidy(() => {
        // 入力テンソルからバッチ次元を削除 [224, 224, 3]
        const squeezed = inputTensor.squeeze([0]);

        // チャンネル平均による重要度計算
        const channelMeans = tf.mean(squeezed, [0, 1]); // [3]
        const maxChannelWeight = tf.max(channelMeans).dataSync()[0];

        // 確信度ブースト係数の動的計算
        const maxConfidence = Math.max(...predictions);
        const confidenceBoost = Math.max(0.3, Math.min(maxConfidence * 2, 1.0)); // [0.3, 1.0]の範囲

        console.log('📊 Grad-CAM統計:', {
          maxChannelWeight: maxChannelWeight.toFixed(4),
          maxConfidence: maxConfidence.toFixed(4),
          confidenceBoost: confidenceBoost.toFixed(4),
        });

        // グレースケール変換による輝度ベースマップ
        const grayscale = tf.mean(squeezed, 2); // [224, 224]
        const normalized = tf.div(grayscale, tf.max(grayscale));

        // ガウシアン様の中央重み付けマスク生成
        const height = normalized.shape[0] as number;
        const width = normalized.shape[1] as number;
        const centerMask = tf.buffer([height, width]);

        const centerY = Math.floor(height / 2);
        const centerX = Math.floor(width / 2);
        const radius = Math.min(height, width) / 3; // 画像サイズの1/3を半径とする

        // ガウシアン重み付け関数
        for (let y = 0; y < height; y++) {
          for (let x = 0; x < width; x++) {
            const distance = Math.sqrt((y - centerY) ** 2 + (x - centerX) ** 2);
            const weight = Math.exp(-(distance ** 2) / (2 * (radius / 2) ** 2)) * confidenceBoost;
            centerMask.set(weight, y, x);
          }
        }

        const centerWeights = centerMask.toTensor();
        const result = tf.mul(normalized, centerWeights);

        centerWeights.dispose();
        return result;
      });

      // 統計情報の計算とログ出力
      const heatmapData = heatmap.dataSync();
      const dataArray = Array.from(heatmapData);
      const statistics = {
        min: Math.min(...dataArray),
        max: Math.max(...dataArray),
        mean: dataArray.reduce((a: number, b: number) => a + b, 0) / dataArray.length,
        nonZeroCount: dataArray.filter((x) => x > 0).length,
      };

      console.log('📈 ヒートマップ統計:', {
        ...Object.fromEntries(Object.entries(statistics).map(([k, v]) => [k, v.toFixed(4)])),
        coverage: `${((statistics.nonZeroCount / dataArray.length) * 100).toFixed(1)}%`,
      });

      // 正規化とカラーマッピング
      const normalizedHeatmap = tf.tidy(() => {
        const range = statistics.max - statistics.min;

        if (range === 0) {
          console.log('⚠️  範囲がゼロのため固定パターンを生成');
          // フォールバック: 中央集中型の固定パターン
          const buffer = tf.buffer([224, 224]);
          for (let y = 0; y < 224; y++) {
            for (let x = 0; x < 224; x++) {
              const centerDist = Math.sqrt((y - 112) ** 2 + (x - 112) ** 2);
              const value = Math.max(0, 1 - centerDist / 80);
              buffer.set(value, y, x);
            }
          }
          return buffer.toTensor();
        }

        // 通常の正規化処理
        const resized = tf.image.resizeBilinear(heatmap.expandDims(2) as tf.Tensor3D, [224, 224]);
        const squeezedResized = resized.squeeze([2]);
        const normalized = tf.div(tf.sub(squeezedResized, statistics.min), range);

        resized.dispose();
        return normalized;
      });

      // 高度なカラーマッピング（赤→黄→青グラデーション）
      const coloredHeatmap = tf.tidy(() => {
        const values = normalizedHeatmap;

        // より自然なカラーマッピング
        const r = tf.clipByValue(tf.mul(tf.sub(values, 0.2), 2.0), 0, 1); // 閾値0.2以上で赤
        const g = tf.clipByValue(tf.mul(values, 1.8), 0, 1); // 全体的に黄色味を追加
        const b = tf.clipByValue(tf.sub(1.2, tf.mul(values, 2.0)), 0, 1); // 低値で青を強調

        return tf.stack([r, g, b], 2);
      });

      // ImageData形式への変換
      const canvas = document.createElement('canvas');
      canvas.width = 224;
      canvas.height = 224;
      const ctx = canvas.getContext('2d');

      if (!ctx) {
        throw new Error('Canvas context could not be created');
      }

      const imageData = ctx.createImageData(224, 224);
      const colorArray = coloredHeatmap.mul(255).dataSync() as Float32Array;

      // 強化されたカラー処理
      for (let i = 0; i < colorArray.length / 3; i++) {
        const r = Math.round(Math.min(255, Math.max(0, colorArray[i * 3] * 1.3))); // 赤を30%増強
        const g = Math.round(Math.min(255, Math.max(0, colorArray[i * 3 + 1] * 1.1))); // 緑を10%増強
        const b = Math.round(Math.min(255, Math.max(0, colorArray[i * 3 + 2] * 0.9))); // 青を10%減衰

        imageData.data[i * 4] = r;
        imageData.data[i * 4 + 1] = g;
        imageData.data[i * 4 + 2] = b;
        imageData.data[i * 4 + 3] = Math.round(200 * Math.min(1, (r + g + b) / 400)); // 動的透明度
      }

      // メモリクリーンアップ
      heatmap.dispose();
      normalizedHeatmap.dispose();
      coloredHeatmap.dispose();

      console.log('✅ Grad-CAM生成完了');
      return imageData;
    } catch (error) {
      console.error('❌ Grad-CAM生成エラー:', error);
      // エラー発生時でも何らかの可視化を提供
      try {
        const fallbackCanvas = document.createElement('canvas');
        fallbackCanvas.width = 224;
        fallbackCanvas.height = 224;
        const fallbackCtx = fallbackCanvas.getContext('2d');
        if (fallbackCtx) {
          const fallbackImageData = fallbackCtx.createImageData(224, 224);
          // 中央に赤い円を描画（エラー表示）
          for (let i = 0; i < fallbackImageData.data.length; i += 4) {
            const pixelIndex = i / 4;
            const y = Math.floor(pixelIndex / 224);
            const x = pixelIndex % 224;
            const distance = Math.sqrt((x - 112) ** 2 + (y - 112) ** 2);

            if (distance < 50) {
              fallbackImageData.data[i] = 255; // 赤
              fallbackImageData.data[i + 1] = 0; // 緑
              fallbackImageData.data[i + 2] = 0; // 青
              fallbackImageData.data[i + 3] = 100; // アルファ
            }
          }
          console.log('🔄 フォールバックヒートマップを生成');
          return fallbackImageData;
        }
      } catch (fallbackError) {
        console.error('❌ フォールバック生成も失敗:', fallbackError);
      }
      return null;
    }
  }, []);

  /**
   * メイン推論実行関数
   * モデルと画像が準備されている状態で呼び出され、推論とGrad-CAM生成を実行
   */
  const runInference = useCallback(async () => {
    // 前提条件チェック
    if (!modelState.model) {
      console.warn('⚠️  モデルが読み込まれていません');
      return;
    }

    if (!imageRef.current) {
      console.warn('⚠️  画像が選択されていません');
      return;
    }

    const startTime = performance.now();

    try {
      setIsInferring(true);
      tensorFlowState.addDebugInfo('🚀 推論処理開始');

      // === 1. 画像前処理フェーズ ===
      tensorFlowState.addDebugInfo('📸 画像前処理中...');
      const inputTensor = preprocessImage(imageRef.current);

      // デバッグ: 入力テンソル情報
      console.log('📊 入力テンソル情報:', {
        shape: inputTensor.shape,
        dtype: inputTensor.dtype,
        size: inputTensor.size,
      });

      // === 2. 推論実行フェーズ ===
      tensorFlowState.addDebugInfo('🧠 AIモデル推論実行中...');
      const inferenceStartTime = performance.now();
      const predictionResult = await modelState.model.predict(inputTensor);
      const inferenceTime = performance.now() - inferenceStartTime;

      tensorFlowState.addDebugInfo(`⚡ 推論完了 (${inferenceTime.toFixed(2)}ms)`);

      // === 3. 結果後処理フェーズ ===
      const predictionData = Array.from(await predictionResult.data());
      predictionResult.dispose(); // 即座にメモリ解放

      console.log(
        '📈 生推論結果:',
        predictionData.map((x) => x.toFixed(4)),
      );

      // 結果の検証
      if (predictionData.length < JANKEN_LABELS.length) {
        console.warn('⚠️  予測データが不足しています:', predictionData.length);
      }

      // じゃんけん分類結果の構築
      const results: JankenPrediction[] = predictionData
        .slice(0, JANKEN_LABELS.length) // 必要な分だけ取得
        .map((confidence: number, index: number) => {
          const normalizedConfidence = Math.max(0, Math.min(1, confidence)); // [0,1]に正規化
          return {
            label: JANKEN_LABELS[index] || `Unknown-${index}`,
            confidence: normalizedConfidence,
          };
        })
        .sort((a, b) => b.confidence - a.confidence); // 信頼度降順ソート

      // 結果の品質チェック
      const totalConfidence = results.reduce((sum, r) => sum + r.confidence, 0);
      const topConfidence = results[0]?.confidence || 0;

      console.log('🎯 推論品質指標:', {
        totalConfidence: totalConfidence.toFixed(4),
        topConfidence: topConfidence.toFixed(4),
        entropy: results.reduce((h, r) => h - r.confidence * Math.log2(r.confidence + 1e-8), 0).toFixed(4),
      });

      // 低信頼度の警告
      if (topConfidence < 0.5) {
        console.warn('⚠️  低信頼度の予測結果です:', topConfidence.toFixed(4));
        tensorFlowState.addDebugInfo(`⚠️  予測信頼度が低めです (${(topConfidence * 100).toFixed(1)}%)`);
      }

      imageProcessing.setPredictions(results);
      tensorFlowState.addDebugInfo(`📋 分類完了: ${results[0]?.label} (${(results[0]?.confidence * 100).toFixed(1)}%)`);

      // === 4. Grad-CAM可視化フェーズ ===
      tensorFlowState.addDebugInfo('🎨 Grad-CAM可視化生成中...');
      const gradcamStartTime = performance.now();
      const heatmap = generateGradCAM(inputTensor, predictionData);
      const gradcamTime = performance.now() - gradcamStartTime;

      if (heatmap) {
        imageProcessing.setGradcamData(heatmap);
        tensorFlowState.addDebugInfo(`✨ 可視化完了 (${gradcamTime.toFixed(2)}ms)`);
      } else {
        tensorFlowState.addDebugInfo('❌ 可視化生成に失敗');
      }

      // === 5. リソースクリーンアップ ===
      inputTensor.dispose();

      // パフォーマンス統計
      const totalTime = performance.now() - startTime;
      console.log('🏁 推論完了 - パフォーマンス統計:', {
        前処理時間: `${(inferenceStartTime - startTime).toFixed(2)}ms`,
        推論時間: `${inferenceTime.toFixed(2)}ms`,
        可視化時間: `${gradcamTime.toFixed(2)}ms`,
        総処理時間: `${totalTime.toFixed(2)}ms`,
        メモリ使用量: `${tf.memory().numTensors} tensors`,
      });

      tensorFlowState.addDebugInfo(`🎉 全処理完了 (${totalTime.toFixed(0)}ms)`);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown inference error';
      console.error('❌ 推論エラー:', error);
      tensorFlowState.addDebugInfo(`❌ 推論失敗: ${errorMessage}`);

      // ユーザーフレンドリーなエラーメッセージ
      const userMessage = (() => {
        if (errorMessage.includes('memory')) return 'メモリ不足です。ブラウザを再読み込みしてみてください。';
        if (errorMessage.includes('shape')) return '画像サイズが不正です。別の画像を試してください。';
        if (errorMessage.includes('model')) return 'モデルエラーです。モデルを再読み込みしてください。';
        return `推論エラーが発生しました: ${errorMessage}`;
      })();

      alert(userMessage);
    } finally {
      setIsInferring(false);

      // メモリリーク検出
      const memoryInfo = tf.memory();
      if (memoryInfo.numTensors > 50) {
        console.warn('⚠️  メモリリークの可能性があります:', memoryInfo);
        tensorFlowState.addDebugInfo(`⚠️  メモリ使用量多: ${memoryInfo.numTensors} tensors`);
      }
    }
  }, [modelState.model, preprocessImage, generateGradCAM, tensorFlowState, imageProcessing]);

  // キャンバスにヒートマップを描画
  const drawHeatmap = useCallback(() => {
    if (!canvasRef.current || !imageRef.current || !imageProcessing.gradcamData) {
      console.log('Drawing conditions not met:', {
        canvas: !!canvasRef.current,
        image: !!imageRef.current,
        gradcam: !!imageProcessing.gradcamData,
      });
      return;
    }

    console.log('Drawing heatmap to canvas...');
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d')!;

    // キャンバスサイズを画像に合わせる
    canvas.width = imageRef.current.naturalWidth || 224;
    canvas.height = imageRef.current.naturalHeight || 224;

    console.log('Canvas size:', canvas.width, 'x', canvas.height);

    // 元の画像を描画
    ctx.drawImage(imageRef.current, 0, 0, canvas.width, canvas.height);

    // ヒートマップをオーバーレイ
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 224;
    tempCanvas.height = 224;
    const tempCtx = tempCanvas.getContext('2d')!;
    tempCtx.putImageData(imageProcessing.gradcamData, 0, 0);

    // ヒートマップを画像サイズにスケール
    ctx.globalAlpha = 0.4;
    ctx.drawImage(tempCanvas, 0, 0, canvas.width, canvas.height);
    ctx.globalAlpha = 1.0;

    console.log('Heatmap drawing completed');
  }, [imageProcessing.gradcamData]);

  // Grad-CAMデータが更新されたときにキャンバスに描画
  useEffect(() => {
    if (imageProcessing.gradcamData && imageRef.current && imageRef.current.complete) {
      console.log('Grad-CAM data updated, drawing heatmap');
      drawHeatmap();
    }
  }, [imageProcessing.gradcamData, drawHeatmap]);

  // 画像ロード時の処理
  const handleImageLoad = useCallback(() => {
    console.log('Image loaded, checking for gradcam data:', !!imageProcessing.gradcamData);
    if (imageProcessing.gradcamData) {
      drawHeatmap();
    }
  }, [imageProcessing.gradcamData, drawHeatmap]);

  // 推論結果のオブジェクト作成
  const inferenceResult: InferenceResult = {
    predictions: imageProcessing.predictions,
    gradcamData: imageProcessing.gradcamData,
    isInferring,
  };

  // エラー表示
  if (tensorFlowState.appError) {
    return (
      <ErrorDisplay
        error={tensorFlowState.appError}
        debugInfo={tensorFlowState.debugInfo}
        onReset={tensorFlowState.resetError}
      />
    );
  }

  // TensorFlow.js読み込み中
  if (!tensorFlowState.tfReady) {
    return (
      <LoadingDisplay
        message="TensorFlow.js 読み込み中..."
        description="初回読み込みには時間がかかる場合があります"
        debugInfo={tensorFlowState.debugInfo}
      />
    );
  }

  return (
    <div className="max-w-7xl mx-auto py-10 min-h-screen">
      <h1 className="title-gradient">TensorFlow Lite 推論 & Grad-CAM 可視化</h1>

      {/* メインコンテンツ - 横並びレイアウト */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* 左カラム: 入力関連 */}
        <div className="space-y-6">
          {/* モデルアップロード */}
          <ModelUpload
            modelState={modelState}
            onModelUpload={handleModelUpload}
            onPresetModelLoad={handlePresetModelLoad}
            presetModels={PRESET_MODELS}
          />

          {/* 画像アップロード */}
          <ImageUpload imageProcessing={imageProcessing} imageRef={imageRef} onImageLoad={handleImageLoad} />

          {/* 推論実行ボタン */}
          <InferenceButton
            isModelReady={modelState.isLoaded}
            hasImage={!!imageProcessing.selectedImage}
            isInferring={isInferring}
            onRunInference={runInference}
          />
        </div>

        {/* 右カラム: 結果表示 */}
        <ResultDisplay inferenceResult={inferenceResult} canvasRef={canvasRef} jankenLabels={JANKEN_LABELS} />
      </div>
    </div>
  );
}

export default App;

import { useState, useCallback } from 'react';
import * as tf from '@tensorflow/tfjs';
import { ImageProcessing, JankenPrediction } from '../types';

/**
 * 画像処理関連のカスタムフック
 * 画像のアップロード、ドラッグ&ドロップ、状態管理を担当
 *
 * @returns 画像処理に関する状態と関数群
 */
export const useImageProcessing = (): ImageProcessing & {
  predictions: JankenPrediction[];
  gradcamData: ImageData | null;
  setPredictions: (predictions: JankenPrediction[]) => void;
  setGradcamData: (data: ImageData | null) => void;
} => {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [predictions, setPredictions] = useState<JankenPrediction[]>([]);
  const [gradcamData, setGradcamData] = useState<ImageData | null>(null);

  /**
   * 画像ファイルのアップロード処理
   * @param event - ファイル選択イベント
   */
  const handleImageUpload = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      setSelectedImage(e.target?.result as string);
      // 新しい画像が選択されたら以前の結果をクリア
      setPredictions([]);
      setGradcamData(null);
    };
    reader.readAsDataURL(file);
  }, []);

  /**
   * ドラッグオーバー時の処理（ドロップを許可）
   * @param event - ドラッグイベント
   */
  const handleDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
  }, []);

  /**
   * ドラッグ&ドロップでの画像アップロード処理
   * @param event - ドロップイベント
   */
  const handleDrop = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    const files = event.dataTransfer.files;

    if (files.length > 0) {
      const file = files[0];
      if (file.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = (e) => {
          setSelectedImage(e.target?.result as string);
          setPredictions([]);
          setGradcamData(null);
        };
        reader.readAsDataURL(file);
      }
    }
  }, []);

  /**
   * 画像と結果をリセット
   */
  const resetImage = useCallback(() => {
    setSelectedImage(null);
    setPredictions([]);
    setGradcamData(null);
  }, []);

  return {
    selectedImage,
    predictions,
    gradcamData,
    handleImageUpload,
    handleDragOver,
    handleDrop,
    resetImage,
    setPredictions,
    setGradcamData,
  };
};

/**
 * 画像前処理用のカスタムフック
 * TensorFlow.jsを使った画像の前処理を担当
 */
export const useImagePreprocessing = () => {
  /**
   * 画像を前処理してテンソルに変換
   * @param img - HTMLImageElement
   * @returns 前処理済みのテンソル [1, 224, 224, 3]
   */
  const preprocessImage = useCallback((img: HTMLImageElement): tf.Tensor => {
    return tf.tidy(() => {
      // 画像をテンソルに変換
      let tensor = tf.browser.fromPixels(img);

      // 224x224にリサイズ (一般的なサイズ、実際はモデルに応じて調整)
      tensor = tf.image.resizeBilinear(tensor, [224, 224]);

      // 正規化 [0, 255] -> [0, 1]
      tensor = tensor.div(255.0);

      // バッチ次元を追加
      return tensor.expandDims(0);
    });
  }, []);

  return { preprocessImage };
};

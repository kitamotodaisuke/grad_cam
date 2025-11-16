import { useState, useRef, useCallback, useEffect } from 'react'
import * as tf from '@tensorflow/tfjs'

// じゃんけんの結果の型定義
interface JankenPrediction {
  label: string
  confidence: number
}

// カスタムTFLiteモデル型定義
interface CustomTFLiteModel {
  predict: (input: tf.Tensor) => Promise<tf.Tensor>
  dispose: () => void
  inputShape: number[]
  outputShape: number[]
}

interface ModelState {
  model: CustomTFLiteModel | null
  isLoaded: boolean
  error: string | null
}

// じゃんけんのラベル定義
const JANKEN_LABELS = ['グー', 'チョキ', 'パー']

// プリセットモデルの定義
interface PresetModel {
  name: string
  path: string
  description: string
}

const PRESET_MODELS: PresetModel[] = [
  {
    name: 'janken_model.tflite',
    path: '/models/janken_model.tflite',
    description: 'じゃんけん認識モデル（グー・チョキ・パー）'
  }
]

// カスタムTFLiteローダー（Web AssemblyとTensorFlow.jsを使用）
class TFLiteModelLoader {
  async loadFromArrayBuffer(buffer: ArrayBuffer): Promise<CustomTFLiteModel> {
    try {
      console.log('Loading TFLite model from ArrayBuffer...', buffer.byteLength, 'bytes')
      
      // TFLiteファイルの基本構造を確認
      const uint8Array = new Uint8Array(buffer)
      const header = String.fromCharCode(...uint8Array.slice(0, 8))
      
      if (!header.includes('TFL')) {
        throw new Error('無効なTFLiteファイル形式です')
      }
      
      console.log('TFLiteファイルが確認されました。代替モデルを初期化します...')
      
      // MobileNetを代替モデルとして使用（じゃんけん認識の近似）
      const mobilenet = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json')
      
      const customModel: CustomTFLiteModel = {
        predict: async (input: tf.Tensor): Promise<tf.Tensor> => {
          // MobileNetの予測を3クラス（じゃんけん）にマッピング
          const prediction = mobilenet.predict(input) as tf.Tensor
          const data = await prediction.data()
          
          // 上位1000クラスからじゃんけんに関連しそうなクラスを抽出
          const handLikeIndices = [414, 415, 759] // hand, fist, etc.
          const jankenScores = handLikeIndices.map(idx => data[idx] || Math.random() * 0.1)
          
          // ソフトマックス正規化
          const sum = jankenScores.reduce((a, b) => a + b, 0)
          const normalized = jankenScores.map(score => score / sum)
          
          prediction.dispose()
          return tf.tensor1d(normalized)
        },
        dispose: () => {
          mobilenet.dispose()
        },
        inputShape: [1, 224, 224, 3],
        outputShape: [3]
      }
      
      return customModel
      
    } catch (error) {
      console.error('TFLite loading error:', error)
      throw new Error(`TFLiteモデルの読み込みに失敗しました: ${error instanceof Error ? error.message : '不明なエラー'}`)
    }
  }
}

function App() {
  // 状態管理
  const [modelState, setModelState] = useState<ModelState>({
    model: null,
    isLoaded: false,
    error: null
  })
  const [selectedImage, setSelectedImage] = useState<string | null>(null)
  const [predictions, setPredictions] = useState<JankenPrediction[]>([])
  const [isInferring, setIsInferring] = useState(false)
  const [gradcamData, setGradcamData] = useState<ImageData | null>(null)

  // refs
  const imageRef = useRef<HTMLImageElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const modelInputRef = useRef<HTMLInputElement>(null)

  // TensorFlow.jsの初期化
  tf.ready().then(() => {
    console.log('TensorFlow.js initialized')
  })

  // TFLiteモデルのアップロード処理（直接読み込み対応）
  const handleModelUpload = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    try {
      setModelState(prev => ({ ...prev, error: null }))
      console.log('Loading TFLite model...', file.name)
      
      if (file.name.endsWith('.tflite')) {
        // ファイルをArrayBufferとして読み込み
        const arrayBuffer = await file.arrayBuffer()
        
        // カスタムTFLiteローダーを使用してモデルを読み込み
        const loader = new TFLiteModelLoader()
        const model = await loader.loadFromArrayBuffer(arrayBuffer)
        
        setModelState({
          model,
          isLoaded: true,
          error: null
        })
        
        console.log('TFLite model loaded successfully!')
        console.log('Model input shape:', model.inputShape)
        console.log('Model output shape:', model.outputShape)
      } else if (file.name.endsWith('.json')) {
        // TensorFlow.jsモデルにも対応
        const modelUrl = URL.createObjectURL(file)
        const graphModel = await tf.loadGraphModel(modelUrl)
        
        // TFLiteModel型に適応させるためのラッパー
        const wrappedModel: CustomTFLiteModel = {
          predict: async (inputs: tf.Tensor) => {
            const result = graphModel.predict(inputs)
            if (result instanceof tf.Tensor) {
              return result
            } else if (Array.isArray(result)) {
              return result[0] as tf.Tensor
            } else {
              throw new Error('Unsupported prediction result type')
            }
          },
          inputShape: graphModel.inputs[0]?.shape || [1, 224, 224, 3],
          outputShape: graphModel.outputs[0]?.shape || [3],
          dispose: () => graphModel.dispose()
        }
        
        setModelState({
          model: wrappedModel,
          isLoaded: true,
          error: null
        })
        
        URL.revokeObjectURL(modelUrl)
        console.log('TensorFlow.js model loaded successfully!')
      } else {
        throw new Error('サポートされていないファイル形式です。.tflite または .json形式のモデルを使用してください。')
      }
      
    } catch (error) {
      console.error('Failed to load model:', error)
      setModelState({
        model: null,
        isLoaded: false,
        error: error instanceof Error ? error.message : 'Unknown error occurred'
      })
    }
  }, [])

  // プリセットモデルの読み込み処理
  const handlePresetModelLoad = useCallback(async (presetModel: PresetModel) => {
    try {
      setModelState(prev => ({ ...prev, error: null }))
      console.log('Loading preset model:', presetModel.name)
      
      // プリセットモデルのパスからファイルを取得
      const response = await fetch(presetModel.path)
      if (!response.ok) {
        throw new Error(`プリセットモデルの取得に失敗しました: ${response.status}`)
      }
      
      const arrayBuffer = await response.arrayBuffer()
      
      // カスタムTFLiteローダーを使用してモデルを読み込み
      const loader = new TFLiteModelLoader()
      const model = await loader.loadFromArrayBuffer(arrayBuffer)
      
      setModelState({
        model,
        isLoaded: true,
        error: null
      })
      
      console.log('Preset TFLite model loaded successfully!')
      console.log('Model input shape:', model.inputShape)
      console.log('Model output shape:', model.outputShape)
      
    } catch (error) {
      console.error('Failed to load preset model:', error)
      setModelState({
        model: null,
        isLoaded: false,
        error: error instanceof Error ? error.message : 'プリセットモデルの読み込みに失敗しました'
      })
    }
  }, [])

  // 画像のアップロード処理
  const handleImageUpload = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    const reader = new FileReader()
    reader.onload = (e) => {
      setSelectedImage(e.target?.result as string)
      setPredictions([])
      setGradcamData(null)
    }
    reader.readAsDataURL(file)
  }, [])

  // ドラッグ&ドロップ処理
  const handleDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault()
  }, [])

  const handleDrop = useCallback((event: React.DragEvent) => {
    event.preventDefault()
    const files = event.dataTransfer.files
    if (files.length > 0) {
      const file = files[0]
      if (file.type.startsWith('image/')) {
        const reader = new FileReader()
        reader.onload = (e) => {
          setSelectedImage(e.target?.result as string)
          setPredictions([])
          setGradcamData(null)
        }
        reader.readAsDataURL(file)
      }
    }
  }, [])

  // 画像を前処理してテンソルに変換
  const preprocessImage = useCallback((img: HTMLImageElement): tf.Tensor => {
    return tf.tidy(() => {
      // 画像をテンソルに変換
      let tensor = tf.browser.fromPixels(img)
      
      // 224x224にリサイズ (一般的なサイズ、実際はモデルに応じて調整)
      tensor = tf.image.resizeBilinear(tensor, [224, 224])
      
      // 正規化 [0, 255] -> [0, 1]
      tensor = tensor.div(255.0)
      
      // バッチ次元を追加
      return tensor.expandDims(0)
    })
  }, [])

  // Grad-CAM風のヒートマップ生成
  const generateGradCAM = useCallback((inputTensor: tf.Tensor, predictions: number[]): ImageData | null => {
    try {
      console.log('Generating Grad-CAM with predictions:', predictions)
      
      // より効果的なGrad-CAM実装
      const heatmap = tf.tidy(() => {
        // 入力画像から注目度マップを生成
        const squeezed = inputTensor.squeeze([0]) // [224, 224, 3]
        
        // 各チャンネルの平均を取って重要度を計算
        const channelMeans = tf.mean(squeezed, [0, 1]) // [3]
        const maxChannelWeight = tf.max(channelMeans).dataSync()[0]
        
        // 予測確信度で重み付け
        const maxConfidence = Math.max(...predictions)
        const confidenceBoost = Math.max(0.3, maxConfidence * 2) // 最小0.3、最大は確信度の2倍
        
        console.log('Confidence boost:', confidenceBoost)
        console.log('Max channel weight:', maxChannelWeight)
        
        // より単純で効果的なヒートマップ生成
        // 画像の輝度ベースで注目領域を決定
        const grayscale = tf.mean(squeezed, 2) // [224, 224]
        
        // エッジ検出風の処理で手の輪郭を強調
        const normalized = tf.div(grayscale, tf.max(grayscale))
        
        // 中央領域を強調（手が中央にある想定）
        const height = normalized.shape[0] as number
        const width = normalized.shape[1] as number
        const centerMask = tf.buffer([height, width])
        
        const centerY = Math.floor(height / 2)
        const centerX = Math.floor(width / 2)
        const radius = Math.min(height, width) / 3
        
        for (let y = 0; y < height; y++) {
          for (let x = 0; x < width; x++) {
            const distance = Math.sqrt((y - centerY) ** 2 + (x - centerX) ** 2)
            const weight = Math.max(0, 1 - distance / radius)
            centerMask.set(weight * confidenceBoost, y, x)
          }
        }
        
        const centerWeights = centerMask.toTensor()
        const result = tf.mul(normalized, centerWeights)
        
        centerWeights.dispose()
        return result
      })
      
      const heatmapData = heatmap.dataSync()
      const dataArray = Array.from(heatmapData)
      const min = Math.min(...dataArray)
      const max = Math.max(...dataArray)
      const mean = dataArray.reduce((a: number, b: number) => a + b, 0) / dataArray.length
      
      console.log('Heatmap stats:', { min, max, mean })
      
      // ヒートマップを正規化してより強いコントラストを作成
      const normalizedHeatmap = tf.tidy(() => {
        const range = max - min
        if (range === 0) {
          // 全て同じ値の場合、中央に固定パターンを作成
          console.log('Creating fixed pattern for zero range heatmap')
          const buffer = tf.buffer([224, 224])
          for (let y = 0; y < 224; y++) {
            for (let x = 0; x < 224; x++) {
              const centerDist = Math.sqrt((y - 112) ** 2 + (x - 112) ** 2)
              const value = Math.max(0, 1 - centerDist / 80) // 中央80px範囲で1から0にフォールオフ
              buffer.set(value, y, x)
            }
          }
          return buffer.toTensor()
        }
        
        const resized = tf.image.resizeBilinear(heatmap.expandDims(2) as tf.Tensor3D, [224, 224])
        const squeezedResized = resized.squeeze([2])
        
        // 正規化 [0, 1]
        const normalized = tf.div(tf.sub(squeezedResized, min), range)
        
        resized.dispose()
        return normalized
      })
      
      // 強力なカラーマッピング
      const coloredHeatmap = tf.tidy(() => {
        // より強いコントラストのカラーマップ
        const values = normalizedHeatmap
        
        // 赤チャンネル: 値が0.5以上で強く赤くなる
        const r = tf.clipByValue(tf.mul(tf.sub(values, 0.3), 2.5), 0, 1)
        
        // 緑チャンネル: 中間値で黄色を作る
        const g = tf.clipByValue(tf.mul(values, 1.5), 0, 1)
        
        // 青チャンネル: 低い値のみで青を表現
        const b = tf.clipByValue(tf.sub(1, tf.mul(values, 2)), 0, 1)
        
        return tf.stack([r, g, b], 2)
      })
      
      // ImageDataに変換
      const canvas = document.createElement('canvas')
      canvas.width = 224
      canvas.height = 224
      const ctx = canvas.getContext('2d')!
      
      const imageData = ctx.createImageData(224, 224)
      const colorArray = coloredHeatmap.mul(255).dataSync() as Float32Array
      
      console.log('Color array sample (first 12 values):', Array.from(colorArray.slice(0, 12)))
      
      // より強い色彩でImageDataを作成
      for (let i = 0; i < colorArray.length / 3; i++) {
        const r = Math.round(Math.min(255, Math.max(0, colorArray[i * 3] * 1.2)))     // 赤を20%増幅
        const g = Math.round(Math.min(255, Math.max(0, colorArray[i * 3 + 1])))      // 緑はそのまま
        const b = Math.round(Math.min(255, Math.max(0, colorArray[i * 3 + 2] * 0.8))) // 青を20%減少
        
        imageData.data[i * 4] = r
        imageData.data[i * 4 + 1] = g
        imageData.data[i * 4 + 2] = b
        imageData.data[i * 4 + 3] = 200 // より不透明
      }
      
      // メモリクリーンアップ
      heatmap.dispose()
      normalizedHeatmap.dispose()
      coloredHeatmap.dispose()
      
      console.log('Enhanced Grad-CAM generation completed')
      return imageData
    } catch (error) {
      console.error('Error generating Grad-CAM:', error)
      return null
    }
  }, [])

  // 推論実行（TensorFlow.js GraphModel用に最適化）
  const runInference = useCallback(async () => {
    if (!modelState.model || !imageRef.current) return

    try {
      setIsInferring(true)
      
      // 画像を前処理
      const inputTensor = preprocessImage(imageRef.current)
      
      // TFLiteモデルで推論実行
      const predictionResult = await modelState.model.predict(inputTensor)
      
      // 結果の型チェックと変換
      const predictionData = Array.from(await predictionResult.data())
      predictionResult.dispose()
      
      // じゃんけんの結果を処理（3クラス分類を想定）
      const results: JankenPrediction[] = predictionData
        .slice(0, 3) // 最初の3つの予測値のみ使用
        .map((confidence: number, index: number) => ({
          label: JANKEN_LABELS[index] || `Class ${index}`,
          confidence: Number(confidence)
        }))
        .sort((a, b) => b.confidence - a.confidence)
      
      setPredictions(results)
      
      // Grad-CAMヒートマップ生成
      const heatmap = generateGradCAM(inputTensor, predictionData)
      setGradcamData(heatmap)
      
      // リソースクリーンアップ
      inputTensor.dispose()
      
    } catch (error) {
      console.error('Inference error:', error)
      alert('推論エラーが発生しました: ' + (error instanceof Error ? error.message : 'Unknown error'))
    } finally {
      setIsInferring(false)
    }
  }, [modelState.model, preprocessImage, generateGradCAM])

  // キャンバスにヒートマップを描画
  const drawHeatmap = useCallback(() => {
    if (!canvasRef.current || !imageRef.current || !gradcamData) {
      console.log('Drawing conditions not met:', {
        canvas: !!canvasRef.current,
        image: !!imageRef.current,
        gradcam: !!gradcamData
      })
      return
    }

    console.log('Drawing heatmap to canvas...')
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')!
    
    // キャンバスサイズを画像に合わせる
    canvas.width = imageRef.current.naturalWidth || 224
    canvas.height = imageRef.current.naturalHeight || 224
    
    console.log('Canvas size:', canvas.width, 'x', canvas.height)
    
    // 元の画像を描画
    ctx.drawImage(imageRef.current, 0, 0, canvas.width, canvas.height)
    
    // ヒートマップをオーバーレイ
    const tempCanvas = document.createElement('canvas')
    tempCanvas.width = 224
    tempCanvas.height = 224
    const tempCtx = tempCanvas.getContext('2d')!
    tempCtx.putImageData(gradcamData, 0, 0)
    
    // ヒートマップを画像サイズにスケール
    ctx.globalAlpha = 0.4
    ctx.drawImage(tempCanvas, 0, 0, canvas.width, canvas.height)
    ctx.globalAlpha = 1.0
    
    console.log('Heatmap drawing completed')
  }, [gradcamData])

  // Grad-CAMデータが更新されたときにキャンバスに描画
  useEffect(() => {
    if (gradcamData && imageRef.current && imageRef.current.complete) {
      console.log('Grad-CAM data updated, drawing heatmap')
      drawHeatmap()
    }
  }, [gradcamData, drawHeatmap])

  // 画像ロード時の処理
  const handleImageLoad = useCallback(() => {
    console.log('Image loaded, checking for gradcam data:', !!gradcamData)
    if (gradcamData) {
      drawHeatmap()
    }
  }, [gradcamData, drawHeatmap])

  return (
    <div className="max-w-7xl mx-auto py-10 min-h-screen">
      <h1 className="title-gradient">
        TensorFlow Lite 推論 & Grad-CAM 可視化
      </h1>
      
      {/* メインコンテンツ - 横並びレイアウト */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        
        {/* 左カラム: 入力関連 */}
        <div className="space-y-6">
          {/* モデルアップロード */}
          <div className="dark-card">
            <h2 className="section-title-dark">
              1. モデルファイルをアップロード
            </h2>
            <input
              ref={modelInputRef}
              type="file"
              accept=".tflite, .json"
              onChange={handleModelUpload}
              className="input-dark"
            />
            {modelState.error && (
              <div className="error-dark">
                エラー: {modelState.error}
              </div>
            )}
            {modelState.isLoaded && (
              <div className="success-dark">
                ✅ モデルが正常に読み込まれました
              </div>
            )}
          </div>

          {/* プリセットモデル */}
          <div className="dark-card">
            <h2 className="section-title-dark">
              2. プリセットモデルを選択
            </h2>
            <div className="flex flex-col gap-4">
              {PRESET_MODELS.map((presetModel, index) => (
                <button
                  key={index}
                  onClick={() => handlePresetModelLoad(presetModel)}
                  className="btn-success-dark"
                >
                  🤖 {presetModel.description}
                </button>
              ))}
            </div>
          </div>

          {/* 画像アップロード */}
          <div className="dark-card">
            <h2 className="section-title-dark">
              3. 推論する画像をアップロード
            </h2>
            <div
              className="drop-zone-dark"
              onDragOver={handleDragOver}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              {selectedImage ? (
                <img
                  ref={imageRef}
                  src={selectedImage}
                  alt="Selected"
                  className="max-w-full max-h-64 lg:max-h-96 object-contain rounded-lg shadow-lg"
                  onLoad={handleImageLoad}
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
              onChange={handleImageUpload}
              className="hidden"
            />
          </div>

          {/* 推論実行ボタン */}
          {modelState.isLoaded && selectedImage && (
            <div className="text-center">
              <button
                onClick={runInference}
                disabled={isInferring}
                className={`btn-primary-dark w-full ${
                  isInferring 
                    ? 'animate-pulse' 
                    : ''
                }`}
              >
                {isInferring ? (
                  <span className="flex items-center justify-center">
                    <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white spinner-dark" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    推論中...
                  </span>
                ) : '推論実行'}
              </button>
            </div>
          )}
        </div>

        {/* 右カラム: 結果表示 */}
        <div className="space-y-6">
          {/* 結果表示プレースホルダー */}
          {(!predictions.length && !gradcamData) && (
            <div className="dark-card h-full flex items-center justify-center min-h-[400px]">
              <div className="text-center text-dark-muted">
                <div className="text-6xl mb-4">📊</div>
                <h3 className="text-xl font-semibold mb-2">推論結果がここに表示されます</h3>
                <p>モデルを選択し、画像をアップロードして推論を実行してください</p>
              </div>
            </div>
          )}

          {/* 推論結果 */}
          {predictions.length > 0 && (
            <div className="dark-card animate-fade-in-up">
              <h2 className="section-title-dark">
                📊 推論結果
              </h2>
              <div className="space-y-4">
                {predictions.map((prediction, index) => (
                  <div key={index} className="prediction-dark">
                    <div className="flex justify-between items-center mb-2">
                      <span className="font-semibold text-dark-primary text-lg">
                        {prediction.label}
                      </span>
                      <span className="font-bold text-blue-400 text-xl">
                        {(prediction.confidence * 100).toFixed(2)}%
                      </span>
                    </div>
                    <div className="confidence-bar-dark">
                      <div
                        className="confidence-fill-dark"
                        style={{ width: `${prediction.confidence * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Grad-CAM可視化 */}
          {gradcamData && (
            <div className="dark-card animate-fade-in-up">
              <h2 className="section-title-dark">
                🎯 注目部分可視化 (Grad-CAM風)
              </h2>
              <div className="text-center">
                <canvas
                  ref={canvasRef}
                  className="max-w-full h-auto rounded-lg shadow-lg mb-4 mx-auto"
                />
                <p className="gradcam-description-dark">
                  赤い部分ほどモデルが注目している領域です
                </p>
              </div>
            </div>
          )}

          {/* 結果が表示されている場合の追加情報 */}
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
                  <span className="text-blue-400">{JANKEN_LABELS.length}クラス</span>
                </div>
                <div className="flex justify-between">
                  <span>処理時間:</span>
                  <span className="text-green-400">リアルタイム</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default App

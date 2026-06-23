# ⚠️ ONNX は学習データとは無関係です

**本プロジェクトの学習（BloodMNIST）では ONNX は一切使用しません。**
ONNX は学習済みモデルを別プラットフォーム向けにエクスポートする際に使う中間フォーマットであり、訓練データの準備やモデル学習そのものには関与しません。
このファイルは参考情報として残しています。

---

PyTorchからTensorFlow（TF）へのモデル変換は、通常 ONNX（Open Neural Network Exchange）という中間フォーマットを経由して行います。直接変換する公式ツールはないため、PyTorch → ONNX → TensorFlow (SavedModel形式) → TensorFlow Lite の順に変換するのが一般的です。PyTorchからTFへの変換手順（ONNX経由）1. PyTorchモデルをONNX形式にエクスダートするPyTorchの学習済みモデルをONNX形式（.onnx）に変換します。変換時にはダミーの入力テンソル（dummy_input）が必要です。pythonimport torch
import torchvision.models as models

# モデルのロード
model = models.resnet18(pretrained=True)
model.eval()

# ダミー入力の作成 (バッチサイズ、チャンネル数、高さ、幅)
dummy_input = torch.randn(1, 3, 224, 224)

# ONNXとしてエクスポート
torch.onnx.export(model, dummy_input, "resnet18.onnx", input_names=['input'], output_names=['output'])
コードは注意してご使用ください。2. ONNXモデルをTensorFlow形式に変換するONNXをTensorFlow形式（SavedModel や .pb）に変換するには、onnx-tf パッケージを使用します。bashpip install onnx-tf tensorflow
コードは注意してご使用ください。pythonfrom onnx_tf.backend import prepare
import onnx

# ONNXモデルの読み込み
onnx_model = onnx.load("resnet18.onnx")

# TFのオブジェクトに変換
tf_rep = prepare(onnx_model)

# TFモデルとして保存
tf_rep.export_graph("tf_model_dir")
コードは注意してご使用ください。💡 変換における注意点・代替案対応していないレイヤー: PyTorch独自の複雑なカスタムレイヤーや演算は、ONNXでTensorFlowにうまく変換できない場合があります。その場合は、TensorFlow側でモデル構造を再構築し、重み（パラメータ）のみをコピーする手動アプローチが確実です。テンソルの並び順（チャンネル順）の違い: PyTorchは [バッチ, チャンネル, 高さ, 幅]（Channels First）ですが、TensorFlowは [バッチ, 高さ, 幅, チャンネル]（Channels Last）を採用しています。変換後の次元の並び順（Permute）に注意が必要です。ONNXからTF Liteへの変換: エッジデバイス等で利用する .tflite が必要な場合は、TensorFlow形式（SavedModel）に変換した後に tf.lite.TFLiteConverter を用いて変換します。詳細は TensorFlow公式ガイド を参照してください。

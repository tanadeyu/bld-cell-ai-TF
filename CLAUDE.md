# Blood Cell AI TensorFlow版 Project

## Communication
- **思考**: 英語
- **応答**: 日本語

## ドキュメント作成ルール
- **すべてのドキュメント（MDファイルなど）は日本語で書くこと**
- 簡体字中国語などは使用せず、日本語（漢字・ひらがな・カタカナ）のみ使用する

## Project Overview
血液細胞の2クラス分類（成熟細胞 vs 未熟細胞）を行う深層学習プロジェクト（TensorFlow/Keras版）

- **データセット**: BloodMNIST (MedMNIST v2)
- **モデル**: ResNet18（BasicBlock [2,2,2,2]）, ViT-B/16（768/12層/12ヘッド）
- **※備考**: PyTorch版 torchvision.models と同じアーキテクチャ構成。`utils/convert_weights.py`でPyTorch ImageNet学習済み重みを変換→ロード可能（`USE_PRETRAINED=True`で切替）
- **タスク**: 2クラス分類（8クラスを臨床的観点から2クラスにグループ化）
- **フレームワーク**: TensorFlow/Keras
- **ライセンス**: MIT License

## Python Environment

| ツール | パス |
|--------|------|
| Python | `.conda/env/python.exe` |
| pip | `.conda/env/Scripts/pip.exe` |
| Jupyter | `.conda/env/Scripts/jupyter.exe` |
| Conda | `C:\Users\hello\miniconda3\Scripts\conda.exe` |

### 環境情報
- **Python**: 3.10.20 (DirectML Plugin互換)
- **TensorFlow**: 2.10.0 + DirectML Plugin (AMD GPU対応)
- **NumPy**: 1.26.4 (<2 必須、TF 2.10と非互換のため)
- **GPU**: AMD Radeon RX 7900 XT (20GB)
- **⚠️ 注意**: `tensorflow-directml-plugin`はTF 2.10.x用。TF 2.15は使用不可

### 環境構築手順（インストール順序が重要）
```bash
# 1. conda環境を作成
conda create -p .conda/env python=3.10 pip -y

# 2. TF本体を明示的にインストール（公式推奨手順）
.conda/env/Scripts/pip.exe install tensorflow-cpu==2.10

# 3. DirectML Plugin（2の後にインストール）
.conda/env/Scripts/pip.exe install tensorflow-directml-plugin

# 4. NumPy 1.xに固定（TF 2.10はNumPy 2.xと非互換）
.conda/env/Scripts/pip.exe install "numpy<2"

# 5. その他のパッケージをインストール
.conda/env/Scripts/pip.exe install scikit-learn matplotlib japanize-matplotlib seaborn pillow tqdm pyyaml jupyter medmnist
```
⚠️ `tensorflow`（メタパッケージ）や`tensorflow-gpu`は絶対に入れないこと。依存関係が壊れる。

### 実行例
```bash
# Jupyter起動
.conda/env/python.exe -m notebook

# パッケージ追加
.conda/env/Scripts/pip.exe install <package>

# ノートブック実行順
1. notebooks/01_環境構築とデータ準備.ipynb
2. notebooks/02_ResNet18学習.ipynb
3. notebooks/03_ViT学習.ipynb
4. notebooks/04_最終評価.ipynb
```

## Installed Packages
主要パッケージ:
- **tensorflow**: 2.10.0 (DirectML Plugin経由)
- **tensorflow-directml-plugin**: 0.4.0.dev230202
- **numpy**: 1.26.4
- **medmnist**: 3.0.2
- **scikit-learn**: 1.7.2
- **matplotlib**: 3.10.9
- **japanize-matplotlib**: 1.1.3
- **jupyter**: 1.1.1
- **seaborn**, **pillow**, **tqdm**, **pyyaml**

## ノートブック実行順
1. `notebooks/01_環境構築とデータ準備.ipynb` - データダウンロード・確認
2. `notebooks/02_ResNet18学習.ipynb` - ResNet18V2（カスタム実装）で学習
3. `notebooks/03_ViT学習.ipynb` - ViT-B/16（カスタム実装）で学習
4. `notebooks/04_最終評価.ipynb` - 両モデルの精度比較

## Important Notes

### ⚠️ Jupyter起動時の注意
**必ずプロジェクトの環境（`.conda/env/`）を使ってください**
```
❌ 誤：WindowsのPythonで起動
   jupyter notebook
   → TensorFlowが使えない場合あり

✅ 正：プロジェクトの環境で起動
   .conda/env/python.exe -m notebook
   → TensorFlowが使える
```

**確認方法**:
```python
import tensorflow as tf
print(tf.__version__)  # 2.10.0 と表示されればOK
print(tf.config.list_physical_devices('GPU'))  # PhysicalDevice(GPU:0) が表示されればOK

# GPU演算テスト
with tf.device('/GPU:0'):
    a = tf.random.normal([1000, 1000])
    b = tf.random.normal([1000, 1000])
    c = tf.matmul(a, b)
print('GPU matmul test OK')  # これが表示されればGPU演算OK
```

### ⚠️ パッケージ互換性
- **NumPy**: 2.xはTF 2.10と非互換。必ず `numpy<2` を維持
- **`tensorflow`メタパッケージ**: 入れると2.15+が入り依存が壊れる。`tensorflow-cpu==2.10`を使用
- **tensorflow-hub / vit-keras**: TF 2.10と競合する可能性あり。別途確認が必要
- **tensorflow-directml**（非推奨）: TF 1.x用なので使用不可。**tensorflow-directml-plugin**を使用
- **DirectML Plugin**: Microsoft公式で開発停止（discontinued）。教育・ポートフォリオ目的なら十分使用可能

### ⚠️ 注意・免責事項
- 本プロジェクトは**教育的・ポートフォリオ目的**です
- **医療診断用途での使用は strictly prohibited（厳禁）** です
- BloodMNISTデータセットは研究・教育用であり、実臨床データとは異なります
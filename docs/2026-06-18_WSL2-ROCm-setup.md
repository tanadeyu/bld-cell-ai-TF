# WSL2 + ROCm セットアップ記録（2026-06-18）

## 目的
DirectML + TF 2.10がRX 7900 XT (RDNA3)で5回連続カーネル死亡したため、WSL2 + ROCmでTensorFlow GPU学習を可能にする。

## 完了済み

| ステップ | 内容 | 状態 |
|---------|------|------|
| 1 | ROCm 7.2 リポジトリ追加 | ✅ |
| 1 | `amdgpu-install_7.2.70200-1_all.deb` インストール | ✅ |
| 1 | `sudo amdgpu-install -y --usecase=wsl,rocm --no-dkms` | ✅ |
| 2 | 環境変数（.bashrc）設定 | ✅ |
| 3a | cmake, g++ インストール | ✅ |
| 3b | librocdxg ビルド・インストール | ✅ |
| - | GPU認識: gfx1100 (Radeon RX 7900 XT) | ✅ |

## 環境変数（~/.bashrcに追加済み）
```bash
# ROCm on WSL2
export HSA_ENABLE_DXG_DETECTION=1
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PATH=$PATH:/opt/rocm/bin
```

## CUDA_VISIBLE_DEVICES環境変数（02_ResNet18学習.ipynb）
```python
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # DirectML用、ROCmでは不要
import tensorflow as tf
```
※ROCm環境では`TF_FORCE_GPU_ALLOW_GROWTH`は不要。GPU認識後に削除可能。

## 失敗したこと

### pip版 tensorflow-rocm（❌ 失敗）
- `tensorflow-rocm==2.14.0.600`（pip最新）はROCm 6.x向けビルド
- ROCm 7.2（`libamdhip64.so.7`）とバージョン不一致
- symblinkで`libamdhip64.so.6 → .7`を作成、NumPy 1.26.4にダウングレード
- TF自体はimportできたが、GPUを認識するも無視:
```
Ignoring visible gpu device (AMD Radeon RX 7900 XT)
with AMDGPU version: gfx1100.
The supported AMDGPU versions are gfx1030gfx1100, gfx900, gfx906, gfx908, gfx90a, gfx940, gfx941, gfx942.
```
- pip版tensorflow-rocmは更新停止（最新でも2.14/ROCm 6.x向け）

### venv（~/tf-rocm-env/）
- Python 3.10.12 + tensorflow-rocm 2.14.0.600
- GPU動作不可。**このvenvは使い物にならない**

## 次にやること: Docker方式

### イメージ
```
rocm/tensorflow:rocm7.2-py3.10-tf2.18-dev
```
ROCm 7.2 + TF 2.18の組み合わせ。pip版よりバージョン整合性が保証される。

### pullコマンド
```bash
docker pull rocm/tensorflow:rocm7.2-py3.10-tf2.18-dev
```
※イメージサイズが大きく、タイムアウトする可能性あり。時間をかけて再試行が必要。

### Docker実行コマンド（予定）
```bash
docker run -it --rm \
  --device /dev/dxg \
  -e HSA_ENABLE_DXG_DETECTION=1 \
  -e HSA_OVERRIDE_GFX_VERSION=11.0.0 \
  -v $(pwd):/workspace \
  -v /opt/rocm:/opt/rocm \
  -v /usr/lib/x86_64-linux-gnu/librocdxg.so:/usr/lib/x86_64-linux-gnu/librocdxg.so \
  -p 8888:8888 \
  rocm/tensorflow:rocm7.2-py3.10-tf2.18-dev \
  bash
```

### 懸念点
- DockerイメージはネイティブLinux向け（`/dev/kfd` + `/dev/dri`）に設計
- 当環境は`/dev/dxg`のみ（ROCDXG方式）
- `--device /dev/dxg`とlibrocdxgマウントで動く可能性はあるが未検証
- 失敗した場合の代替: PyTorch移行（既に動作実縮あり）

### Docker内でのJupyter起動（予定）
```bash
pip install jupyter matplotlib japanize-matplotlib medmnist scikit-learn
cd /workspace
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

## WSL2環境情報

| 項目 | 値 |
|------|-----|
| OS | Ubuntu 22.04.5 LTS (WSL2) |
| カーネル | 6.6.87.2-microsoft-standard-WSL2 |
| Windows Build | 10.0.26200.8655 |
| AMD Driver | 32.0.22001.17002 |
| GPU | AMD Radeon RX 7900 XT (gfx1100) |
| CPU | AMD Ryzen 7 5700X 8-Core |
| RAM | 15Gi（WSL2割当） |
| Disk | 943GB空き |
| ROCm | 7.2.0 (/opt/rocm) |
| librocdxg | /opt/rocm/lib/librocdxg.so |
| /dev/dxg | 存在（DirectX経由GPU接続） |
| /dev/kfd | なし（ROCDXG経由で代替） |
| /dev/dri | なし（ROCDXG経由で代替） |
| Docker | 29.5.5（WSL2内） |
| Windows SDK | 10.0.26100.0 |

## 本日の試行まとめ

| 試したこと | 結果 |
|---|---|
| DirectML + TF 2.10（Windows） | ❌ カーネル死亡×5回（RDNA3未対応バグ） |
| pip版 tensorflow-rocm 2.14（WSL2） | ❌ ROCm 6.x向け、gfx1100を無視 |
| Docker（rocm/tensorflow） | ⏳ pullタイムアウト、未完 |

## 代替案（Dockerが動かない場合）

| 方法 | 確実性 | 時間 |
|------|--------|------|
| PyTorch移行（既に動作実績あり） | ✅ 100% | 早い |
| WSL2内でCPU学習 | ✅ 100% | 20-40分 |

## 備考
- DirectML Plugin開発停止かつRDNA3未対応のため、Windows+DirectMLでのTF GPU学習は不可能
- PyTorch+DirectMLは同一PCで正常動作済み（ResNet18 finetuning 95.6%、ViT-B/16 finetuning 88.8%）
- ノートブックは02番がFunctional API + .weights.h5形式に書き直し済み

## 明日やること（ノートブック再開について）

- **ノートブック01はやり直し不要**（`data/bloodmnist.npz`、`models/resnet18_imagenet_weights.npz`、`models/vit_b16_imagenet_weights.npz`は既に存在）
- **ノートブック02（ResNet18学習）から再開**
- Docker/ROCm環境で起動する場合、ノートブック内のDirectML用コードを修正が必要:
  - `os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'` → 削除（不要）
  - GPU検出が`DMLGPUDevice`から`GPU:0`に変わる

### Docker環境でのJupyter起動手順
```bash
# 1. Docker pull（タイムアウトする場合は再試行）
docker pull rocm/tensorflow:rocm7.2-py3.10-tf2.18-dev

# 2. Docker起動
docker run -it --rm \
  --device /dev/dxg \
  -e HSA_ENABLE_DXG_DETECTION=1 \
  -e HSA_OVERRIDE_GFX_VERSION=11.0.0 \
  -v $(pwd):/workspace \
  -v /opt/rocm:/opt/rocm \
  -v /usr/lib/x86_64-linux-gnu/librocdxg.so:/usr/lib/x86_64-linux-gnu/librocdxg.so \
  -p 8888:8888 \
  rocm/tensorflow:rocm7.2-py3.10-tf2.18-dev \
  bash

# 3. Jupyter起動
pip install jupyter matplotlib japanize-matplotlib medmnist scikit-learn
cd /workspace
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

### もしDockerが動かない場合の代替案
| 方法 | 確実性 | 時間 |
|------|--------|------|
| PyTorch移行（既に動作実績あり） | ✅ 100% | 早い |
| WSL2内でCPU学習 | ✅ 100% | 20-40分 |
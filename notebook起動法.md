# ノートブック起動方法（ROCm + Docker）

## 起動
```bash
./run_jupyter.sh
```
ブラウザで http://localhost:8888

## シェル（デバッグ用）
```bash
./run_shell.sh
```

## 注意
- 初回のみDockerイメージビルド（約10分）
- GPU: RX 7900 XT / ROCm 7.2 / TF 2.19

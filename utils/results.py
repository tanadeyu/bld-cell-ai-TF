"""
結果の保存・読み込みを行うユーティリティモジュール
学習結果（精度、F1スコア、学習時間など）をYAML形式で保存・読み込みする。
PyTorch版/TensorFlow版 共通で使用可能。
"""
import yaml
import os
from typing import Dict, Any, List


def save_results(results: Dict[str, Any], filepath: str) -> None:
    """学習結果をYAMLファイルに保存"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        yaml.dump(results, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    print(f"結果を保存しました: {filepath}")


def load_results(filepath: str) -> Dict[str, Any]:
    """YAMLファイルから結果を読み込み"""
    with open(filepath, "r", encoding="utf-8") as f:
        results = yaml.safe_load(f)
    return results


def load_multiple_results(pattern: str) -> List[Dict[str, Any]]:
    """複数の結果ファイルを読み込み"""
    import glob
    results = []
    filepaths = glob.glob(pattern)
    for filepath in filepaths:
        result = load_results(filepath)
        results.append(result)
    results.sort(key=lambda x: x.get("approach", ""))
    return results


def format_results_table(results: List[Dict[str, Any]]) -> str:
    """結果を見やすい表形式に変換"""
    lines = []
    lines.append("=" * 90)
    lines.append("                             学習結果比較")
    lines.append("=" * 90)
    lines.append("")
    lines.append(f"{'モデル':<15} | {'アプローチ':<20} | {'精度':<10} | {'F1スコア':<10} | {'学習時間(分)':<15}")
    lines.append("-" * 90)
    for r in results:
        model = r.get("model", "")
        approach = r.get("approach", "")
        accuracy = r.get("test_accuracy", 0)
        f1 = r.get("test_f1", 0)
        time_min = r.get("training_time_minutes", 0)
        lines.append(f"{model:<15} | {approach:<20} | {accuracy:<10.4f} | {f1:<10.4f} | {time_min:<15.1f}")
    lines.append("=" * 90)
    return "\n".join(lines)


def print_results_summary(results: List[Dict[str, Any]]) -> None:
    """結果のサマリーを表示"""
    print(format_results_table(results))
    if results:
        best = max(results, key=lambda x: x.get("test_accuracy", 0))
        print(f"\n最高精度: {best['model']} ({best['approach']}) - {best['test_accuracy']:.4f}")


def get_model_results(base_path: str, model_names: List[str], approaches: List[str]) -> List[Dict[str, Any]]:
    """指定したモデルとアプローチの結果を読み込み"""
    results = []
    for model_name in model_names:
        for approach in approaches:
            filepath = f"{base_path}/{model_name}_{approach}_results.yaml"
            try:
                result = load_results(filepath)
                results.append(result)
            except FileNotFoundError:
                print(f"警告: ファイルが見つかりません: {filepath}")
    return results
"""
PyTorch ImageNet学習済み重み → TensorFlow/Keras .npz 変換スクリプト

PyTorch環境で実行:
  python convert_weights.py

出力:
  resnet18_imagenet_weights.npz
  vit_b16_imagenet_weights.npz
"""
import os
import numpy as np


def convert_resnet18():
    """torchvision ResNet18の重みをTF/Keras用に変換"""
    import torch
    from torchvision import models
    print("torchvision ResNet18の学習済み重みをダウンロード中...")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    state = model.state_dict()
    weights = {}
    # 畳み込み: (out, in, H, W) → (H, W, in, out)
    conv_map = {
        'conv1': 'conv1',
        'layer1.0.conv1': 'layer1_0_conv1', 'layer1.0.conv2': 'layer1_0_conv2',
        'layer1.0.downsample.0': 'layer1_0_downsample_0',
        'layer1.1.conv1': 'layer1_1_conv1', 'layer1.1.conv2': 'layer1_1_conv2',
        'layer2.0.conv1': 'layer2_0_conv1', 'layer2.0.conv2': 'layer2_0_conv2',
        'layer2.0.downsample.0': 'layer2_0_downsample_0',
        'layer2.1.conv1': 'layer2_1_conv1', 'layer2.1.conv2': 'layer2_1_conv2',
        'layer3.0.conv1': 'layer3_0_conv1', 'layer3.0.conv2': 'layer3_0_conv2',
        'layer3.0.downsample.0': 'layer3_0_downsample_0',
        'layer3.1.conv1': 'layer3_1_conv1', 'layer3.1.conv2': 'layer3_1_conv2',
        'layer4.0.conv1': 'layer4_0_conv1', 'layer4.0.conv2': 'layer4_0_conv2',
        'layer4.0.downsample.0': 'layer4_0_downsample_0',
        'layer4.1.conv1': 'layer4_1_conv1', 'layer4.1.conv2': 'layer4_1_conv2',
    }
    for pt_name, tf_name in conv_map.items():
        key = f'{pt_name}.weight'
        if key in state:
            w = state[key].numpy()
            weights[f'{tf_name}/kernel'] = np.transpose(w, (2, 3, 1, 0))
    # BatchNorm
    bn_map = {
        'bn1': 'bn1',
        'layer1.0.bn1': 'layer1_0_bn1', 'layer1.0.bn2': 'layer1_0_bn2',
        'layer1.0.downsample.1': 'layer1_0_downsample_1',
        'layer1.1.bn1': 'layer1_1_bn1', 'layer1.1.bn2': 'layer1_1_bn2',
        'layer2.0.bn1': 'layer2_0_bn1', 'layer2.0.bn2': 'layer2_0_bn2',
        'layer2.0.downsample.1': 'layer2_0_downsample_1',
        'layer2.1.bn1': 'layer2_1_bn1', 'layer2.1.bn2': 'layer2_1_bn2',
        'layer3.0.bn1': 'layer3_0_bn1', 'layer3.0.bn2': 'layer3_0_bn2',
        'layer3.0.downsample.1': 'layer3_0_downsample_1',
        'layer3.1.bn1': 'layer3_1_bn1', 'layer3.1.bn2': 'layer3_1_bn2',
        'layer4.0.bn1': 'layer4_0_bn1', 'layer4.0.bn2': 'layer4_0_bn2',
        'layer4.0.downsample.1': 'layer4_0_downsample_1',
        'layer4.1.bn1': 'layer4_1_bn1', 'layer4.1.bn2': 'layer4_1_bn2',
    }
    for pt_name, tf_name in bn_map.items():
        for suffix, tf_suffix in [('.weight', '/gamma'), ('.bias', '/beta'),
                                   ('.running_mean', '/moving_mean'),
                                   ('.running_var', '/moving_variance')]:
            key = f'{pt_name}{suffix}'
            if key in state:
                weights[f'{tf_name}{tf_suffix}'] = state[key].numpy()
    out_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'resnet18_imagenet_weights.npz')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, **weights)
    print(f"ResNet18: {len(weights)}重みを保存 → {out_path}")


def convert_vit_b16():
    """torchvision ViT-B/16の重みをTF/Keras用に変換"""
    import torch
    from torchvision import models
    print("torchvision ViT-B/16の学習済み重みをダウンロード中...")
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
    state = model.state_dict()
    weights = {}
    # conv_proj (パッチ埋め込み): (768, 3, 16, 16) → (16, 16, 3, 768)
    if 'conv_proj.weight' in state:
        w = state['conv_proj.weight'].numpy()
        weights['conv_proj/kernel'] = np.transpose(w, (2, 3, 1, 0))
    if 'conv_proj.bias' in state:
        weights['conv_proj/bias'] = state['conv_proj.bias'].numpy()
    # class_token: (1, 1, 768) → (1, 1, 768) ※squeezeしない
    if 'class_token' in state:
        weights['cls_token'] = state['class_token'].numpy()  # (1, 1, 768)
    # encoder.pos_embedding: (1, 197, 768) → (197, 768)
    if 'encoder.pos_embedding' in state:
        weights['position_embedding'] = state['encoder.pos_embedding'].numpy().squeeze(0)
    # encoder.layers.encoder_layer_{i} を変換
    for i in range(12):
        pt_prefix = f'encoder.layers.encoder_layer_{i}'
        tf_prefix = f'encoder_layer_{i}'
        # LayerNorm (ln_1, ln_2)
        for ln_name in ['ln_1', 'ln_2']:
            for suffix, tf_suffix in [('.weight', '/gamma'), ('.bias', '/beta')]:
                key = f'{pt_prefix}.{ln_name}{suffix}'
                if key in state:
                    weights[f'{tf_prefix}_{ln_name}{tf_suffix}'] = state[key].numpy()
        # Self-Attention: in_proj_weight (2304, 768) → Q, K, V
        key = f'{pt_prefix}.self_attention.in_proj_weight'
        if key in state:
            in_proj = state[key].numpy()
            q_w, k_w, v_w = np.split(in_proj, 3, axis=0)
            weights[f'{tf_prefix}_mha/query/kernel'] = q_w.T
            weights[f'{tf_prefix}_mha/key/kernel'] = k_w.T
            weights[f'{tf_prefix}_mha/value/kernel'] = v_w.T
        key = f'{pt_prefix}.self_attention.in_proj_bias'
        if key in state:
            in_proj_bias = state[key].numpy()
            q_b, k_b, v_b = np.split(in_proj_bias, 3)
            weights[f'{tf_prefix}_mha/query/bias'] = q_b
            weights[f'{tf_prefix}_mha/key/bias'] = k_b
            weights[f'{tf_prefix}_mha/value/bias'] = v_b
        # out_proj
        key = f'{pt_prefix}.self_attention.out_proj.weight'
        if key in state:
            weights[f'{tf_prefix}_mha/attention_output/kernel'] = state[key].numpy().T
        key = f'{pt_prefix}.self_attention.out_proj.bias'
        if key in state:
            weights[f'{tf_prefix}_mha/attention_output/bias'] = state[key].numpy()
        # MLP: linear_1 (mlp.0) と linear_2 (mlp.3)
        # mlp.0 = Dense(768→3072, GELU), mlp.3 = Dense(3072→768)
        key = f'{pt_prefix}.mlp.0.weight'
        if key in state:
            weights[f'{tf_prefix}_mlp_linear1/kernel'] = state[key].numpy().T
        key = f'{pt_prefix}.mlp.0.bias'
        if key in state:
            weights[f'{tf_prefix}_mlp_linear1/bias'] = state[key].numpy()
        key = f'{pt_prefix}.mlp.3.weight'
        if key in state:
            weights[f'{tf_prefix}_mlp_linear2/kernel'] = state[key].numpy().T
        key = f'{pt_prefix}.mlp.3.bias'
        if key in state:
            weights[f'{tf_prefix}_mlp_linear2/bias'] = state[key].numpy()
    # encoder.ln (最終LayerNorm)
    for suffix, tf_suffix in [('.weight', '/gamma'), ('.bias', '/beta')]:
        key = f'encoder.ln{suffix}'
        if key in state:
            weights[f'encoder_ln{tf_suffix}'] = state[key].numpy()
    # heads.head (分類層、1000クラス)
    if 'heads.head.weight' in state:
        weights['heads_head/kernel'] = state['heads.head.weight'].numpy().T
    if 'heads.head.bias' in state:
        weights['heads_head/bias'] = state['heads.head.bias'].numpy()
    out_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'vit_b16_imagenet_weights.npz')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, **weights)
    print(f"ViT-B/16: {len(weights)}重みを保存 → {out_path}")


if __name__ == '__main__':
    convert_resnet18()
    convert_vit_b16()
    print("\n変換完了！以下のファイルがmodels/に保存されました:")
    print("  - resnet18_imagenet_weights.npz")
    print("  - vit_b16_imagenet_weights.npz")
    print("\nTFノートブックでUSE_PRETRAINED=Trueに設定すると読み込まれます。")
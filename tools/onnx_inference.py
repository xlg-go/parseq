from torchvision import transforms
import onnxruntime as ort
import cv2
from PIL import Image
import torch
from strhub.models.utils import load_from_checkpoint
from torch import Tensor
from typing import List, Optional, Tuple
import numpy as np


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def _filter(probs: Tensor, ids: Tensor) -> Tuple[Tensor, List[int]]:
    ids = ids.tolist()
    try:
        eos_idx = ids.index(0)
    except ValueError:
        eos_idx = len(ids)  # Nothing to truncate.
    # Truncate after EOS
    ids = ids[:eos_idx]
    probs = probs[:eos_idx + 1]  # but include prob. for EOS (if it exists)
    return probs, ids


def _ids2tok(token_ids: List[int], join: bool = True) -> str:
    _itos = ('[E]', '0', '1', '2', '[B]', '[P]')
    tokens = [_itos[i] for i in token_ids]
    return ''.join(tokens) if join else tokens


def decode(token_dists: Tensor, raw: bool = False) -> Tuple[List[str], List[Tensor]]:
    batch_tokens = []
    batch_probs = []
    for dist in token_dists:
        probs, ids = dist.max(-1)  # greedy selection
        if not raw:
            probs, ids = _filter(probs, ids)
        tokens = _ids2tok(ids, not raw)
        batch_tokens.append(tokens)
        batch_probs.append(probs)
    return batch_tokens, batch_probs


def decode_sequence(pred):
    # pred 的形状是 (1, 27, 4)
    _, seq_length, num_classes = pred.shape

    # 转换为概率分布，通常需要使用 softmax
    pred_softmax = softmax(pred, axis=-1)

    # 获取每个位置上概率最高的类别
    pred_labels = np.argmax(pred_softmax, axis=-1)
    # 计算置信度
    pred_probabilities = np.max(pred_softmax, axis=-1)

    # 将类别映射回字符
    characters = ['[E]', '0', '1', '2']  # 根据你的实际情况修改
    pred_labels = pred_labels[0].tolist()
    # 移除 EOS（如果存在）
    eos_idx = pred_labels.index(characters.index('[E]'))
    pred_labels = pred_labels[:eos_idx]
    decoded_sequence = ''.join([characters[label] for label in pred_labels])
    confidences = pred_probabilities[0][:eos_idx + 1]

    return decoded_sequence, confidences


def softmax(x, axis=None):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / exp_x.sum(axis=axis, keepdims=True)


def infer():
    # parseq = load_from_checkpoint('../outputs/parseq/line_score/2023-11-20_08-31-55/checkpoints/last.ckpt')

    image = Image.open("../ccccc.jpg").convert("RGB")
    target_size = (512, 32)  # w, h
    # 计算等比例缩放后的大小
    aspect_ratio = image.width / image.height
    if aspect_ratio > 1:
        new_height = target_size[1]
        new_width = int(target_size[1] * aspect_ratio)
    else:
        new_width = target_size[0]
        new_height = int(target_size[0] / aspect_ratio)

    # 计算填充大小
    pad_width = max(0, target_size[0] - new_width)
    pad_height = max(0, target_size[1] - new_height)

    # 计算填充的位置
    padding = (0, 0, pad_width, pad_height)

    target_transform = transforms.Compose([
        transforms.Resize((new_height, new_width), transforms.InterpolationMode.BICUBIC),
        transforms.Pad(padding, fill=255),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])

    image = target_transform(image).unsqueeze(0)
    image = to_numpy(image)

    ort_sess = ort.InferenceSession('../outputs/parseq/line_score/2023-11-20_08-31-55/checkpoints/last.onnx')
    input_tensor = ort_sess.get_inputs()[0]

    ort_logits = ort_sess.run(None, {input_tensor.name: image})[0]

    decode_pred = decode_sequence(ort_logits)

    logits = torch.tensor(ort_logits)
    probs = logits.softmax(-1)
    preds, confis = decode(probs)
    # _, _ = parseq.tokenizer.decode(probs)
    print(preds)
    print(confis)
    # for pred, confi in zip(preds, confis):
    #     confidence += prob.prod().item()
    #     pred = self.charset_adapter(pred)
    #     if pred == gt:
    #         correct += 1
    #     total += 1
    #     label_length += len(pred)
    pass


if __name__ == '__main__':
    infer()

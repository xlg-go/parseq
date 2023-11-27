import torch
from onnxsim import simplify
from strhub.models.utils import load_from_checkpoint
import onnx


def converter():
    # parseq = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval()
    # # (1, 3, 32, 128) by default
    # dummy_input = torch.rand(1, 3, *parseq.hparams.img_size)
    # # To ONNX
    # # opset v14 or newer is required
    # parseq.to_onnx('parseq.onnx', dummy_input, opset_version=14)

    parseq = load_from_checkpoint('../outputs/parseq/line_score/2023-11-20_08-31-55/checkpoints/last.ckpt')
    parseq = parseq.to('cpu').eval()
    # parseq.decode_ar = False

    in_shape = [1, 3, 32, 512]
    dumpy_input = torch.rand(in_shape)
    onnx_path = '../outputs/parseq/line_score/2023-11-20_08-31-55/checkpoints/last.onnx'
    dynamic_axes = {
        'input': {0: 'batch_size', 2: "height", 3: "width"},
        'output': {0: 'batch_size', 1: "dim1", 2: "dim2"}
    }

    parseq.to_onnx(onnx_path,
                   dumpy_input,
                   # do_constant_folding=True,
                   opset_version=14,
                   # dynamic_axes=dynamic_axes,
                   input_names=['input'], output_names=['output'],
                   )

    # check
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model, full_check=True)
    # onnx_model, check = simplify(onnx_model)


if __name__ == '__main__':
    converter()

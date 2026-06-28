import tempfile
from pathlib import Path

import onnx
import onnxsim
import torch

from onyx_conv.models.mdx.models import Conv_TDF_net_trim

MDX_CONFIGS = {
    'vocals': {'L': 11, 'l': 3, 'g': 32, 'bn': 8, 'dim_f': 11, 'dim_t': 8},
    'bass':   {'L': 11, 'l': 3, 'g': 32, 'bn': 8, 'dim_f': 11, 'dim_t': 8},
    'drums':  {'L': 9,  'l': 3, 'g': 32, 'bn': 8, 'dim_f': 11, 'dim_t': 7},
    'other':  {'L': 11, 'l': 3, 'g': 32, 'bn': 8, 'dim_f': 11, 'dim_t': 8},
}

HOP_LENGTH = 1024
DIM_F = 2048


def get_config(target_name):
    return MDX_CONFIGS[target_name]


def export_mdx_onnx(model_path: str, target_name: str, output_path: str,
                    opset: int = 13, device='cpu') -> str:
    cfg = MDX_CONFIGS[target_name]
    net = Conv_TDF_net_trim(
        L=cfg['L'], l=cfg['l'], g=cfg['g'],
        dim_f=cfg['dim_f'], dim_t=cfg['dim_t'],
        target_name=target_name, bn=cfg['bn'], bias=False,
    )

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    if 'state_dict' in ckpt:
        sd = ckpt['state_dict']
    elif 'weight' in ckpt:
        sd = ckpt['weight']
    else:
        sd = ckpt

    net.load_state_dict(sd, strict=False)
    net.eval().to(device)

    print(f"  Loaded {target_name}: {model_path}")

    dim_t_full = 2 ** cfg['dim_t']
    dummy = torch.randn(1, 4, DIM_F, dim_t_full)

    torch.onnx.export(
        net, dummy, output_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        verbose=False,
    )

    model_simp, _ = onnxsim.simplify(output_path)
    onnx.save(model_simp, output_path)
    print(f"  Exported ONNX: {output_path}")
    return output_path


def export_or_load(source_path: str, target_name: str, opset: int = 13) -> bytes:
    p = Path(source_path)
    if p.suffix.lower() == '.onnx':
        print(f"  Using pre-existing ONNX: {source_path}")
        return p.read_bytes()
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp:
        tmp_path = tmp.name
    export_mdx_onnx(str(p), target_name, tmp_path, opset=opset)
    data = Path(tmp_path).read_bytes()
    Path(tmp_path).unlink(missing_ok=True)
    return data

import tempfile
from pathlib import Path

import onnx
import onnxsim
import torch

from onyx_conv.models.rvc import SynthesizerTrnMsNSFsidM

RVC_FILE_SPEC = {
    "model": "model.onnx",
    "pitch": "rmvpe.onnx",
    "embedding": "contentvec.onnx",
    "index": "model.index",
}


def export_rvc_onnx(model_path: str, output_path: str, opset: int = 18) -> str:
    cpt = torch.load(model_path, map_location="cpu", weights_only=False)

    config = cpt["config"]
    config[-3] = cpt["weight"]["emb_g.weight"].shape[0]

    version = cpt.get("version", "v1")
    print(f"  Detected: {version} model, phone dim={256 if version == 'v1' else 768}")
    is_half = False

    window_size = None
    for k, v in cpt["weight"].items():
        if "emb_rel_k" in k:
            window_size = (v.shape[1] - 1) // 2
            break

    net_g = SynthesizerTrnMsNSFsidM(*config, is_half=is_half, version=version, window_size=window_size)
    net_g.load_state_dict(cpt["weight"], strict=False)
    net_g.eval()
    net_g.remove_weight_norm()

    vec_channels = 256 if version == "v1" else 768
    T = 200

    test_phone = torch.rand(1, T, vec_channels)
    test_phone_lengths = torch.tensor([T]).long()
    test_pitch = torch.randint(size=(1, T), low=5, high=255)
    test_pitchf = torch.rand(1, T)
    test_ds = torch.LongTensor([0])
    test_rnd = torch.rand(1, 192, T)

    input_names = ["phone", "phone_lengths", "pitch", "pitchf", "ds", "rnd"]
    output_names = ["audio"]

    torch.onnx.export(
        net_g,
        (test_phone, test_phone_lengths, test_pitch, test_pitchf, test_ds, test_rnd),
        output_path,
        dynamic_axes={
            "phone": [1], "pitch": [1], "pitchf": [1], "rnd": [2],
        },
        do_constant_folding=False,
        opset_version=opset,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        dynamo=False,
    )

    model, _ = onnxsim.simplify(output_path)
    onnx.save(model, output_path)
    return output_path


def convert_rvc(input_path, output_path, opset=18,
                index_path=None, rmvpe_path=None, cv_path=None):
    from onyx.core.container import create_package

    out = Path(output_path)
    inp = Path(input_path)
    is_onyx = out.suffix.lower() == ".onyx"

    print(f"Input: {input_path}")
    print(f"Output: {out.name}")
    print(f"ONNX opset: {opset}")

    if inp.suffix.lower() == ".onnx":
        print(f"  Using existing ONNX: {input_path}")
        onnx_bytes = inp.read_bytes()
    else:
        print(f"  Exporting to ONNX...")
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
            onnx_tmp = tmp.name
        export_rvc_onnx(input_path, onnx_tmp, opset)
        onnx_bytes = Path(onnx_tmp).read_bytes()
        Path(onnx_tmp).unlink(missing_ok=True)

    if is_onyx:
        files = {"model": "model.onnx"}
        file_data = {"model.onnx": onnx_bytes}
        for role, path in [("pitch", rmvpe_path), ("embedding", cv_path), ("index", index_path)]:
            if path and Path(path).exists():
                arcname = RVC_FILE_SPEC[role]
                files[role] = arcname
                file_data[arcname] = Path(path).read_bytes()

        meta = {"files": files}
        meta.update(_auto_detect_rvc(onnx_bytes))
        create_package(output_path=str(out), model_type="rvc", metadata=meta, file_data=file_data)
        print(f"Packed: {out}")
    else:
        out.write_bytes(onnx_bytes)
        print(f"Saved: {out}")


def _auto_detect_rvc(onnx_bytes):
    meta = {}
    try:
        model = onnx.load_model_from_string(onnx_bytes)
        ratio = 1
        for node in model.graph.node:
            if node.op_type == "ConvTranspose":
                for attr in node.attribute:
                    if attr.name == "strides":
                        ratio *= attr.ints[0]
        meta["sample_rate"] = ratio * 100
        phone_dim = 768
        for inp in model.graph.input:
            if inp.name == "phone":
                dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
                if len(dims) == 3 and dims[2] > 0:
                    phone_dim = dims[2]
        meta["phone_dim"] = phone_dim
    except Exception:
        pass
    return meta

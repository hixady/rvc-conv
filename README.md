# onyx-conv

Multi-arch model converter for [onyx](https://github.com/hixady/onyx). Exports PyTorch checkpoints to ONNX and packs them into `.onyx` containers.

## Install

```bash
pip install onyx-conv            # core only (no torch)
pip install onyx-conv[rvc]       # with torch for RVC export
pip install onyx-conv[mdx]       # with torch for MDX export
pip install onyx-conv[all]       # both
```

## Usage

```bash
# Convert RVC checkpoint to .onyx (PyTorch → ONNX → pack)
onyx-conv convert rvc -i model.pth -o model.onyx \
    --rmvpe rmvpe.onnx --cv contentvec.onnx --index model.index

# Convert RVC from existing ONNX
onyx-conv convert rvc -i model.onnx -o model.onyx

# Convert MDX source separation (accepts .pth or .onnx per source)
onyx-conv convert mdx -o mdx.onyx \
    --vocals vocals.pth --bass bass.pth \
    --drums drums.pth --other other.pth \
    --mixer mixer.ckpt

# Export only (no packing)
onyx-conv convert rvc -i model.pth -o model.onnx
```

## Architectures

| Type | Description | Extras |
|------|-------------|--------|
| `rvc` | RVC voice conversion synthesizer | `[rvc]` |
| `mdx` | MDX-Net source separation (4 stems + optional mixer) | `[mdx]` |

Each arch accepts either `.pth` (PyTorch) or `.onnx` files. For `.pth` the converter exports the model to ONNX; for `.onnx` it uses the file directly.

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="onyx-conv — Convert PyTorch models to .onyx containers"
    )
    sub = parser.add_subparsers(dest="command")

    p_convert = sub.add_parser("convert", help="Convert a model to .onnx or .onyx")
    conv_sub = p_convert.add_subparsers(dest="arch")

    p_rvc = conv_sub.add_parser("rvc", help="Convert RVC .pth to .onnx or .onyx")
    p_rvc.add_argument("-i", "--input", required=True, help="Input .pth checkpoint")
    p_rvc.add_argument("-o", "--output", required=True, help="Output (.onnx or .onyx)")
    p_rvc.add_argument("--opset", type=int, default=18, help="ONNX opset version")
    p_rvc.add_argument("--index", default=None, help="FAISS index to bundle")
    p_rvc.add_argument("--rmvpe", default=None, help="RMVPE model to bundle (.onnx)")
    p_rvc.add_argument("--cv", default=None, help="ContentVec model to bundle (.onnx)")
    p_rvc.set_defaults(func=_convert_rvc)

    p_mdx = conv_sub.add_parser("mdx", help="Convert MDX-Net .pth/.onnx to .onyx")
    p_mdx.add_argument("-o", "--output", required=True, help="Output .onyx file")
    for s in ['vocals', 'bass', 'drums', 'other']:
        p_mdx.add_argument(f"--{s}", default=None, help=f"{s.capitalize()} model (.pth or .onnx)")
    p_mdx.add_argument("--mixer", default=None, help="Mixer model (.pth or .onnx)")
    p_mdx.add_argument("--opset", type=int, default=13, help="ONNX opset version")
    p_mdx.set_defaults(func=_convert_mdx)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    args.func(args)


def _convert_rvc(args):
    from onyx_conv.convert import convert_rvc
    convert_rvc(
        input_path=args.input,
        output_path=args.output,
        opset=args.opset,
        index_path=args.index,
        rmvpe_path=args.rmvpe,
        cv_path=args.cv,
    )


def _convert_mdx(args):
    from pathlib import Path
    from onyx.core.container import create_package
    from onyx_conv.models.mdx.export import export_or_load, MDX_CONFIGS, HOP_LENGTH, DIM_F

    n_fft_scale = {'vocals': 3, 'bass': 8, 'drums': 2, 'other': 4}
    file_data = {}
    files = {}
    sources = {}

    provided = [s for s in ['vocals', 'bass', 'drums', 'other'] if getattr(args, s)]
    if not provided:
        print("Error: at least one source model required (--vocals, --bass, --drums, --other)")
        sys.exit(1)

    for name in provided:
        src_path = getattr(args, name)
        p = Path(src_path)
        if p.suffix.lower() == '.onnx':
            print(f"  {name}: using ONNX directly")
            onnx_bytes = p.read_bytes()
        else:
            print(f"  {name}: exporting {p.name} to ONNX...")
            onnx_bytes = export_or_load(src_path, name, args.opset)
        arcname = f'{name}.onnx'
        files[f'model_{name}'] = arcname
        file_data[arcname] = onnx_bytes
        sources[name] = {'n_fft': DIM_F * n_fft_scale[name], 'dim_f': DIM_F}

    if args.mixer:
        mp = Path(args.mixer)
        files['mixer'] = 'mixer.onnx'
        if mp.suffix.lower() == '.onnx':
            file_data['mixer.onnx'] = mp.read_bytes()
        else:
            print(f"  mixer: exporting {mp.name} to ONNX...")
            import torch
            import torch.nn as nn
            ckpt = torch.load(str(mp), map_location='cpu', weights_only=False)
            sd = ckpt.get('state_dict', ckpt.get('weight', ckpt))
            linear = nn.Linear(10, 8, bias=False)
            linear.load_state_dict({'weight': sd['linear.weight']})
            linear.eval()
            dummy = torch.randn(1, 1024, 10)
            import tempfile, onnx, onnxsim
            with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp:
                tpath = tmp.name
            torch.onnx.export(linear, dummy, tpath, input_names=['input'], output_names=['output'],
                            dynamic_axes={'input': {0: 'batch', 1: 'time'}, 'output': {0: 'batch', 1: 'time'}},
                            opset_version=args.opset)
            ms, _ = onnxsim.simplify(tpath)
            onnx.save(ms, tpath)
            file_data['mixer.onnx'] = Path(tpath).read_bytes()
            Path(tpath).unlink(missing_ok=True)

    meta = {
        'sample_rate': 44100,
        'files': files,
        'config': {
            'hop_length': HOP_LENGTH,
            'dim_t': 512,
            'dim_c': 4,
            'sources': sources,
        },
        'supports': provided,
    }
    create_package(output_path=args.output, model_type='mdx', metadata=meta, file_data=file_data)


if __name__ == "__main__":
    main()

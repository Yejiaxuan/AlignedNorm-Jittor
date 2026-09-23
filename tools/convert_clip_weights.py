import argparse
import os
import os.path as osp
import subprocess
import sys
import tempfile

import numpy as np
import torch


def _to_numpy(value):
    if torch.is_tensor(value):
        tensor = value.detach().cpu()
        if tensor.is_floating_point():
            tensor = tensor.float()
        return tensor.numpy()
    return np.asarray(value)


def _load_torch_state(path):
    try:
        model = torch.jit.load(path, map_location="cpu").eval()
        state = model.state_dict()
    except RuntimeError:
        state = torch.load(path, map_location="cpu")

    if hasattr(state, "state_dict"):
        state = state.state_dict()
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(state)!r}")
    return state


def _ensure_parent(path):
    os.makedirs(osp.dirname(osp.abspath(path)), exist_ok=True)


def _save_with_jittor(state, output):
    _ensure_parent(output)
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        tmp_path = f.name
    try:
        np.savez(tmp_path, **state)
        code = (
            "import sys, numpy as np, jittor as jt; "
            "data=np.load(sys.argv[1], allow_pickle=False); "
            "jt.save({k:data[k] for k in data.files}, sys.argv[2])"
        )
        subprocess.check_call([sys.executable, "-c", code, tmp_path, output])
    finally:
        if osp.exists(tmp_path):
            os.remove(tmp_path)


def _validate_jittor_file(path):
    code = (
        "import sys, jittor as jt; "
        "state=jt.load(sys.argv[1]); "
        "assert isinstance(state, dict) and len(state) > 0"
    )
    subprocess.check_call([sys.executable, "-c", code, path])


def convert_clip(args):
    torch_state = _load_torch_state(args.input)
    jittor_state = {key: _to_numpy(value) for key, value in torch_state.items()}
    _save_with_jittor(jittor_state, args.output)
    if not args.no_validate:
        _validate_jittor_file(args.output)
    print(f"Saved Jittor CLIP weights: {args.output}")


def build_parser():
    parser = argparse.ArgumentParser(description="Convert original PyTorch CLIP weights to Jittor pkl.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    clip_parser = subparsers.add_parser("clip", help="convert CLIP checkpoint")
    clip_parser.add_argument("--input", required=True, help="input PyTorch CLIP .pt file")
    clip_parser.add_argument("--output", required=True, help="output Jittor .pkl file")
    clip_parser.add_argument("--no-validate", action="store_true", help="skip Jittor load validation")
    clip_parser.set_defaults(func=convert_clip)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

import hashlib
import os
import urllib.request
import warnings
from typing import Union, List

import jittor as jt
from PIL import Image
from jittor import transform
from tqdm import tqdm

from .model import build_model
from .model_mmrl import build_model_MMRL
from .model_mmrlpp import build_model_MMRLpp
from .model_alignednorm import build_model_ALIGNEDNORM

from .simple_tokenizer import SimpleTokenizer as _Tokenizer


__all__ = ["available_models", "load", "tokenize"]
_tokenizer = _Tokenizer()

_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
}


def _download(url: str, root: str = None):
    if root is None:
        root = os.environ.get("CLIP_MODEL_ROOT", os.path.expanduser("~/.cache/clip"))
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError(f"Model has been downloaded but the SHA256 checksum does not not match")

    return download_target


def _transform(n_px):
    return transform.Compose([
        transform.Resize(n_px, mode=Image.BICUBIC),
        transform.CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        transform.ToTensor(),
        transform.ImageNormalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def available_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())


def load(name: str, device="cuda", jit=False):
    """Load a converted Jittor CLIP model

    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a converted Jittor checkpoint

    device : str
        The device to use for the loaded model

    jit : bool
        JIT archives are not supported by Jittor

    Returns
    -------
    model : jittor.nn.Module
        The CLIP model

    preprocess : Callable[[PIL.Image], numpy.ndarray]
        A Jittor transform that converts a PIL image into a tensor that the returned model can take as its input
    """
    if jit:
        raise NotImplementedError("Jittor CLIP loader does not support jit=True")

    if name in _MODELS:
        model_path = os.path.splitext(_download(_MODELS[name]))[0] + ".pkl"
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Jittor weights not found at {model_path}. Run tools/convert_clip_weights.py first."
        )

    if str(device) == "cpu":
        jt.flags.use_cuda = 0
    elif str(device).startswith("cuda"):
        jt.flags.use_cuda = 1

    model = build_model(jt.load(model_path)).eval()
    return model, _transform(model.visual.input_resolution)


def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False):
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = jt.zeros((len(all_tokens), context_length), dtype="int32")

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = jt.array(tokens)

    return result.int32().stop_grad()

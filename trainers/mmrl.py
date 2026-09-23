import copy
import os.path as osp

import jittor as jt
from jittor import nn
from tqdm import tqdm

from clip import clip
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.optim import build_lr_scheduler, build_optimizer
from dassl.utils import load_pretrained_weights


CUSTOM_TEMPLATES = {
    'OxfordPets': 'a photo of a {}, a type of pet.',
    'OxfordFlowers': 'a photo of a {}, a type of flower.',
    'FGVCAircraft': 'a photo of a {}, a type of aircraft.',
    'DescribableTextures': '{} texture.',
    'EuroSAT': 'a centered satellite photo of {}.',
    'StanfordCars': 'a photo of a {}.',
    'Food101': 'a photo of {}, a type of food.',
    'SUN397': 'a photo of a {}.',
    'Caltech101': 'a photo of a {}.',
    'UCF101': 'a photo of a person doing {}.',
    'ImageNet': 'a photo of a {}.',
    'ImageNetSketch': 'a photo of a {}.',
    'ImageNetV2': 'a photo of a {}.',
    'ImageNetA': 'a photo of a {}.',
    'ImageNetR': 'a photo of a {}.'
}


def _load_jittor(path):
    if not osp.exists(path):
        raise FileNotFoundError(path)
    return jt.load(path)


def _argmax(x, dim=-1):
    out = x.argmax(dim=dim)
    return out[0] if isinstance(out, tuple) else out


def _normalize(x):
    return x / x.norm(dim=-1, keepdim=True)


def _project(x, weight):
    return x.astype(weight.dtype) @ weight


def _linear(x, layer):
    x = x.astype(layer.weight.dtype)
    y = x @ layer.weight.transpose(1, 0)
    if layer.bias is not None:
        y = y + layer.bias.astype(y.dtype)
    return y


def _image_input(x, dtype):
    return x.astype(dtype)


def _cosine_similarity(x, y):
    y = y.astype(x.dtype)
    return (x * y).sum(dim=-1) / (x.norm(dim=-1) * y.norm(dim=-1))


def load_clip_to_cpu(cfg, model_name="CLIP"):
    state_dict = _load_jittor(cfg.JITTOR.CLIP_WEIGHTS)
    design_details = {
        "model": model_name,
        "rep_tokens_layers": cfg.TRAINER.MMRL.REP_LAYERS,
        "n_rep_tokens": cfg.TRAINER.MMRL.N_REP_TOKENS,
    }
    return clip.build_model_MMRL(state_dict, design_details)


class TextEncoder_MMRL(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def execute(self, prompts, tokenized_prompts, compound_rep_tokens_text):
        n_rep_tokens = compound_rep_tokens_text[0].shape[0]
        x = prompts + self.positional_embedding.astype(self.dtype)
        x = x.permute((1, 0, 2))
        eot_index = _argmax(tokenized_prompts, dim=-1)
        combined = [x, compound_rep_tokens_text, 0, eot_index]
        # The upstream code incorrectly unpacks this three-element list into two values.
        outputs = self.transformer(combined)
        x = outputs[0]
        x = x.permute((1, 0, 2))
        x = self.ln_final(x).astype(self.dtype)
        return _project(x[jt.arange(x.shape[0]), eot_index + n_rep_tokens], self.text_projection)


class TextEncoder_CLIP(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def execute(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.astype(self.dtype)
        x = x.permute((1, 0, 2))
        outputs = self.transformer(x)
        x = outputs.permute((1, 0, 2))
        x = self.ln_final(x).astype(self.dtype)
        return _project(x[jt.arange(x.shape[0]), _argmax(tokenized_prompts, dim=-1)], self.text_projection)


def _get_text_base_features_zero_shot(cfg, classnames, clip_model, text_encoder):
    template = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
    with jt.no_grad():
        tokenized_prompts = []
        for text in tqdm(classnames, desc="Extracting text features"):
            tokenized_prompts.append(clip.tokenize(template.format(text.replace("_", " "))))
        tokenized_prompts = jt.concat(tokenized_prompts, dim=0).int32().stop_grad()
        embeddings = clip_model.token_embedding(tokenized_prompts).astype(clip_model.dtype)
        text_embeddings = text_encoder(embeddings, tokenized_prompts)
    return text_embeddings


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class MultiModalRepresentationLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_rep_tokens = cfg.TRAINER.MMRL.N_REP_TOKENS
        self.dtype = clip_model.dtype
        text_dim = clip_model.ln_final.weight.shape[0]
        visual_dim = clip_model.visual.ln_post.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        rep_dim = cfg.TRAINER.MMRL.REP_DIM
        self.rep_layers_length = len(cfg.TRAINER.MMRL.REP_LAYERS)
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        template = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        tokenized_prompts = [clip.tokenize(template.format(text.replace("_", " "))) for text in classnames]
        self.register_buffer(
            "tokenized_prompts",
            jt.concat(tokenized_prompts, dim=0).int32().stop_grad(),
            persistent=False,
        )
        with jt.no_grad():
            self.prompt_embeddings = clip_model.token_embedding(self.tokenized_prompts).astype(self.dtype).stop_grad()
        self.compound_rep_tokens = jt.randn((n_rep_tokens, rep_dim)) * 0.02
        single_layer_r2v = nn.Linear(rep_dim, visual_dim)
        single_layer_r2t = nn.Linear(rep_dim, text_dim)
        self.compound_rep_tokens_r2vproj = _get_clones(single_layer_r2v, self.rep_layers_length)
        self.compound_rep_tokens_r2tproj = _get_clones(single_layer_r2t, self.rep_layers_length)

    def execute(self):
        compound_rep_tokens_visual = []
        compound_rep_tokens_text = []
        for index in range(self.rep_layers_length):
            rep_tokens = self.compound_rep_tokens
            rep_mapped_to_text = _linear(rep_tokens, self.compound_rep_tokens_r2tproj[index])
            rep_mapped_to_visual = _linear(rep_tokens, self.compound_rep_tokens_r2vproj[index])
            compound_rep_tokens_text.append(rep_mapped_to_text.astype(self.dtype))
            compound_rep_tokens_visual.append(rep_mapped_to_visual.astype(self.dtype))
        return compound_rep_tokens_text, compound_rep_tokens_visual


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.alpha = cfg.TRAINER.MMRL.ALPHA
        self.classnames = classnames
        self.representation_learner = MultiModalRepresentationLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.representation_learner.tokenized_prompts
        self.register_buffer("prompt_embeddings", self.representation_learner.prompt_embeddings)
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder_MMRL(clip_model)
        self.dtype = clip_model.dtype
        self.text_features_for_inference = None
        self.compound_rep_tokens_text_for_inference = None
        self.compound_rep_tokens_visual_for_inference = None

    def execute(self, image):
        if self.representation_learner.training:
            self.text_features_for_inference = None
            self.compound_rep_tokens_text_for_inference = None
            self.compound_rep_tokens_visual_for_inference = None
            compound_rep_tokens_text, compound_rep_tokens_visual = self.representation_learner()
            text_features = self.text_encoder(self.prompt_embeddings, self.tokenized_prompts, compound_rep_tokens_text)
        else:
            if self.text_features_for_inference is None:
                self.compound_rep_tokens_text_for_inference, self.compound_rep_tokens_visual_for_inference = self.representation_learner()
                self.text_features_for_inference = self.text_encoder(
                    self.prompt_embeddings,
                    self.tokenized_prompts,
                    self.compound_rep_tokens_text_for_inference,
                )
                self.text_features_for_inference.persistent = False
            compound_rep_tokens_visual = self.compound_rep_tokens_visual_for_inference
            text_features = self.text_features_for_inference
        image_features, image_features_rep = self.image_encoder([_image_input(image, self.image_encoder.conv1.weight.dtype), compound_rep_tokens_visual])
        image_features = _normalize(image_features)
        image_features_rep = _normalize(image_features_rep)
        text_features = _normalize(text_features)
        logits = 100.0 * (image_features @ text_features.astype(image_features.dtype).transpose(1, 0))
        logits_rep = 100.0 * (image_features_rep @ text_features.astype(image_features_rep.dtype).transpose(1, 0))
        logits_fusion = self.alpha * logits + (1.0 - self.alpha) * logits_rep
        return logits, logits_rep, logits_fusion, image_features, text_features


class MMRL_Loss(nn.Module):
    def __init__(self, reg_weight=1.0, alpha=0.7):
        super().__init__()
        self.reg_weight = reg_weight
        self.alpha = alpha

    def execute(self, logits, logits_rep, image_features, text_features, image_features_clip, text_features_clip, label):
        logits = logits.float32()
        logits_rep = logits_rep.float32()
        xe_loss1 = nn.cross_entropy_loss(logits, label)
        xe_loss2 = nn.cross_entropy_loss(logits_rep, label)

        cossim_reg_img = 1.0 - _cosine_similarity(image_features, image_features_clip).mean()
        cossim_reg_text = 1.0 - _cosine_similarity(text_features, text_features_clip).mean()

        return self.alpha * xe_loss1 + (1.0 - self.alpha) * xe_loss2 + self.reg_weight * cossim_reg_img + self.reg_weight * cossim_reg_text


@TRAINER_REGISTRY.register()
class MMRL(TrainerX):
    checkpoint_ignored_keys = ("prompt_embeddings",)

    def check_cfg(self, cfg):
        if cfg.TRAINER.MMRL.PREC != "fp32":
            raise NotImplementedError("Jittor MMRL currently supports PREC='fp32' only")

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        self.num_classes = len(classnames)
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg, "MMRL")
        clip_model_zero_shot = load_clip_to_cpu(cfg)
        self.dtype = clip_model.dtype
        with jt.no_grad():
            self.text_encoder_clip = TextEncoder_CLIP(clip_model_zero_shot)
            text_features_clip = _get_text_base_features_zero_shot(cfg, classnames, clip_model_zero_shot, self.text_encoder_clip)
            self.text_features_clip = _normalize(text_features_clip).stop_grad()
        self.image_encoder_clip = clip_model_zero_shot.visual
        for _, param in self.image_encoder_clip.named_parameters():
            param.stop_grad()
        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)
        print("Turning off gradients in both the image and the text encoder")
        names_to_update = ["representation_learner", "image_encoder.proj_rep"]

        for name, param in self.model.named_parameters():
            update = False

            for name_to_update in names_to_update:
                if name_to_update in name:
                    update = True
                    break
            if update:
                param.start_grad()
            else:
                param.stop_grad()

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if not param.is_stop_grad():
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)
        reg_weight = cfg.TRAINER.MMRL.REG_WEIGHT
        alpha = cfg.TRAINER.MMRL.ALPHA
        self.criterion = MMRL_Loss(reg_weight=reg_weight, alpha=alpha)

        # NOTE: only give representation_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("MultiModalRepresentationLearner", self.model, self.optim, self.sched)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        with jt.no_grad():
            image_features_clip = _normalize(self.image_encoder_clip(_image_input(image, self.image_encoder_clip.conv1.weight.dtype))).stop_grad()
        logits, logits_rep, logits_fusion, image_features, text_features = self.model(image)
        text_features = text_features[0 : self.num_classes]
        loss = self.criterion(
            logits,
            logits_rep,
            image_features,
            text_features,
            image_features_clip,
            self.text_features_clip,
            label,
        )
        self.optim.step(loss)
        loss_summary = {
            "loss": float(loss.item()),
            "acc": float(compute_accuracy(logits_fusion, label)[0].item()),
        }
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
        return loss_summary

    def parse_batch_train(self, batch):
        return batch["img"], batch["label"].int32()

    def parse_batch_test(self, batch):
        return batch["img"], batch["label"].int32()

    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()
        sub_cls = self.cfg.DATASET.SUBSAMPLE_CLASSES
        dataset = self.cfg.DATASET.NAME
        task = self.cfg.TASK
        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        with jt.no_grad():
            for batch_idx, batch in enumerate(tqdm(data_loader)):
                input, label = self.parse_batch_test(batch)
                logits, _, logits_fusion, _, _, = self.model(input)

                if task == "B2N":
                    output = (
                        logits
                        if sub_cls == "new" and self.cfg.TEST.B2N_DECOUPLED
                        else logits_fusion
                    )
                elif task == "FS":
                    output = logits_fusion
                elif task == "CD":
                    output = logits_fusion if dataset == "ImageNet" else logits
                else:
                    raise ValueError("The TASK must be either B2N, CD, or FS.")

                self.evaluator.process(output, label)
        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]

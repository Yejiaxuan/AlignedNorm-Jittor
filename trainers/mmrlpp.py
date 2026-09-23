import os
import os.path as osp

import jittor as jt
from jittor import nn
from tqdm import tqdm
import math

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

CUSTOM_LOGITS = {
    'OxfordPets': 79.11041209,
    'OxfordFlowers': 77.40526995,
    'FGVCAircraft': 73.33903909,
    'DescribableTextures': 97.99665718,
    'EuroSAT': 92.54050771,
    'StanfordCars': 68.07360825,
    'Food101': 84.31223162,
    'SUN397': 86.96927229,
    'Caltech101': 90.01534722,
    'UCF101': 88.9097187,
    'ImageNet': 79.01900356,
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


def _linear(x, weight, bias=None):
    x = x.astype(weight.dtype)
    y = x @ weight.transpose(1, 0)
    if bias is not None:
        y = y + bias.astype(y.dtype)
    return y


def _project(x, weight):
    return x.astype(weight.dtype) @ weight


def _image_input(x, dtype):
    return x.astype(dtype)


def _normalize(x):
    return x / x.norm(dim=-1, keepdim=True)


def _cosine_similarity(x, y):
    y = y.astype(x.dtype)
    return (x * y).sum(dim=-1) / (x.norm(dim=-1) * y.norm(dim=-1))


def load_clip_to_cpu(cfg, model_name="CLIP"):
    state_dict = _load_jittor(cfg.JITTOR.CLIP_WEIGHTS)
    design_details = {
        "model": model_name,
        "rep_tokens_layers": cfg.TRAINER.MMRLpp.REP_LAYERS,
        "n_rep_tokens": cfg.TRAINER.MMRLpp.N_REP_TOKENS,
        "proj_lora_dim": cfg.TRAINER.MMRLpp.PROJ_LORA_DIM,
        "beta": cfg.TRAINER.MMRLpp.BETA,
    }
    return clip.build_model_MMRLpp(state_dict, design_details)


class TextEncoder_MMRLpp(nn.Module):
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
        combined = [x, compound_rep_tokens_text, 0]
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


class Residual_Aligner(nn.Module):
    def __init__(self, weight, bias, rank):
        super().__init__()
        self.weight_shape = weight.shape
        self.rank = rank
        self.A = nn.init.kaiming_uniform_(jt.zeros((self.weight_shape[0], self.rank)), a=math.sqrt(5))
        self.B = jt.zeros((self.rank, self.weight_shape[-1]))
        self.bias = bias.clone().detach() if bias is not None else None

    def execute(self, x, weight):
        return _linear(x, weight + (self.A @ self.B), self.bias)


class Shared_Residual_Representation_Aligner(nn.Module):
    def __init__(self, base_linear, num_layers, rank):
        super().__init__()
        self.weight = base_linear.weight.clone().detach()
        self.srra = nn.ModuleList([
            Residual_Aligner(weight=self.weight, bias=base_linear.bias, rank=rank)
            for _ in range(num_layers)
        ])

    def execute(self, x, idx):
        return self.srra[idx](x, self.weight)


class MultiModalRepresentationLearner_pp(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_rep_tokens = cfg.TRAINER.MMRLpp.N_REP_TOKENS
        self.dtype = clip_model.dtype
        text_dim = clip_model.ln_final.weight.shape[0]
        visual_dim = clip_model.visual.ln_post.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        rep_dim = cfg.TRAINER.MMRLpp.REP_DIM
        self.rep_layers_length = len(cfg.TRAINER.MMRLpp.REP_LAYERS)
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
        shared_layer_r2v = nn.Linear(rep_dim, visual_dim)
        shared_layer_r2t = nn.Linear(rep_dim, text_dim)
        res_lora_dim = cfg.TRAINER.MMRLpp.RES_LORA_DIM
        self.srra_r2vproj = Shared_Residual_Representation_Aligner(shared_layer_r2v, self.rep_layers_length, res_lora_dim)
        self.srra_r2tproj = Shared_Residual_Representation_Aligner(shared_layer_r2t, self.rep_layers_length, res_lora_dim)

    def execute(self):
        compound_rep_tokens_visual = []
        compound_rep_tokens_text = []
        for index in range(self.rep_layers_length):
            rep_tokens = self.compound_rep_tokens
            rep_mapped_to_text = self.srra_r2tproj(rep_tokens, index)
            rep_mapped_to_visual = self.srra_r2vproj(rep_tokens, index)
            compound_rep_tokens_text.append(rep_mapped_to_text.astype(self.dtype))
            compound_rep_tokens_visual.append(rep_mapped_to_visual.astype(self.dtype))
        return compound_rep_tokens_text, compound_rep_tokens_visual


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.alpha = cfg.TRAINER.MMRLpp.ALPHA
        self.classnames = classnames
        self.representation_learner = MultiModalRepresentationLearner_pp(cfg, classnames, clip_model)
        self.tokenized_prompts = self.representation_learner.tokenized_prompts
        self.register_buffer("prompt_embeddings", self.representation_learner.prompt_embeddings)
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder_MMRLpp(clip_model)
        self.dtype = clip_model.dtype
        dataset = cfg.DATASET.NAME
        # self.logits_scale = CUSTOM_LOGITS[dataset]
        self.logits_scale = 100.
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
        image_features_inter, image_features_rep_inter = self.image_encoder([_image_input(image, self.image_encoder.conv1.weight.dtype), compound_rep_tokens_visual])
        image_features = _normalize(image_features_inter)
        image_features_rep = _normalize(image_features_rep_inter)
        text_features = _normalize(text_features)
        logits = self.logits_scale * (image_features @ text_features.astype(image_features.dtype).transpose(1, 0))
        logits_rep = self.logits_scale * (image_features_rep @ text_features.astype(image_features_rep.dtype).transpose(1, 0))
        logits_fusion = self.alpha * logits + (1.0 - self.alpha) * logits_rep
        image_hybrid = self.alpha * image_features + (1.0 - self.alpha) * image_features_rep
        image_repr = image_features_rep
        image_hybrid_inter = self.alpha * image_features_inter + (1.0 - self.alpha) * image_features_rep_inter
        image_repr_inter = image_features_rep_inter
        return logits, logits_rep, logits_fusion, image_features, text_features, (
            image_features_rep_inter,
            image_features_inter,
            (image_hybrid, image_repr, image_hybrid_inter, image_repr_inter, text_features),
        )


class MMRLpp_Loss(nn.Module):
    def __init__(self, reg_weight=1.0, alpha=0.7):
        super().__init__()
        self.reg_weight = reg_weight
        self.alpha = alpha

    def execute(self, logits, logits_rep, image_features, text_features, image_features_clip, text_features_clip, image_features_rep, image_features_ref_inter, label):
        logits = logits.float32()
        logits_rep = logits_rep.float32()
        xe_loss1 = nn.cross_entropy_loss(logits, label)
        xe_loss2 = nn.cross_entropy_loss(logits_rep, label)
        cossim_reg_img = 1.0 - _cosine_similarity(image_features, image_features_clip).mean()
        cossim_reg_text = 1.0 - _cosine_similarity(text_features, text_features_clip).mean()
        rep_norm = image_features_rep.norm(dim=-1, keepdim=True)
        ref_norm = image_features_ref_inter.norm(dim=-1, keepdim=True)
        norm_loss = nn.relu(ref_norm.detach().astype(rep_norm.dtype) - rep_norm).mean()
        loss = self.alpha * xe_loss1 + (1.0 - self.alpha) * xe_loss2 + self.reg_weight * cossim_reg_img + self.reg_weight * cossim_reg_text
        return loss, 0.1 * norm_loss


@TRAINER_REGISTRY.register()
class MMRLpp(TrainerX):
    checkpoint_ignored_keys = ("prompt_embeddings",)

    def check_cfg(self, cfg):
        if cfg.TRAINER.MMRLpp.PREC != "fp32":
            raise NotImplementedError("Jittor MMRLpp currently supports PREC='fp32' only")

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        self.num_classes = len(classnames)
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg, "MMRLpp")
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
        names_to_update = ["representation_learner", "image_encoder.proj_rep", "image_encoder.A", "image_encoder.B"]

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

        reg_weight = cfg.TRAINER.MMRLpp.REG_WEIGHT
        alpha = cfg.TRAINER.MMRLpp.ALPHA
        self.criterion = MMRLpp_Loss(reg_weight=reg_weight, alpha=alpha)
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("MultiModalRepresentationLearner", self.model, self.optim, self.sched)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        with jt.no_grad():
            image_features_clip = _normalize(self.image_encoder_clip(_image_input(image, self.image_encoder_clip.conv1.weight.dtype))).stop_grad()
        logits, logits_rep, logits_fusion, image_features, text_features, inter_feature = self.model(image)
        text_features = text_features[0 : self.num_classes]
        image_features_rep, image_features_refer_inter, _ = inter_feature
        loss, norm_loss = self.criterion(
            logits,
            logits_rep,
            image_features,
            text_features,
            image_features_clip,
            self.text_features_clip,
            image_features_rep,
            image_features_refer_inter,
            label,
        )
        self.optim.step(loss)
        loss_summary = {
            "loss": float(loss.item()),
            "norm_loss": float(norm_loss.item()),
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
        self.set_model_mode("eval")
        self.evaluator.reset()
        if split is None:
            split = self.cfg.TEST.SPLIT
        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"
            data_loader = self.test_loader
        print(f"Evaluate on the *{split}* set")

        sub_cls = self.cfg.DATASET.SUBSAMPLE_CLASSES
        dataset = self.cfg.DATASET.NAME
        task = self.cfg.TASK
        feature_lists = {
            "image_hybrid": [],
            "image_repr": [],
            "image_hybrid_inter": [],
            "image_repr_inter": [],
            "text_feature": [],
            "logits_cls": [],
            "logits_repr": [],
            "logits_fusion": [],
            "label": [],
        }
        with jt.no_grad():
            for batch in tqdm(data_loader):
                image, label = self.parse_batch_test(batch)
                logits, logits_repr, logits_fusion, _, _, inter_feature = self.model(image)
                _, _, store_list = inter_feature
                image_hybrid, image_repr, image_hybrid_inter, image_repr_inter, text_feature = store_list
                feature_lists["image_hybrid"].append(image_hybrid.numpy())
                feature_lists["image_repr"].append(image_repr.numpy())
                feature_lists["image_hybrid_inter"].append(image_hybrid_inter.numpy())
                feature_lists["image_repr_inter"].append(image_repr_inter.numpy())
                feature_lists["text_feature"].append(text_feature.numpy())
                feature_lists["logits_cls"].append(logits.numpy())
                feature_lists["logits_repr"].append(logits_repr.numpy())
                feature_lists["logits_fusion"].append(logits_fusion.numpy())
                feature_lists["label"].append(label.numpy())

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

        if feature_lists["label"]:
            import numpy as np

            final_dict = {key: np.concatenate(values, axis=0) for key, values in feature_lists.items()}
            save_dir = osp.join(self.cfg.OUTPUT_DIR, "features", dataset)
            os.makedirs(save_dir, exist_ok=True)
            save_path = osp.join(save_dir, f"feat_{sub_cls}_{self.cfg.SEED}_{split}.npz")
            np.savez_compressed(save_path, **final_dict)
            print(f"Saved MMRLpp features: {save_path}")

        results = self.evaluator.evaluate()
        for key, value in results.items():
            self.write_scalar(f"{split}/{key}", value, self.epoch)
        return list(results.values())[0]

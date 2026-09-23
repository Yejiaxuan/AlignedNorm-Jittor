from ast import mod
from collections import OrderedDict
from typing import Tuple, Union

import jittor as jt
import numpy as np
from jittor import nn
import jittor.attention as F
from jittor.attention import MultiheadAttention
from jittor import init


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU()
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def execute(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = jt.randn((spacial_dim ** 2 + 1, embed_dim)) / embed_dim ** 0.5
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def execute(self, x):
        x = x.reshape((x.shape[0], x.shape[1], x.shape[2] * x.shape[3])).permute((2, 0, 1))  # NCHW -> (HW)NC
        x = jt.concat([x.mean(dim=0, keepdims=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].astype(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=jt.concat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.is_training(),
            need_weights=False
        )

        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU()

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)
        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def execute(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.astype(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass Jittor's LayerNorm to handle fp16."""

    def execute(self, x):
        orig_type = x.dtype
        ret = super().execute(x.float32())
        return ret.astype(orig_type)


class QuickGELU(nn.Module):
    def execute(self, x):
        return x * jt.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask=None, text_layer=False, design_details=None, i=0):
        super().__init__()

        self.attn = MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        if self.attn_mask is not None:
            self.attn_mask.persistent = False
        
        self.layer = i + 1
        self.rep_tokens_layers = design_details["rep_tokens_layers"]
        self.text_layer = text_layer
        self.n_rep_tokens = design_details["n_rep_tokens"]
        self.model = design_details["model"]

        if self.model == "MMRLpp":
            self.beta = design_details["beta"]

    def attention(self, x):
        attn_mask = self.attn_mask.astype(x.dtype) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def execute(self, inputs):

        if self.model == "CLIP":
            x = inputs
            x = x + self.attention(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
            return x

        elif self.model == "MMRL":
            x = inputs[0]
            compound_rep_tokens = inputs[1]
            counter = inputs[2]

            if len(compound_rep_tokens) > 0:
                if not self.text_layer:
                    if self.layer in self.rep_tokens_layers:
                        
                        if self.layer == self.rep_tokens_layers[0]:
                            prefix = x[:1, :, :]
                            suffix = x[1:, :, :]
                        else:
                            prefix = x[:1, :, :]
                            suffix = x[1 + self.n_rep_tokens:, :, :]                 

                        visual_context = compound_rep_tokens[counter]
                        visual_context = visual_context.unsqueeze(1).broadcast((visual_context.shape[0], x.shape[1], visual_context.shape[1]))
                        x = jt.concat([prefix, visual_context, suffix], dim=0)
                        counter += 1    

                else:
                    if self.layer in self.rep_tokens_layers:
                        
                        #insert tokens after bot
                        if self.layer == self.rep_tokens_layers[0]:
                            width = x.shape[0]
                            prefix = x[:1, :, :]
                            suffix = x[1:, :, :]
                        else:
                            width = x.shape[0] - self.n_rep_tokens
                            prefix = x[:1, :, :]
                            suffix = x[1 + self.n_rep_tokens:, :, :]                 

                        textual_context = compound_rep_tokens[counter]
                        textual_context = textual_context.unsqueeze(1).broadcast((textual_context.shape[0], x.shape[1], textual_context.shape[1]))
                        x = jt.concat([prefix, textual_context, suffix], dim=0)
                        counter += 1    

                    if self.layer >= self.rep_tokens_layers[0]:
                        width = x.shape[0]
                        self.attn_mask = jt.triu(jt.full((width, width), float("-inf"), dtype="float32"), diagonal=1).stop_grad()
                        self.attn_mask.persistent = False
            
            x = x + self.attention(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))

            return [x, compound_rep_tokens, counter] # return again as a list, so that nn.seq can work   
        
        elif self.model == "MMRLpp":
            x = inputs[0]
            compound_rep_tokens = inputs[1]
            counter = inputs[2]
            beta = self.beta

            if len(compound_rep_tokens) > 0:
                if not self.text_layer:
                    if self.layer in self.rep_tokens_layers:

                        visual_context = compound_rep_tokens[counter]
                        visual_context = visual_context.unsqueeze(1).broadcast((visual_context.shape[0], x.shape[1], visual_context.shape[1]))
                        
                        if self.layer == self.rep_tokens_layers[0]:
                            prefix = x[:1, :, :]
                            suffix = x[1:, :, :]
                            x = jt.concat([prefix, visual_context, suffix], dim=0)
                        else:
                            prefix = x[:1, :, :]
                            rep_tokens_prelayer = x[1:1 + self.n_rep_tokens, :, :]
                            suffix = x[1 + self.n_rep_tokens:, :, :]     
                            x = jt.concat([prefix, beta * visual_context + (1 - beta) * rep_tokens_prelayer, suffix], dim=0)
   
                        counter += 1    

                else:
                    if self.layer in self.rep_tokens_layers:

                        textual_context = compound_rep_tokens[counter]
                        textual_context = textual_context.unsqueeze(1).broadcast((textual_context.shape[0], x.shape[1], textual_context.shape[1]))

                        #insert tokens after bot
                        if self.layer == self.rep_tokens_layers[0]:
                            width = x.shape[0]
                            prefix = x[:1, :, :]
                            suffix = x[1:, :, :]
                            x = jt.concat([prefix, textual_context, suffix], dim=0)
                        else:
                            width = x.shape[0] - self.n_rep_tokens
                            prefix = x[:1, :, :]
                            rep_tokens_prelayer = x[1:1 + self.n_rep_tokens, :, :]
                            suffix = x[1 + self.n_rep_tokens:, :, :]       
                            x = jt.concat([prefix, beta * textual_context + (1 - beta) * rep_tokens_prelayer, suffix], dim=0)

                        counter += 1    

                    if self.layer >= self.rep_tokens_layers[0]:
                        width = x.shape[0]
                        self.attn_mask = jt.triu(jt.full((width, width), float("-inf"), dtype="float32"), diagonal=1).stop_grad()
                        self.attn_mask.persistent = False
                
            x = x + self.attention(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))

            return [x, compound_rep_tokens, counter] # return again as a list, so that nn.seq can work   


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask=None, text_layer=False, design_details=None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask, text_layer, design_details, i) for i in range(layers)])

    def execute(self, x):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int, design_details):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = scale * jt.randn((width,))
        self.positional_embedding = scale * jt.randn(((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads, design_details=design_details)

        self.ln_post = LayerNorm(width)
        self.proj = scale * jt.randn((width, output_dim))
        self.proj_rep = scale * jt.randn((width, output_dim))

        self.model = design_details["model"]

    def execute(self, inputs):
        if self.model == "CLIP":
            x = inputs
        elif self.model == "MMRL":
            x = inputs[0]
            compound_rep_tokens = inputs[1]

        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape((x.shape[0], x.shape[1], -1))  # shape = [*, width, grid ** 2]
        x = x.permute((0, 2, 1))  # shape = [*, grid ** 2, width]
        x = jt.concat([self.class_embedding.astype(x.dtype) + jt.zeros((x.shape[0], 1, x.shape[-1]), dtype=x.dtype), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.astype(x.dtype)
        x = self.ln_pre(x)

        x = x.permute((1, 0, 2))  # NLD -> LND
        if self.model == "CLIP":
            outputs = self.transformer(x)
            x = outputs
        elif self.model == "MMRL":
            outputs = self.transformer([x, compound_rep_tokens, 0])
            x = outputs[0]

        x = x.permute((1, 0, 2))  # LND -> NLD

        if self.proj is not None:
            if self.model == "MMRL":
                n_tokens= compound_rep_tokens[0].shape[0]
                x_rep = self.ln_post(x[:, 1:1+n_tokens, :])
                x_rep = x_rep.mean(dim=1)
                proj_rep = self.proj_rep.astype(x_rep.dtype)
                x_rep = x_rep @ proj_rep
                
            x = self.ln_post(x[:, 0, :])
            x = x.astype(self.proj.dtype) @ self.proj
    
        if self.model == "CLIP":     
            return x
        else:
            return x, x_rep


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 design_details
                 ):
        super().__init__()

        model = design_details["model"]
        assert model in ["MMRL", "MMRLpp", "CLIP"], "Your model must be in MMRL, MMRLpp or CLIP"
        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                design_details=design_details
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
            text_layer=True,
            design_details=design_details
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = jt.empty((self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = jt.empty((transformer_width, embed_dim))
        self.logit_scale = jt.ones(()) * np.log(1 / 0.07)

        self.initialize_parameters()

    def initialize_parameters(self):
        init.gauss_(self.token_embedding.weight, std=0.02)
        init.gauss_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                init.gauss_(self.visual.attnpool.q_proj.weight, std=std)
                init.gauss_(self.visual.attnpool.k_proj.weight, std=std)
                init.gauss_(self.visual.attnpool.v_proj.weight, std=std)
                init.gauss_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        init.zero_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            init.gauss_(block.attn.in_proj_weight, std=attn_std)
            init.gauss_(block.attn.out_proj.weight, std=proj_std)
            init.gauss_(block.mlp.c_fc.weight, std=fc_std)
            init.gauss_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            init.gauss_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # Jittor uses additive attention mask; fill with -inf
        mask = jt.full((self.context_length, self.context_length), float("-inf"), dtype="float32")
        return jt.triu(mask, diagonal=1).stop_grad()  # zero out the lower diagonal

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.astype(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).astype(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.astype(self.dtype)
        x = x.permute((1, 0, 2))  # NLD -> LND
        x = self.transformer(x)
        x = x.permute((1, 0, 2))  # LND -> NLD
        x = self.ln_final(x).astype(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        eot = text.argmax(dim=-1)
        if isinstance(eot, tuple):
            eot = eot[0]
        x = x[jt.arange(x.shape[0]), eot].astype(self.text_projection.dtype) @ self.text_projection

        return x

    def execute(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * (image_features @ text_features.astype(image_features.dtype).transpose(1, 0))
        logits_per_text = logit_scale * (text_features @ image_features.astype(text_features.dtype).transpose(1, 0))

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.assign(l.weight.float16())
            if l.bias is not None:
                l.bias.assign(l.bias.float16())

        if isinstance(l, MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.assign(tensor.float16())

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.assign(attr.float16())

    model.apply(_convert_weights_to_fp16)


def build_model_MMRL(state_dict: dict, design_details):
    vit = "visual.proj" in state_dict

    model = design_details["model"]
    assert model in ["MMRL", "MMRLpp", "CLIP"], "Your model must be in MMRL, MMRLpp or CLIP "

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers, design_details
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    model.load_state_dict(state_dict)
    return model.eval()

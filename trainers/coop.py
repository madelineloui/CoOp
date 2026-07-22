# import os.path as osp

# import torch
# import torch.nn as nn
# from torch.nn import functional as F
# from torch.cuda.amp import GradScaler, autocast

# from dassl.engine import TRAINER_REGISTRY, TrainerX
# from dassl.metrics import compute_accuracy
# from dassl.utils import load_pretrained_weights, load_checkpoint
# from dassl.optim import build_optimizer, build_lr_scheduler

# from clip import clip
# from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

# import open_clip

# _tokenizer = _Tokenizer()


# def load_clip_to_cpu(cfg):
#     backbone_name = cfg.MODEL.BACKBONE.NAME
#     print("-> using backbone:", backbone_name)

#     url = clip._MODELS["ViT-L/14"]
#     model_path = clip._download(url)

#     try:
#         model = torch.jit.load(model_path, map_location="cpu").eval()
#         state_dict = None
#     except RuntimeError:
#         state_dict = torch.load(model_path, map_location="cpu")

#     model = clip.build_model(state_dict or model.state_dict())

#     if backbone_name == "clip-14":
#         print("LOADED CLIP-14!")

#     elif backbone_name == "openclip-14":
#         model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
#         model.dtype = next(model.visual.parameters()).dtype
#         print("LOADED OPENCLIP-14!")

#     elif backbone_name == "remoteclip-14":
#         state_dict = torch.load(
#             "/home/gridsan/manderson/ovdsat/weights/RemoteCLIP-ViT-L-14.pt",
#             map_location="cpu"
#         )
#         model = clip.build_model(state_dict)
#         print("LOADED REMOTECLIP-14!")

#     elif backbone_name == "georsclip-14":
#         state_dict = torch.load(
#             "/home/gridsan/manderson/ovdsat/weights/RS5M_ViT-L-14.pt",
#             map_location="cpu"
#         )
#         model = clip.build_model(state_dict)
#         print("LOADED GEORSCLIP-14!")

#     elif backbone_name == "openclip-14-remote-fmow":
#         state_dict = torch.load(
#             "/home/gridsan/manderson/ovdsat/weights/vlm4rs/openclip-remote-fmow-summary-epoch100.pt",
#             map_location="cpu"
#         )
#         model = clip.build_model(state_dict)
#         print("LOADED RemoteCLIP-14+FMOW!")

#     elif backbone_name == "openclip-14-geors-fmow":
#         state_dict = torch.load(
#             "/home/gridsan/manderson/ovdsat/weights/vlm4rs/openclip-geors-fmow-summary-epoch100.pt",
#             map_location="cpu"
#         )
#         model = clip.build_model(state_dict)
#         print("LOADED GEORSCLIP-14+FMOW!")

#     return model


# class TextEncoder(nn.Module):
#     def __init__(self, clip_model):
#         super().__init__()
#         self.transformer = clip_model.transformer
#         self.positional_embedding = clip_model.positional_embedding
#         self.ln_final = clip_model.ln_final
#         self.text_projection = clip_model.text_projection
#         self.dtype = clip_model.dtype

#     def forward(self, prompts, tokenized_prompts):
#         x = prompts + self.positional_embedding.type(self.dtype)
#         x = x.permute(1, 0, 2)
#         x = self.transformer(x)
#         x = x.permute(1, 0, 2)
#         x = self.ln_final(x).type(self.dtype)
#         x = x[
#             torch.arange(x.shape[0], device=x.device),
#             tokenized_prompts.argmax(dim=-1)
#         ] @ self.text_projection
#         return x


# class PromptLearner(nn.Module):
#     def __init__(self, cfg, classnames, clip_model):
#         super().__init__()

#         n_cls = len(classnames)
#         n_ctx = cfg.TRAINER.COOP.N_CTX
#         ctx_init = cfg.TRAINER.COOP.CTX_INIT
#         dtype = clip_model.dtype
#         ctx_dim = clip_model.ln_final.weight.shape[0]

#         if hasattr(clip_model.visual, "input_resolution"):
#             clip_imsize = clip_model.visual.input_resolution
#         elif hasattr(clip_model.visual, "image_size"):
#             image_size = clip_model.visual.image_size
#             clip_imsize = image_size if isinstance(image_size, int) else image_size[0]
#         else:
#             raise AttributeError("Could not determine CLIP image size")

#         cfg_imsize = cfg.INPUT.SIZE[0]
#         assert cfg_imsize == clip_imsize, (
#             f"cfg_imsize ({cfg_imsize}) must equal clip_imsize ({clip_imsize})"
#         )

#         # === METADATA CHANGE === Store whether this run should use per-image metadata.
#         self.use_metadata = cfg.DATASET.METADATA

#         if ctx_init:
#             # use given words to initialize context vectors
#             ctx_init = ctx_init.replace("_", " ")
#             n_ctx = len(ctx_init.split(" "))
#             prompt = clip.tokenize(ctx_init)
        
#             with torch.no_grad():
#                 embedding = clip_model.token_embedding(prompt).type(dtype)
        
#             ctx_vectors = embedding[0, 1:1 + n_ctx, :]
#             prompt_prefix = ctx_init
        
#         else:
#             # random initialization
#             if cfg.TRAINER.COOP.CSC:
#                 print("Initializing class-specific contexts")
#                 ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
#             else:
#                 print("Initializing a generic context")
#                 ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        
#             nn.init.normal_(ctx_vectors, std=0.02)
#             prompt_prefix = " ".join(["X"] * n_ctx)

#         self.ctx = nn.Parameter(ctx_vectors)

#         classnames = [name.replace("_", " ") for name in classnames]
#         name_lens = [len(_tokenizer.encode(name)) for name in classnames]

#         # === METADATA CHANGE === Keep class names and token embedding so metadata prompts can be built per batch.
#         self.classnames = classnames
#         self.token_embedding = clip_model.token_embedding
#         self.dtype = dtype

#         # === METADATA CHANGE === Keep the exact original static prompt setup when metadata is disabled.
#         if not self.use_metadata:
#             prompts = [prompt_prefix + " " + name + "." for name in classnames]
#             tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])

#             with torch.no_grad():
#                 embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

#             self.register_buffer("token_prefix", embedding[:, :1, :])
#             self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])
#             self.tokenized_prompts = tokenized_prompts
#         else:
#             self.tokenized_prompts = None

#         self.n_cls = n_cls
#         self.n_ctx = n_ctx
#         self.name_lens = name_lens
#         self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

#         print("CLASSNAMES", classnames)
#         print("METADATA", self.use_metadata)
#         print("CLASS_TOKEN_POSITION", self.class_token_position)
#         print("CSC", cfg.TRAINER.COOP.CSC)

#     # === METADATA CHANGE === Move the original forward logic into this method (no metadata)
#     def forward_original(self):
#         ctx = self.ctx

#         if ctx.dim() == 2:
#             ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

#         prefix = self.token_prefix
#         suffix = self.token_suffix

#         if self.class_token_position == "end":
#             prompts = torch.cat([prefix, ctx, suffix], dim=1)

#         elif self.class_token_position == "middle":
#             half_n_ctx = self.n_ctx // 2
#             prompts = []

#             for i in range(self.n_cls):
#                 name_len = self.name_lens[i]
#                 prefix_i = prefix[i:i + 1]
#                 class_i = suffix[i:i + 1, :name_len]
#                 suffix_i = suffix[i:i + 1, name_len:]
#                 ctx_i_half1 = ctx[i:i + 1, :half_n_ctx]
#                 ctx_i_half2 = ctx[i:i + 1, half_n_ctx:]

#                 prompt = torch.cat(
#                     [prefix_i, ctx_i_half1, class_i, ctx_i_half2, suffix_i],
#                     dim=1
#                 )
#                 prompts.append(prompt)

#             prompts = torch.cat(prompts, dim=0)

#         elif self.class_token_position == "front":
#             prompts = []

#             for i in range(self.n_cls):
#                 name_len = self.name_lens[i]
#                 prefix_i = prefix[i:i + 1]
#                 class_i = suffix[i:i + 1, :name_len]
#                 suffix_i = suffix[i:i + 1, name_len:]
#                 ctx_i = ctx[i:i + 1]

#                 prompt = torch.cat(
#                     [prefix_i, class_i, ctx_i, suffix_i],
#                     dim=1
#                 )
#                 prompts.append(prompt)

#             prompts = torch.cat(prompts, dim=0)

#         else:
#             raise ValueError(
#                 f"Invalid CLASS_TOKEN_POSITION: {self.class_token_position}"
#             )

#         return prompts

#     def expand_metadata_context(self, batch_size):
#         # === METADATA CHANGE === Repeat generic or class-specific context for every image-class pair.
#         if self.ctx.dim() == 2:
#             return self.ctx.unsqueeze(0).expand(
#                 batch_size * self.n_cls, -1, -1
#             )

#         return self.ctx.unsqueeze(0).expand(
#             batch_size, -1, -1, -1
#         ).reshape(
#             batch_size * self.n_cls,
#             self.n_ctx,
#             self.ctx.shape[-1]
#         )

#     def forward_metadata(self, metadata):
#         # === METADATA CHANGE === Build B×C prompts so every image uses its own metadata for every candidate class.
#         batch_size = len(metadata)
#         device = self.ctx.device
#         half_n_ctx = self.n_ctx // 2
#         second_half_n_ctx = self.n_ctx - half_n_ctx

#         all_ctx = " ".join(["X"] * self.n_ctx)
#         first_ctx = " ".join(["X"] * half_n_ctx)
#         second_ctx = " ".join(["X"] * second_half_n_ctx)

#         prompt_strings = []
#         description_lengths = []

#         for metadata_text in metadata:
#             metadata_text = str(metadata_text).strip()

#             for classname in self.classnames:
#                 description = (
#                     f"{classname} {metadata_text}"
#                     if metadata_text
#                     else classname
#                 )
#                 description_lengths.append(len(_tokenizer.encode(description)))

#                 if self.class_token_position == "front":
#                     prompt = f"{description} {all_ctx}."
#                 elif self.class_token_position == "middle":
#                     prompt = f"{first_ctx} {description} {second_ctx}."
#                 elif self.class_token_position == "end":
#                     prompt = f"{all_ctx} {description}."
#                 else:
#                     raise ValueError(
#                         f"Invalid CLASS_TOKEN_POSITION: {self.class_token_position}"
#                     )

#                 prompt_strings.append(prompt)

#         tokenized_prompts = torch.cat(
#             [clip.tokenize(p) for p in prompt_strings]
#         ).to(device)

#         with torch.no_grad():
#             embedding = self.token_embedding(tokenized_prompts).type(self.dtype)

#         ctx = self.expand_metadata_context(batch_size)

#         if self.class_token_position == "end":
#             # === METADATA CHANGE === Layout: [context] [class name + metadata].
#             prompts = torch.cat(
#                 [embedding[:, :1], ctx, embedding[:, 1 + self.n_ctx:]],
#                 dim=1
#             )

#         elif self.class_token_position == "front":
#             # === METADATA CHANGE === Layout: [class name + metadata] [context].
#             prompts = []

#             for i, description_length in enumerate(description_lengths):
#                 ctx_start = 1 + description_length
#                 prompt = torch.cat(
#                     [
#                         embedding[i:i + 1, :ctx_start],
#                         ctx[i:i + 1],
#                         embedding[i:i + 1, ctx_start + self.n_ctx:]
#                     ],
#                     dim=1
#                 )
#                 prompts.append(prompt)

#             prompts = torch.cat(prompts, dim=0)

#         else:
#             # === METADATA CHANGE === Layout: [first context half] [class name + metadata] [second context half].
#             prompts = []

#             for i, description_length in enumerate(description_lengths):
#                 description_start = 1 + half_n_ctx
#                 description_end = description_start + description_length
#                 suffix_start = description_end + second_half_n_ctx

#                 prompt = torch.cat(
#                     [
#                         embedding[i:i + 1, :1],
#                         ctx[i:i + 1, :half_n_ctx],
#                         embedding[i:i + 1, description_start:description_end],
#                         ctx[i:i + 1, half_n_ctx:],
#                         embedding[i:i + 1, suffix_start:]
#                     ],
#                     dim=1
#                 )
#                 prompts.append(prompt)

#             prompts = torch.cat(prompts, dim=0)

#         return prompts, tokenized_prompts

#     def forward(self, metadata=None):
#         # === METADATA CHANGE === Select original CoOp or metadata-aware CoOp from cfg.DATASET.METADATA.
#         if not self.use_metadata:
#             return self.forward_original()
#         else:
#             return self.forward_metadata(metadata)


# class CustomCLIP(nn.Module):
#     def __init__(self, cfg, classnames, clip_model):
#         super().__init__()

#         # === METADATA CHANGE === Store mode so the original forward path remains unchanged when metadata is False.
#         self.use_metadata = cfg.DATASET.METADATA

#         self.prompt_learner = PromptLearner(cfg, classnames, clip_model)

#         if not self.use_metadata:
#             self.tokenized_prompts = self.prompt_learner.tokenized_prompts

#         self.image_encoder = clip_model.visual
#         self.text_encoder = TextEncoder(clip_model)
#         self.logit_scale = clip_model.logit_scale
#         self.dtype = clip_model.dtype

#     def forward(self, image, metadata=None):
#         image_features = self.image_encoder(image.type(self.dtype))

#         if not self.use_metadata:
#             # === METADATA CHANGE === Preserve the exact original static CoOp computation.
#             prompts = self.prompt_learner()
#             text_features = self.text_encoder(
#                 prompts,
#                 self.tokenized_prompts
#             )

#             image_features = image_features / image_features.norm(
#                 dim=-1, keepdim=True
#             )
#             text_features = text_features / text_features.norm(
#                 dim=-1, keepdim=True
#             )

#             return (
#                 self.logit_scale.exp()
#                 * image_features
#                 @ text_features.t()
#             )
        
#         else:
#             # === METADATA CHANGE === Generate separate class prompts for each image's metadata.
#             prompts, tokenized_prompts = self.prompt_learner(metadata)
#             text_features = self.text_encoder(
#                 prompts,
#                 tokenized_prompts
#             )
    
#             image_features = image_features / image_features.norm(
#                 dim=-1, keepdim=True
#             )
#             text_features = text_features / text_features.norm(
#                 dim=-1, keepdim=True
#             )
    
#             # === METADATA CHANGE === Reshape B×C prompts so each image is compared only with prompts using its metadata.
#             text_features = text_features.reshape(
#                 image_features.shape[0],
#                 self.prompt_learner.n_cls,
#                 -1
#             )
    
#             return self.logit_scale.exp() * torch.einsum(
#                 "bd,bcd->bc",
#                 image_features,
#                 text_features
#             )


# @TRAINER_REGISTRY.register()
# class CoOp(TrainerX):
#     """Context Optimization (CoOp).

#     Learning to Prompt for Vision-Language Models
#     https://arxiv.org/abs/2109.01134
#     """

#     def check_cfg(self, cfg):
#         assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

#     def build_model(self):
#         cfg = self.cfg
#         classnames = self.dm.dataset.classnames

#         print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
#         clip_model = load_clip_to_cpu(cfg)

#         if cfg.TRAINER.COOP.PREC in ["fp32", "amp"]:
#             clip_model.float()

#         print("Building custom CLIP")
#         self.model = CustomCLIP(cfg, classnames, clip_model)

#         print("Turning off gradients in both the image and the text encoder")
#         for name, param in self.model.named_parameters():
#             if "prompt_learner" not in name:
#                 param.requires_grad_(False)

#         if cfg.MODEL.INIT_WEIGHTS:
#             load_pretrained_weights(
#                 self.model.prompt_learner,
#                 cfg.MODEL.INIT_WEIGHTS
#             )

#         self.model.to(self.device)
#         self.optim = build_optimizer(
#             self.model.prompt_learner,
#             cfg.OPTIM
#         )
#         self.sched = build_lr_scheduler(
#             self.optim,
#             cfg.OPTIM
#         )
#         self.register_model(
#             "prompt_learner",
#             self.model.prompt_learner,
#             self.optim,
#             self.sched
#         )

#         self.scaler = (
#             GradScaler()
#             if cfg.TRAINER.COOP.PREC == "amp"
#             else None
#         )

#         device_count = torch.cuda.device_count()
#         if device_count > 1:
#             print(
#                 f"Multiple GPUs detected "
#                 f"(n_gpus={device_count}), use all of them!"
#             )
#             self.model = nn.DataParallel(self.model)

#     def forward_backward(self, batch):
#         image, label, metadata = self.parse_batch_train(batch)
#         prec = self.cfg.TRAINER.COOP.PREC

#         if prec == "amp":
#             with autocast():
#                 # === METADATA CHANGE === Passing None preserves the original model path when metadata is disabled.
#                 output = self.model(image, metadata)
#                 loss = F.cross_entropy(output, label)

#             self.optim.zero_grad()
#             self.scaler.scale(loss).backward()
#             self.scaler.step(self.optim)
#             self.scaler.update()

#         else:
#             # === METADATA CHANGE === Passing None preserves the original model path when metadata is disabled.
#             output = self.model(image, metadata)
#             loss = F.cross_entropy(output, label)
#             self.model_backward_and_update(loss)

#         loss_summary = {
#             "loss": loss.item(),
#             "acc": compute_accuracy(output, label)[0].item(),
#         }

#         if (self.batch_idx + 1) == self.num_batches:
#             self.update_lr()

#         return loss_summary

#     def parse_batch_train(self, batch):
#         input = batch["img"].to(self.device)
#         label = batch["label"].to(self.device)

#         # === METADATA CHANGE === Do not require a metadata batch field when metadata is disabled.
#         metadata = (batch["metadata"] if self.cfg.DATASET.METADATA else None)

#         return input, label, metadata

#     def parse_batch_test(self, batch):
#         input = batch["img"].to(self.device)
#         label = batch["label"].to(self.device)

#         # === METADATA CHANGE === Save test metadata without changing Dassl's expected two-value return.
#         self.test_metadata = (batch["metadata"] if self.cfg.DATASET.METADATA else None)

#         return input, label

#     def model_inference(self, input):
#         # === METADATA CHANGE === Use saved metadata during validation/test; None invokes original CoOp.
#         return self.model(input, self.test_metadata)

#     def load_model(self, directory, epoch=None):
#         if not directory:
#             print(
#                 "Note that load_model() is skipped as no "
#                 "pretrained model is given"
#             )
#             return

#         names = self.get_model_names()
#         model_file = "model-best.pth.tar"

#         if epoch is not None:
#             model_file = "model.pth.tar-" + str(epoch)

#         for name in names:
#             model_path = osp.join(
#                 directory,
#                 name,
#                 model_file
#             )

#             if not osp.exists(model_path):
#                 raise FileNotFoundError(
#                     f'Model not found at "{model_path}"'
#                 )

#             checkpoint = load_checkpoint(model_path)
#             state_dict = checkpoint["state_dict"]
#             epoch = checkpoint["epoch"]

#             if "token_prefix" in state_dict:
#                 del state_dict["token_prefix"]

#             if "token_suffix" in state_dict:
#                 del state_dict["token_suffix"]

#             print(
#                 "Loading weights to {} "
#                 'from "{}" (epoch = {})'.format(
#                     name,
#                     model_path,
#                     epoch
#                 )
#             )

#             self._models[name].load_state_dict(
#                 state_dict,
#                 strict=False
#             )

import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

import open_clip

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    print("-> using backbone:", backbone_name)

    url = clip._MODELS["ViT-L/14"]
    model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    if backbone_name == "clip-14":
        print("LOADED CLIP-14!")

    elif backbone_name == "openclip-14":
        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-L-14",
            pretrained="openai"
        )
        model.dtype = next(model.visual.parameters()).dtype
        print("LOADED OPENCLIP-14!")

    elif backbone_name == "remoteclip-14":
        state_dict = torch.load(
            "/home/gridsan/manderson/ovdsat/weights/RemoteCLIP-ViT-L-14.pt",
            map_location="cpu"
        )
        model = clip.build_model(state_dict)
        print("LOADED REMOTECLIP-14!")

    elif backbone_name == "georsclip-14":
        state_dict = torch.load(
            "/home/gridsan/manderson/ovdsat/weights/RS5M_ViT-L-14.pt",
            map_location="cpu"
        )
        model = clip.build_model(state_dict)
        print("LOADED GEORSCLIP-14!")

    elif backbone_name == "openclip-14-remote-fmow":
        state_dict = torch.load(
            "/home/gridsan/manderson/ovdsat/weights/vlm4rs/openclip-remote-fmow-summary-epoch100.pt",
            map_location="cpu"
        )
        model = clip.build_model(state_dict)
        print("LOADED RemoteCLIP-14+FMOW!")

    elif backbone_name == "openclip-14-geors-fmow":
        state_dict = torch.load(
            "/home/gridsan/manderson/ovdsat/weights/vlm4rs/openclip-geors-fmow-summary-epoch100.pt",
            map_location="cpu"
        )
        model = clip.build_model(state_dict)
        print("LOADED GEORSCLIP-14+FMOW!")

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[
            torch.arange(x.shape[0], device=x.device),
            tokenized_prompts.argmax(dim=-1)
        ] @ self.text_projection
        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()

        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        if hasattr(clip_model.visual, "input_resolution"):
            clip_imsize = clip_model.visual.input_resolution
        elif hasattr(clip_model.visual, "image_size"):
            image_size = clip_model.visual.image_size
            clip_imsize = image_size if isinstance(image_size, int) else image_size[0]
        else:
            raise AttributeError("Could not determine CLIP image size")

        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, (
            f"cfg_imsize ({cfg_imsize}) must equal clip_imsize ({clip_imsize})"
        )

        # === METADATA CHANGE === Store whether this run should use per-image metadata.
        self.use_metadata = cfg.DATASET.METADATA

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)

            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)

            ctx_vectors = embedding[0, 1:1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, n_ctx if False else ctx_dim, dtype=dtype)

            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]

        # === METADATA CHANGE === Keep class names and token embedding so metadata prompts can be built per batch.
        self.classnames = classnames
        self.token_embedding = clip_model.token_embedding
        self.dtype = dtype

        # === METADATA CHANGE === Keep the exact original static prompt setup when metadata is disabled.
        if not self.use_metadata:
            prompts = [prompt_prefix + " " + name + "." for name in classnames]
            tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])

            with torch.no_grad():
                embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

            self.register_buffer("token_prefix", embedding[:, :1, :])
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])
            self.tokenized_prompts = tokenized_prompts
        else:
            self.tokenized_prompts = None

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

        # === METADATA CHANGE === Print one metadata-conditioned prompt only once.
        self._printed_metadata_debug = False

        print("CLASSNAMES", classnames)
        print("METADATA", self.use_metadata)
        print("CLASS_TOKEN_POSITION", self.class_token_position)
        print("CSC", cfg.TRAINER.COOP.CSC)

    # === METADATA CHANGE === Move the original forward logic into this method for metadata-disabled runs.
    def forward_original(self):
        ctx = self.ctx

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat([prefix, ctx, suffix], dim=1)

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []

            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i:i + 1]
                class_i = suffix[i:i + 1, :name_len]
                suffix_i = suffix[i:i + 1, name_len:]
                ctx_i_half1 = ctx[i:i + 1, :half_n_ctx]
                ctx_i_half2 = ctx[i:i + 1, half_n_ctx:]

                prompt = torch.cat(
                    [prefix_i, ctx_i_half1, class_i, ctx_i_half2, suffix_i],
                    dim=1
                )
                prompts.append(prompt)

            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []

            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i:i + 1]
                class_i = suffix[i:i + 1, :name_len]
                suffix_i = suffix[i:i + 1, name_len:]
                ctx_i = ctx[i:i + 1]

                prompt = torch.cat(
                    [prefix_i, class_i, ctx_i, suffix_i],
                    dim=1
                )
                prompts.append(prompt)

            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError(
                f"Invalid CLASS_TOKEN_POSITION: {self.class_token_position}"
            )

        return prompts

    def expand_metadata_context(self, batch_size):
        # === METADATA CHANGE === Expand context for every image-class prompt.
        if self.ctx.dim() == 2:
            return self.ctx.unsqueeze(0).expand(
                batch_size * self.n_cls,
                -1,
                -1
            )

        return self.ctx.unsqueeze(0).expand(
            batch_size,
            -1,
            -1,
            -1
        ).reshape(
            batch_size * self.n_cls,
            self.n_ctx,
            self.ctx.shape[-1]
        )

    def forward_metadata(self, metadata):
        # === METADATA CHANGE === Build one metadata-conditioned prompt per image-class pair.
        batch_size = len(metadata)
        device = self.ctx.device
        half_n_ctx = self.n_ctx // 2
        second_half_n_ctx = self.n_ctx - half_n_ctx

        all_ctx = " ".join(["X"] * self.n_ctx)
        first_ctx = " ".join(["X"] * half_n_ctx)
        second_ctx = " ".join(["X"] * second_half_n_ctx)

        prompt_strings = []
        description_lengths = []

        for metadata_text in metadata:
            metadata_text = str(metadata_text).strip()

            for classname in self.classnames:
                description = (
                    f"{classname} {metadata_text}"
                    if metadata_text
                    else classname
                )
                description_lengths.append(
                    len(_tokenizer.encode(description))
                )

                if self.class_token_position == "front":
                    prompt = f"{description} {all_ctx}."
                elif self.class_token_position == "middle":
                    prompt = f"{first_ctx} {description} {second_ctx}."
                elif self.class_token_position == "end":
                    prompt = f"{all_ctx} {description}."
                else:
                    raise ValueError(
                        f"Invalid CLASS_TOKEN_POSITION: {self.class_token_position}"
                    )

                prompt_strings.append(prompt)

        # === METADATA CHANGE === Print one generated prompt once to verify metadata placement.
        if not self._printed_metadata_debug:
            print("METADATA PROMPT EXAMPLE:", prompt_strings[0])
            self._printed_metadata_debug = True

        tokenized_prompts = torch.cat(
            [clip.tokenize(p) for p in prompt_strings]
        ).to(device)

        with torch.no_grad():
            embedding = self.token_embedding(
                tokenized_prompts
            ).type(self.dtype)

        ctx = self.expand_metadata_context(batch_size)

        if self.class_token_position == "end":
            # === METADATA CHANGE === Layout: [context] [class name + metadata].
            prompts = torch.cat(
                [
                    embedding[:, :1],
                    ctx,
                    embedding[:, 1 + self.n_ctx:]
                ],
                dim=1
            )

        elif self.class_token_position == "front":
            # === METADATA CHANGE === Layout: [class name + metadata] [context].
            prompts = []

            for i, description_length in enumerate(description_lengths):
                ctx_start = 1 + description_length

                prompt = torch.cat(
                    [
                        embedding[i:i + 1, :ctx_start],
                        ctx[i:i + 1],
                        embedding[
                            i:i + 1,
                            ctx_start + self.n_ctx:
                        ]
                    ],
                    dim=1
                )
                prompts.append(prompt)

            prompts = torch.cat(prompts, dim=0)

        else:
            # === METADATA CHANGE === Layout: [first context half] [class name + metadata] [second context half].
            prompts = []

            for i, description_length in enumerate(description_lengths):
                description_start = 1 + half_n_ctx
                description_end = (
                    description_start + description_length
                )
                suffix_start = (
                    description_end + second_half_n_ctx
                )

                prompt = torch.cat(
                    [
                        embedding[i:i + 1, :1],
                        ctx[i:i + 1, :half_n_ctx],
                        embedding[
                            i:i + 1,
                            description_start:description_end
                        ],
                        ctx[i:i + 1, half_n_ctx:],
                        embedding[i:i + 1, suffix_start:]
                    ],
                    dim=1
                )
                prompts.append(prompt)

            prompts = torch.cat(prompts, dim=0)

        return prompts, tokenized_prompts

    def forward(self, metadata=None):
        # === METADATA CHANGE === Select original or metadata-aware prompt construction.
        if not self.use_metadata:
            return self.forward_original()

        return self.forward_metadata(metadata)


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()

        # === METADATA CHANGE === Store whether metadata mode is enabled.
        self.use_metadata = cfg.DATASET.METADATA

        self.prompt_learner = PromptLearner(
            cfg,
            classnames,
            clip_model
        )

        if not self.use_metadata:
            self.tokenized_prompts = (
                self.prompt_learner.tokenized_prompts
            )

        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, metadata=None):
        image_features = self.image_encoder(
            image.type(self.dtype)
        )

        if not self.use_metadata:
            # === METADATA CHANGE === Preserve the original static CoOp computation.
            prompts = self.prompt_learner()
            text_features = self.text_encoder(
                prompts,
                self.tokenized_prompts
            )

            image_features = image_features / image_features.norm(
                dim=-1,
                keepdim=True
            )
            text_features = text_features / text_features.norm(
                dim=-1,
                keepdim=True
            )

            return (
                self.logit_scale.exp()
                * image_features
                @ text_features.t()
            )

        # === METADATA CHANGE === Generate class prompts using each image's metadata.
        prompts, tokenized_prompts = self.prompt_learner(
            metadata
        )
        text_features = self.text_encoder(
            prompts,
            tokenized_prompts
        )

        image_features = image_features / image_features.norm(
            dim=-1,
            keepdim=True
        )
        text_features = text_features / text_features.norm(
            dim=-1,
            keepdim=True
        )

        # === METADATA CHANGE === Group the prompts by image and class.
        text_features = text_features.reshape(
            image_features.shape[0],
            self.prompt_learner.n_cls,
            -1
        )

        return self.logit_scale.exp() * torch.einsum(
            "bd,bcd->bc",
            image_features,
            text_features
        )


@TRAINER_REGISTRY.register()
class CoOp(TrainerX):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in [
            "fp16",
            "fp32",
            "amp"
        ]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(
            f"Loading CLIP "
            f"(backbone: {cfg.MODEL.BACKBONE.NAME})"
        )
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.COOP.PREC in ["fp32", "amp"]:
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(
            cfg,
            classnames,
            clip_model
        )

        print(
            "Turning off gradients in both the image "
            "and the text encoder"
        )
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(
                self.model.prompt_learner,
                cfg.MODEL.INIT_WEIGHTS
            )

        self.model.to(self.device)

        self.optim = build_optimizer(
            self.model.prompt_learner,
            cfg.OPTIM
        )
        self.sched = build_lr_scheduler(
            self.optim,
            cfg.OPTIM
        )
        self.register_model(
            "prompt_learner",
            self.model.prompt_learner,
            self.optim,
            self.sched
        )

        self.scaler = (
            GradScaler()
            if cfg.TRAINER.COOP.PREC == "amp"
            else None
        )

        device_count = torch.cuda.device_count()

        if device_count > 1:
            print(
                f"Multiple GPUs detected "
                f"(n_gpus={device_count}), use all of them!"
            )
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label, metadata = self.parse_batch_train(
            batch
        )
        prec = self.cfg.TRAINER.COOP.PREC

        if prec == "amp":
            with autocast():
                # === METADATA CHANGE === Pass metadata when enabled; None uses the original path.
                output = self.model(
                    image,
                    metadata
                )
                loss = F.cross_entropy(
                    output,
                    label
                )

            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()

        else:
            # === METADATA CHANGE === Pass metadata when enabled; None uses the original path.
            output = self.model(
                image,
                metadata
            )
            loss = F.cross_entropy(
                output,
                label
            )
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(
                output,
                label
            )[0].item(),
        }

        if (
            self.batch_idx + 1
        ) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"].to(self.device)
        label = batch["label"].to(self.device)

        # === METADATA CHANGE === Read metadata only when metadata mode is enabled.
        metadata = (
            batch["metadata"]
            if self.cfg.DATASET.METADATA
            else None
        )

        return input, label, metadata

    def parse_batch_test(self, batch):
        input = batch["img"].to(self.device)
        label = batch["label"].to(self.device)

        # === METADATA CHANGE === Save metadata because SimpleTrainer.test() passes only input to model_inference().
        self.test_metadata = (
            batch["metadata"]
            if self.cfg.DATASET.METADATA
            else None
        )

        return input, label

    def model_inference(self, input):
        # === METADATA CHANGE === Use the current test batch metadata; None invokes original CoOp.
        return self.model(
            input,
            self.test_metadata
        )

    def load_model(self, directory, epoch=None):
        if not directory:
            print(
                "Note that load_model() is skipped as no "
                "pretrained model is given"
            )
            return

        names = self.get_model_names()
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = (
                "model.pth.tar-" + str(epoch)
            )

        for name in names:
            model_path = osp.join(
                directory,
                name,
                model_file
            )

            if not osp.exists(model_path):
                raise FileNotFoundError(
                    f'Model not found at "{model_path}"'
                )

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print(
                "Loading weights to {} "
                'from "{}" (epoch = {})'.format(
                    name,
                    model_path,
                    epoch
                )
            )

            self._models[name].load_state_dict(
                state_dict,
                strict=False
            )
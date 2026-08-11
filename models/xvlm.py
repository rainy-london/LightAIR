# X^2-VLM: All-In-One Pre-trained Model For Vision-Language Tasks (https://arxiv.org/abs/2211.12402)
# Github: https://github.com/zengyan-97/X2-VLM
# Copyright (c) 2022, ByteDance Inc.
# All rights reserved.
import json
# Multi-Grained Vision Language Pre-Training: Aligning Texts with Visual Concepts (https://arxiv.org/abs/2111.08276)
# Github: https://github.com/zengyan-97/X-VLM
# Copyright (c) 2022, ByteDance Inc.
# All rights reserved.

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from collections import OrderedDict
from torch.nn import CrossEntropyLoss

from einops import rearrange

from timm.models.layers import trunc_normal_

from models.xbert import BertConfig, BertForMaskedLM, BertModel
from models.xroberta import RobertaForMaskedLM, RobertaModel, RobertaConfig
import copy

from transformers import XLMRobertaTokenizer, BertTokenizer

def read_json(rpath):
    with open(rpath, 'r') as f:
        return json.load(f)


def compute_entropy(probs):
    return -torch.sum(probs * torch.log(probs + 1e-8), dim =-1)

class VanillaConfig(object):
    def __init__(self):
        pass


# def build_tokenizer(text_encoder: str):
#     tokenizer = XLMRobertaTokenizer.from_pretrained(text_encoder)
#     tokenizer.add_special_tokens({'bos_token': tokenizer.cls_token, 'eos_token': tokenizer.sep_token})
#     return tokenizer


def build_tokenizer(text_encoder: str, dropout=0):
    if ('bert-base-uncased' in text_encoder) or ('bert-large-uncased' in text_encoder):
        tokenizer = BertTokenizer.from_pretrained(text_encoder)

    elif ('xlm-roberta-base' in text_encoder) or ('xlm-roberta-large' in text_encoder):
        tokenizer = XLMRobertaTokenizer.from_pretrained(text_encoder)

    else:
        raise NotImplementedError(f"tokenizer for {text_encoder}")

    # always use cls and sep
    tokenizer.add_special_tokens({'bos_token': tokenizer.cls_token})
    tokenizer.add_special_tokens({'eos_token': tokenizer.sep_token})

    return tokenizer



def load_params_change_prefix(state_dict: dict, prefix: str, new_prefix: str):
    if prefix == new_prefix:
        return state_dict

    state_dict_new = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            k = k.replace(prefix, new_prefix)

        state_dict_new[k] = v

    return state_dict_new


def load_roberta_lm_head(state_dict):
    def _replace(old_key: str, new_key: str):
        if new_key != old_key:
            state_dict[new_key] = state_dict[old_key]
            del state_dict[old_key]

    _replace('lm_head.bias', 'cls.predictions.bias')
    _replace('lm_head.dense.weight', 'cls.predictions.transform.dense.weight')
    _replace('lm_head.dense.bias', 'cls.predictions.transform.dense.bias')
    _replace('lm_head.layer_norm.weight', 'cls.predictions.transform.LayerNorm.weight')
    _replace('lm_head.layer_norm.bias', 'cls.predictions.transform.LayerNorm.bias')
    _replace('lm_head.decoder.weight', 'cls.predictions.decoder.weight')


def rename_tf_layernorm(state_dict):
    for k in list(state_dict.keys()):
        if 'LayerNorm.' in k:
            new_k = k.strip().replace('LayerNorm.beta', 'LayerNorm.bias')
            new_k = new_k.strip().replace('LayerNorm.gamma', 'LayerNorm.weight')
            state_dict[new_k] = state_dict[k]
            if new_k != k:
                del state_dict[k]


def load_params_choose_layers(prefix: str, state_dict: dict, mapper: dict, do_expand=False):
    """
        mapper: {old_layer: new_layer}
    """
    # fixed a bug
    # when mapper is for example {0: 0, 2: 1, 4: 2, 5: 3}
    # in the case, 4 -> 2 -> 1, causes error

    assert len(set(mapper.values())) == len(mapper), f"{set(mapper.values())} != {len(mapper)}"  # no overlap

    k_list = sorted([int(k) for k in mapper.keys()])
    mapper = {k: mapper[k] for k in k_list}

    if not len(mapper):
        return state_dict

    param_sorted = []

    for k in list(state_dict.keys()):
        if k.startswith(prefix):
            i_layer = k[len(prefix) + 1:]
            i_layer = int(i_layer.strip().split('.')[0])
            param_sorted.append((k, i_layer))
        else:
            param_sorted.append((k, -1))  # any is ok

    param_sorted = sorted(param_sorted, key=lambda p: p[1])
    param_sorted = [p[0] for p in param_sorted]

    for k in param_sorted:  # must start from lower layers
        if k.startswith(prefix):
            new_k = None
            for i in mapper.keys():
                if k.startswith(f'{prefix}.{i}.'):
                    new_k = k.replace(f'{prefix}.{i}.', f'{prefix}.{mapper[i]}.')
                    break

            if new_k:
                state_dict[new_k] = state_dict[k]

            if (new_k != k) and (not do_expand):
                del state_dict[k]

    return state_dict


def get_bert_config(encoder_rpath, num_hidden_layers=12, cross_start_at=12):
    """
    Args:
        cross_start_at: if it >= num_hidden_layers, no cross attn
    """
    if 'roberta' in encoder_rpath:
        config = RobertaConfig.from_json_file(os.path.join(encoder_rpath, 'config.json'))
    else:
        config = BertConfig.from_json_file(os.path.join(encoder_rpath, 'config.json'))

    # set configs
    config.num_hidden_layers = num_hidden_layers
    config.fusion_layer = cross_start_at
    config.embedding_dim = config.hidden_size

    return config


class AllGather(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, rank, world_size):
        output = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(output, tensor)
        ctx.rank = rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, 0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank: ctx.batch_size * (ctx.rank + 1)],
            None,
            None
        )


allgather = AllGather.apply


def build_mlp(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, input_dim * 2),
        nn.LayerNorm(input_dim * 2),
        nn.GELU(),
        nn.Linear(input_dim * 2, output_dim)
    )


def build_vision_encoder(config, load_params=False):
    """
    Args:
        load_params: False when building fine-tuning models
    """
    num_patches = (config['image_res'] // config['patch_size']) ** 2
    # image_res = config['image_res']
    # if isinstance(image_res, list):
    #     image_res = tuple(image_res)
    #     num_patches = image_res[0] * image_res[1] // config['patch_size'] ** 2
    # else:
    #     num_patches = (config['image_res'] // config['patch_size']) ** 2

    if config.get('use_clip_vit', False):  # good performance, but only base model available
        from models.clip_vit import CLIPVisionTransformer, interpolate_pos_embed

        vision_config = read_json(config['vision_config'])
        assert config['patch_size'] == vision_config['patch_size']
        vision_width = vision_config['vision_width']

        vision_encoder = CLIPVisionTransformer(image_size=config['image_res'], patch_size=vision_config['patch_size'],
                                               hidden_size=vision_config['vision_width'],
                                               hidden_act=vision_config['hidden_act'],
                                               num_attention_heads=vision_config['num_attention_heads'],
                                               attention_dropout=vision_config['attention_dropout'],
                                               intermediate_size=vision_config['intermediate_size'],
                                               num_hidden_layers=vision_config['num_hidden_layers'],
                                               local_attn_depth=vision_config['local_attn_depth'])

        if load_params:
            # download from https://huggingface.co/openai/clip-vit-base-patch16/tree/main
            state_dict_orig = torch.load(vision_config['ckpt'], map_location="cpu")
            state_dict = {}
            for k, v in state_dict_orig.items():
                if k.startswith('vision_model.'):
                    k = k[13:]
                    if k.startswith('embeddings.'):
                        k = k[11:]
                        k = k.replace('patch_embedding.weight', 'patch_embed.weight')
                        k = k.replace('position_embedding.weight', 'pos_embed.weight')

                    if k != 'position_ids':
                        state_dict[k] = v

            pos_embed_reshaped = interpolate_pos_embed(state_dict['pos_embed.weight'].unsqueeze(dim=0),
                                                       num_patches=num_patches, num_extra_tokens=1)
            state_dict['pos_embed.weight'] = pos_embed_reshaped.squeeze(dim=0)

            assert vision_config['num_hidden_layers'] in [6, 12], "param initialization not implemented"
            if vision_config['num_hidden_layers'] == 6:
                mapper = {1: 0, 3: 1, 5: 2, 7: 3, 9: 4, 11: 5}
                load_params_choose_layers('encoder.layers', state_dict, mapper)

    elif config.get('use_swin', False):
        from models.swin_transformer import SwinTransformer

        vision_config = read_json(config['vision_config'])
        assert config['image_res'] == vision_config['image_res']
        vision_width = vision_config['vision_width']

        vision_encoder = SwinTransformer(img_size=vision_config['image_res'],
                                         patch_size=4,
                                         in_chans=3,
                                         embed_dim=vision_config['embed_dim'],
                                         depths=vision_config['depths'],
                                         num_heads=vision_config['num_heads'],
                                         window_size=vision_config['window_size'],
                                         mlp_ratio=4.,
                                         qkv_bias=True,
                                         drop_rate=0.0,
                                         drop_path_rate=0.1,
                                         ape=False,
                                         patch_norm=True,
                                         use_checkpoint=False, add_cls=config.get('swin_add_cls', True))

        if load_params:
            from models.swin_transformer import load_pretrained_swin
            state_dict = load_pretrained_swin(vision_encoder, vision_config['ckpt'])

    elif config.get('use_beit_v2', False):

        vision_config = read_json(config['vision_config'])
        assert config['patch_size'] == vision_config['patch_size']
        vision_width = vision_config['vision_width']

        if 'base' in config['vision_config']:
            from models.beit2 import beit_base_patch16 as beit_model
        elif 'large' in config['vision_config']:
            from models.beit2 import beit_large_patch16 as beit_model
        else:
            raise ValueError
        use_mim = config['mim']['use_mim'] or config['itm']['use_mask']
        vision_encoder = beit_model(img_size=config['image_res'],
                                    drop_rate=0.0, drop_path_rate=0.1, attn_drop_rate=0.0,
                                    use_mean_pooling=True,
                                    init_scale=0.001,
                                    use_rel_pos_bias=True, use_abs_pos_emb=False,
                                    init_values=0.1, qkv_bias=True, local_attn_depth=config.get('local_attn_depth', -1),
                                    vision_num_hidden_layers=config.get('vision_num_hidden_layers', -1),
                                    use_mim=use_mim,
                                    mim_mask_ratio=config['mim']['mask_ratio'])

        if load_params:
            from models.beit2 import load_pretrained_beit2
            load_pretrained_beit2(vision_encoder, vision_config['ckpt'])

    else:
        raise ValueError

    if load_params and (not config.get('use_beit_v2', False)):
        print("### Load ViT: ", flush=True)
        msg = vision_encoder.load_state_dict(state_dict, strict=False)
        print("missing_keys: ", msg.missing_keys)
        print("unexpected_keys: ", msg.unexpected_keys)

    # set attrs
    vision_encoder.vision_width = vision_width

    return vision_encoder


def build_text_encoder(config, vision_width, load_text_params=False, use_mlm_loss=False, config_text=None):
    # if config_text is None:
    config_text = get_bert_config(config['text_encoder'], num_hidden_layers=config['text_num_hidden_layers'],
                                  cross_start_at=config['text_fusion_start_at'])
    # else:
    #     assert isinstance(config_text, BertConfig)

    tokenizer = build_tokenizer(config['text_encoder'])
    config_text.pad_token_id = tokenizer.pad_token_id

    config_text.text_encoder = config['text_encoder']
    config_text.hidden_dropout_prob = config.get('dropout', config_text.hidden_dropout_prob)  # changable dropout rate
    config_text.encoder_width = vision_width
    config_text.text_drop_path_rate = config.get('text_drop_path_rate', 0.0)
    config_text.cross_drop_path_rate = config.get('cross_drop_path_rate', 0.0)

    if use_mlm_loss:
        # assert load_text_params is True  # for domain pre-training
        if ('accelerator' in config.keys()) and (config['accelerator']['FP16_OPT_LEVEL'] != 'O0'):
            config_text.fp16 = True  # will use some operations to avoid gradient overflow

        if 'roberta' in config['text_encoder']:
            text_encoder = BertForMaskedLM(config=config_text)
        else:
            text_encoder = BertForMaskedLM(config=config_text)

    else:
        if 'roberta' in config['text_encoder']:
            text_encoder = BertModel(config=config_text, add_pooling_layer=False)
        else:
            text_encoder = BertModel(config=config_text, add_pooling_layer=False)

    missing_keys = []
    if load_text_params:
        print("### Initializing text encoder from ", os.path.join(config['text_encoder'], 'pytorch_model.bin'))
        state_dict = torch.load(os.path.join(config['text_encoder'], 'pytorch_model.bin'), map_location="cpu")
        if 'model' in state_dict.keys():
            state_dict = state_dict['model']

        prefix = "roberta.encoder.layer" if 'roberta' in config['text_encoder'] else 'bert.encoder.layer'
        if not use_mlm_loss:
            state_dict = {k.replace('roberta.', '').replace('bert.', ''): v for k, v in state_dict.items()}
            prefix = "encoder.layer"

        if 'roberta' in config['text_encoder']:
            pass
        else:
            if 'bert-base-uncased' in config['text_encoder']:
                rename_tf_layernorm(state_dict)
                if config_text.num_hidden_layers == 18:
                    assert config['text_fusion_start_at'] == 12
                    mapper = {6: 12, 7: 13, 8: 14, 9: 15, 10: 16, 11: 17}
                    load_params_choose_layers(prefix, state_dict, mapper, do_expand=True)

                else:
                    pass

            elif 'bert-large-uncased-12l' in config['text_encoder']:
                if config['text_num_hidden_layers'] == 18:
                    assert config['text_fusion_start_at'] == 12
                    mapper = {6: 12, 7: 13, 8: 14, 9: 15, 10: 16, 11: 17}
                    load_params_choose_layers(prefix, state_dict, mapper, do_expand=True)
                else:
                    raise NotImplementedError

            elif 'bert-large-uncased' in config['text_encoder']:
                rename_tf_layernorm(state_dict)

                if config_text.num_hidden_layers == 12:
                    mapper = {layer: i for i, layer in enumerate(list(range(1, 24 + 1, 2)))}
                    load_params_choose_layers(prefix, state_dict, mapper)

                else:
                    raise NotImplementedError

            elif 'chinese-roberta-wwm-ext' in config['text_encoder']:
                if config_text.num_hidden_layers == 6:
                    mapper = {1: 0, 3: 1, 5: 2, 7: 3, 9: 4, 11: 5}
                    load_params_choose_layers(prefix, state_dict, mapper)

            else:
                raise NotImplementedError

        if config.get('init_word_embeddings', False):
            print("### Train word_embeddings from scratch...", flush=True)
            for k in list(state_dict.keys()):
                if 'word_embeddings' in k:
                    del state_dict[k]

                elif k == 'cls.predictions.decoder.weight':
                    del state_dict[k]

                elif k == 'cls.predictions.bias':
                    del state_dict[k]

        msg = text_encoder.load_state_dict(state_dict, strict=False)
        print("missing_keys: ", msg.missing_keys, flush=True)
        print("unexpected_keys: ", msg.unexpected_keys, flush=True)

        missing_keys = msg.missing_keys

    return text_encoder, missing_keys


def load_pretrained(model, ckpt_rpath, config, is_eval=False, load_text=False):
    checkpoint = torch.load(ckpt_rpath, map_location='cpu')
    state_dict = checkpoint['model'] if 'model' in checkpoint.keys() else checkpoint

    if is_eval:
        return state_dict

    print("### Loading pretrained vision encoder", flush=True)

    if config.get('use_clip_vit', False):
        from models.clip_vit import interpolate_pos_embed
        del state_dict['vision_encoder.position_ids']
        num_patches = (config['image_res'] // config['patch_size']) ** 2
        pos_embed_reshaped = interpolate_pos_embed(state_dict['vision_encoder.pos_embed.weight'].unsqueeze(dim=0),
                                                   num_patches=num_patches, num_extra_tokens=1)
        state_dict['vision_encoder.pos_embed.weight'] = pos_embed_reshaped.squeeze(dim=0)

    elif config.get('use_swin', False) or config.get('use_swin_v2', False):
        from models.swin_transformer import load_pretrained_swin

        vision_state_dict = {}
        for k in list(state_dict.keys()):
            if k.startswith('vision_encoder.'):
                vision_state_dict[k[15:]] = state_dict[k]
                del state_dict[k]

        vision_state_dict = load_pretrained_swin(model.vision_encoder, state_dict=vision_state_dict)

        for k in vision_state_dict.keys():
            state_dict['vision_encoder.' + k] = vision_state_dict[k]

    elif config.get('use_beit_v2', False):
        from models.beit2 import interpolate_pos_embed

        vision_state_dict = {}
        for k in list(state_dict.keys()):
            if k.startswith('vision_encoder.'):
                vision_state_dict[k[15:]] = state_dict[k]
                del state_dict[k]

        vision_state_dict = interpolate_pos_embed(model.vision_encoder, vision_state_dict)
        for k in vision_state_dict.keys():
            state_dict['vision_encoder.' + k] = vision_state_dict[k]

    else:
        raise ValueError

    if load_text:
        print("### Loading pretrained text encoder", flush=True)
        for key in list(state_dict.keys()):
            if key.startswith('text_encoder.') or key.startswith('cross_encoder.'):
                encoder_key = key.replace('roberta.', '').replace('bert.', '').strip()
                state_dict[encoder_key] = state_dict[key]
                if encoder_key != key:
                    del state_dict[key]

    if config.get('init_timesformer', False):
        map_dict = {
            "temporal_norm1": "norm1",
            "time_attn": "attn",
            "temporal_norm2": "norm2",
            "temporal_mlp": "mlp",
            "time_gamma_1": "gamma_1",
            "time_gamma_2": "gamma_2"
        }
        for from_key, to_key in map_dict.items():
            for key in list(state_dict.keys()):
                if to_key in key:
                    state_dict[key.replace(to_key, from_key)] = copy.deepcopy(state_dict[key])

    return state_dict


class XVLMBase(nn.Module):
    def __init__(self, config=None, num_classes=None, load_vision_params=False, load_text_params=False, load_cross_params=False,
                 use_contrastive_loss=True, use_matching_loss=True, use_mlm_loss=False, use_bbox_loss=False,
                 config_text=None, pretraining=False):
        super().__init__()

        self.init_params = []
        self.vision_encoder = self.build_vision_encoder(config, load_params=load_vision_params)
        self.vision_width = self.vision_encoder.vision_width

        self.text_encoder, missing_keys = self.build_text_encoder(config, vision_width=self.vision_width,
                                                                  load_text_params=load_text_params,
                                                                  use_mlm_loss=use_mlm_loss,
                                                                  config_text=config_text)
        self.update_init_params([f'text_encoder.{k}' for k in missing_keys])

        self.build_cross_encoder(config, self.text_encoder.config, load_cross_params=load_cross_params)
        self.update_init_params([f'cross_encoder.{k}' for k in missing_keys])

        self.use_contrastive_loss = use_contrastive_loss
        self.itm_mask = config['itm']['use_mask']
        self.use_mlm = config['mlm']['is_mlm']

        if self.use_contrastive_loss:
            self.embed_dim = config['embed_dim']
            self.vision_proj = nn.Linear(self.vision_width, self.embed_dim)
            self.text_proj = nn.Linear(self.text_width, self.embed_dim)
            self.update_init_params(['vision_proj.' + n for n, _ in self.vision_proj.named_parameters()])
            self.update_init_params(['text_proj.' + n for n, _ in self.text_proj.named_parameters()])
            # self.temp = torch.ones([]) * config.model.temp
            if config.get('fix_temp', False):
                self.temp = torch.ones([]) * config['temp']

            else:
                self.temp = nn.Parameter(torch.ones([]) * config['temp'])
                self.update_init_params(['temp'])

        self.use_matching_loss = True
        if self.use_matching_loss:
            self.itm_head = build_mlp(input_dim=self.text_width, output_dim=2)
            self.update_init_params(['itm_head.' + n for n, _ in self.itm_head.named_parameters()])

        self.use_sdm_loss = config['sdm']['use_sdm']
        if self.use_sdm_loss:
            self.sdm_logit_scale = torch.ones([]) * (1 / config['temp'])
        self.use_id_loss = config['id']['use_id']
        if self.use_id_loss:
            self.classifier = nn.Linear(self.embed_dim, num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.01)
            nn.init.constant_(self.classifier.bias.data, val=0.1)

        self.use_mim = config['mim']['use_mim']
        
        if self.use_mim:
            self.mim_head = build_mlp(input_dim=self.text_width, output_dim=self.text_width)
            self.update_init_params(['mim_head.' + n for n, _ in self.mim_head.named_parameters()])

        self.tl_text_proj = nn.Linear(self.text_width, self.embed_dim)
        self.update_init_params(['tl_text_proj.' + n for n, _ in self.tl_text_proj.named_parameters()])

        if pretraining:
            print("Train From Scratch: ", sorted(self.init_params))
        else:
            self.init_params = []
        self.action_dim = config['embed_dim']
        self.num_actions = config.get('num_actions',1012) 
        self.top_k = config.get('top_k', 3)
        self.entropy_tau = config.get('entropy_tau', 1.0)
        self.projection_epsilon = config.get('projection_epsilon', 1e-6)
        self.action_mapper_input = config.get('action_mapper_input', 'projected')
        if self.action_mapper_input not in {'raw', 'projected'}:
            raise ValueError("action_mapper_input must be either 'raw' or 'projected'")

        self.action_mapper = nn.Sequential(
            nn.Linear(self.vision_width, self.vision_width),
            nn.GELU(),
            nn.Linear(self.vision_width, self.vision_width),
            nn.GELU(),
            nn.Linear(self.vision_width, self.num_actions),
        )
        self.update_init_params(['action_mapper.' + n for n, _ in self.action_mapper.named_parameters()])

        self.action_mapper_projected = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.num_actions),
        )
        self.update_init_params(['action_mapper_projected.' + n for n, _ in self.action_mapper_projected.named_parameters()])
        inactive_action_mapper = self.action_mapper if self.action_mapper_input == 'projected' else self.action_mapper_projected
        for param in inactive_action_mapper.parameters():
            param.requires_grad = False

        codebook = torch.randn(self.num_actions, self.action_dim)
        self.register_buffer("action_codebook", codebook)
        self.codebook_initialized = False
        self.current_epoch = 0
        self.max_epoch = 1

        self.action_temp = config.get('action_temp', 0.1)
        
        self.nsc = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim * 2),
            nn.Sigmoid()
        )
        self.update_init_params(['nsc.' + n for n, _ in self.nsc.named_parameters()])


    @torch.no_grad()
    def init_action_codebook(self, action_text_ids, action_text_atts):
        text_embeds = self.get_text_embeds(action_text_ids, action_text_atts)
        text_feats = self.get_text_feat(text_embeds) #(K,embed_dim)

        self.action_codebook.data.copy_(text_feats)
        self.codebook_initialized = True
       

    def decouple_features(self, image_embeds):
        v_raw = image_embeds[:, 0, :] 
        v_proj = self.vision_proj(v_raw)
        v_proj = F.normalize(v_proj, dim=-1)

        if self.action_mapper_input == 'projected':
            logits = self.action_mapper_projected(v_proj)
        else:
            logits = self.action_mapper(v_raw) 

        topk_values, topk_indices = torch.topk(logits, self.top_k, dim=-1)
        mask = torch.full_like(logits, float('-inf'))
        mask.scatter_(-1, topk_indices, topk_values)
        probs = F.softmax(mask / self.action_temp, dim=-1)
        entropy_probs = probs
        
        z_act_recon = torch.matmul(probs, self.action_codebook)
        z_act_recon = F.normalize(z_act_recon, dim=-1)
        
        projection_epsilon = self.projection_epsilon
        dot_prod = (v_proj * z_act_recon).sum(dim=-1, keepdim=True)
        norm_sq = z_act_recon.square().sum(dim=-1, keepdim=True)
        projection = dot_prod / (norm_sq + projection_epsilon) * z_act_recon

        entropy = compute_entropy(entropy_probs)
        omega = torch.exp(-entropy / self.entropy_tau).unsqueeze(-1)

        projection_fixed = projection.detach()
        effective_projection = omega * projection_fixed + (1 - omega) * projection

        z_id = v_proj - effective_projection
        z_id = F.normalize(z_id, dim=-1)
        
        z_concat = torch.cat([z_act_recon, z_id], dim=-1)
        gate_weights = self.nsc(z_concat)
        w_act, w_app = torch.chunk(gate_weights, 2, dim=-1)
        
        z_final = w_act * z_act_recon + w_app * z_id
        z_final = F.normalize(z_final, dim=-1)

        return z_final, z_id, z_act_recon, v_proj, logits
    
    def build_vision_encoder(self, config, load_params=False):
        return build_vision_encoder(config, load_params=load_params)

    def build_text_encoder(self, config, vision_width, load_text_params=False, use_mlm_loss=False, config_text=None):
        """
        in XVLMBase, text_encoder includes cross encoder parts
        """
        text_encoder, missing_keys = build_text_encoder(config, vision_width, load_text_params=load_text_params,
                                                        use_mlm_loss=use_mlm_loss, config_text=config_text)

        # set attrs
        self.vocab_size = text_encoder.config.vocab_size
        self.num_text_layers = text_encoder.config.fusion_layer
        self.text_width = text_encoder.config.hidden_size  # i.e. cross_width
        print("### X-VLM, num_text_layers: ", self.num_text_layers, flush=True)

        return text_encoder, missing_keys

    def build_cross_encoder(self, config, config_text, load_cross_params=False):
        """
        in XVLMBase, text_encoder includes cross encoder parts
        """

        # set attrs
        self.num_cross_layers = self.text_encoder.config.num_hidden_layers - self.num_text_layers
        self.cross_width = self.text_encoder.config.hidden_size
        print("### X-VLM, num_cross_layers: ", self.num_cross_layers, flush=True)

        if self.num_cross_layers == 0:
            self.cross_attn = nn.MultiheadAttention(self.vision_width,
                                                    self.vision_width // 64,
                                                    batch_first=True)
            self.cross_modal_transformer = Transformer(width=self.vision_width,
                                                       layers=4,
                                                       heads=self.vision_width // 64)
            scale = self.cross_modal_transformer.width ** -0.5

            self.ln_pre_t = LayerNorm(self.vision_width)
            self.ln_pre_i = LayerNorm(self.vision_width)
            self.ln_post = LayerNorm(self.vision_width)

            proj_std = scale * ((2 * self.cross_modal_transformer.layers) ** -0.5)
            attn_std = scale
            fc_std = (2 * self.cross_modal_transformer.width) ** -0.5
            for block in self.cross_modal_transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            # init cross attn
            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

    def update_init_params(self, missing_keys=None):
        if missing_keys is not None:
            assert isinstance(missing_keys, list)
            for k in missing_keys:
                if k not in self.init_params:
                    self.init_params.append(k)

        # check
        named_parameters = set([n for n, _ in self.named_parameters()])
        for n in set(self.init_params):
            if n not in named_parameters:
                self.init_params.remove(n)

    def load_pretrained(self, ckpt_rpath, config, is_eval=False):
        ckpt_rpath = ckpt_rpath
        print('load checkpoint from %s' % ckpt_rpath)
        state_dict = load_pretrained(self, ckpt_rpath, config, is_eval=is_eval, load_text=True)

        # rename weights to fit the model
        temp_dict = {}
        for i in range(6):
            for key, value in state_dict.items():
                if key.startswith(f"cross_encoder.encoder.layer.{i}."):
                    if config['mlm']['is_mlm']:
                        new_key = key.replace(f"cross_encoder.encoder.layer.{i}", f"text_encoder.bert.encoder.layer.{i + 12}")
                    else:
                        new_key = key.replace(f"cross_encoder.encoder.layer.{i}",
                                              f"text_encoder.encoder.layer.{i + 12}")
                    temp_dict[new_key] = value
        keys_to_remove = [key for key in state_dict.keys() if key.startswith("cross_encoder.encoder.layer.")]
        for key in keys_to_remove:
            del state_dict[key]

        if config['mlm']['is_mlm']:
            keys_to_remove = []
            for key, value in state_dict.items():
                if key.startswith(f"text_encoder."):
                    new_key = key.replace(f"text_encoder", f"text_encoder.bert")
                    temp_dict[new_key] = value
                    keys_to_remove.append(key)
                elif key.startswith(f"mlm_head."):
                    new_key = key.replace(f"mlm_head", f"text_encoder.mlm_head")
                    temp_dict[new_key] = value
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                del state_dict[key]

        # Initialize tl_text_proj's weights using the pre-trained weights from text_proj
        temp_dict["tl_text_proj.weight"] = state_dict['text_proj.weight'].detach().clone()
        temp_dict["tl_text_proj.bias"] = state_dict['text_proj.bias'].detach().clone()

        state_dict.update(temp_dict)

        if hasattr(self, 'absolute_frame_pos_embed') and ('absolute_frame_pos_embed' in state_dict.keys()):
            pretrained = state_dict['absolute_frame_pos_embed']
            if pretrained.shape != self.absolute_frame_pos_embed.shape:
                frame_len = min(pretrained.shape[1], self.absolute_frame_pos_embed.shape[1])
                self.absolute_frame_pos_embed.data[:, :frame_len, :, :] = pretrained.data[:, :frame_len, :, :]
                print(
                    f"load absolute_frame_pos_embed[:{frame_len}] ({pretrained.shape[1]}/{self.absolute_frame_pos_embed.shape[1]})",
                    flush=True)
                del state_dict['absolute_frame_pos_embed']

        msg = self.load_state_dict(state_dict, strict=False)
        print("unexpected_keys: ", msg.unexpected_keys)
        missing_keys = [p for p in msg.missing_keys]  # if 'vision_encoder' not in p
        self.update_init_params(missing_keys)
        print("train from scratch: ", sorted(self.init_params))

    def load_finetune(self, config, is_eval=False):
        ckpt_rpath = config.model.resume
        print('load checkpoint from %s' % ckpt_rpath)
        state_dict = load_pretrained(self, ckpt_rpath, config, is_eval=is_eval, load_text=True)['state_dict']

        # rename weights to fit the model
        temp_dict = {}
        keys_to_remove = []
        for key, value in state_dict.items():
            if key.startswith(f"text_encoder.bert"):
                new_key = key.replace(f"text_encoder.bert", f"text_encoder")
                temp_dict[new_key] = value
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del state_dict[key]
        state_dict.update(temp_dict)

        if hasattr(self, 'absolute_frame_pos_embed') and ('absolute_frame_pos_embed' in state_dict.keys()):
            pretrained = state_dict['absolute_frame_pos_embed']
            if pretrained.shape != self.absolute_frame_pos_embed.shape:
                frame_len = min(pretrained.shape[1], self.absolute_frame_pos_embed.shape[1])
                self.absolute_frame_pos_embed.data[:, :frame_len, :, :] = pretrained.data[:, :frame_len, :, :]
                print(
                    f"load absolute_frame_pos_embed[:{frame_len}] ({pretrained.shape[1]}/{self.absolute_frame_pos_embed.shape[1]})",
                    flush=True)
                del state_dict['absolute_frame_pos_embed']

        msg = self.load_state_dict(state_dict, strict=False)
        print("unexpected_keys: ", msg.unexpected_keys)
        missing_keys = [p for p in msg.missing_keys]  # if 'vision_encoder' not in p
        self.update_init_params(missing_keys)
        print("train from scratch: ", sorted(self.init_params))


    def get_image_feat(self, image_embeds):
        return F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)
        # x = self.avgpool(image_embeds[:, 1:, ].transpose(1, 2))
        # x = x.transpose(1, 2)
        # x = torch.cat([x[:, 0, :], x[:, 1, :], x[:, 2, :]], dim=1)
        # image_feat = self.vision_proj(x)
        # return image_feat


    def get_text_feat(self, text_embeds):
        return F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)
        # x = self.avgpool(text_embeds[:, 1:, ].transpose(1, 2))  # B C 3
        # x = x.transpose(1, 2)  # B 3 C
        # x = torch.cat([text_embeds[:, 0, :], x[:, 0, :], x[:, 1, :], x[:, 2, :]], dim=1)
        # text_feat = self.text_proj(x)
        # return text_feat



    def get_image_embeds(self, image, image_atts=None, idx_to_group_img=None, output_hidden_states=None,
                         output_attentions=None, is_mask=False):
        assert image.dim() == 4
        assert output_hidden_states == output_attentions

        if idx_to_group_img is None:
            image_embeds = self.vision_encoder(image, output_hidden_states=output_hidden_states,
                                               output_attentions=output_attentions, is_mask=is_mask)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
            return image_embeds, image_atts  # full attention

        else:  # image < bsz
            if output_attentions or output_hidden_states:
                raise NotImplementedError

            if image_atts is None:
                image_embeds_fullatts = self.vision_encoder(image)

                image_embeds_fullatts = torch.gather(image_embeds_fullatts, dim=0,
                                                     index=idx_to_group_img.view(-1, 1, 1).expand(
                                                         -1, image_embeds_fullatts.shape[1],
                                                         image_embeds_fullatts.shape[2]))  # expend to bsz

                image_atts = torch.ones(image_embeds_fullatts.size()[:-1], dtype=torch.long).to(image.device)

                return image_embeds_fullatts, image_atts

            else:
                assert image_atts.size(0) == idx_to_group_img.size(0)  # bsz
                image_embeds, image_embeds_fullatts = \
                    self.vision_encoder(image, idx_to_group_img=idx_to_group_img, image_atts=image_atts)

                image_embeds_fullatts = torch.gather(image_embeds_fullatts, dim=0,
                                                     index=idx_to_group_img.view(-1, 1, 1).expand(
                                                         -1, image_embeds_fullatts.shape[1],
                                                         image_embeds_fullatts.shape[2]))

                return image_embeds, image_atts, image_embeds_fullatts

    def get_vision_embeds(self, image, image_atts=None, idx_to_group_img=None, output_hidden_states=None,
                          output_attentions=None, is_mask=False):
        """
        vision_embeds: cls + patch embeds
        """
        assert output_hidden_states == output_attentions

        if image.dim() == 5:  # encode video
            # image: (bsz, frame_len, c, h, w)
            assert idx_to_group_img is None, "not supported"
            return self.get_frame_embeds(image, output_hidden_states=output_hidden_states,
                                         output_attentions=output_attentions)

        assert image.dim() == 4
        return self.get_image_embeds(image, image_atts=image_atts, idx_to_group_img=idx_to_group_img,
                                     output_hidden_states=output_hidden_states, output_attentions=output_attentions,
                                     is_mask=is_mask)

    def get_text_embeds(self, text_ids, text_atts, output_hidden_states=None, output_attentions=None):
        assert output_hidden_states == output_attentions

        encoder = self.text_encoder.bert if hasattr(self.text_encoder, 'bert') else self.text_encoder

        outputs = encoder(text_ids, attention_mask=text_atts, return_dict=True, mode='text',
                          output_hidden_states=output_hidden_states, output_attentions=output_attentions)

        if output_hidden_states:
            assert len(outputs.hidden_states) == len(outputs.attentions) + 1
            return {'last_hidden_state': outputs.last_hidden_state,
                    'hidden_states': outputs.hidden_states,
                    'attentions': outputs.attentions}

        else:
            return outputs.last_hidden_state

    def get_text_embeds_12L(self, text_ids, text_atts, output_hidden_states=None, output_attentions=None):
        assert output_hidden_states == output_attentions

        encoder = self.text_encoder.bert if hasattr(self.text_encoder, 'bert') else self.text_encoder
        outputs = encoder(text_ids,
                          attention_mask=text_atts,
                          encoder_hidden_states=None,
                          encoder_attention_mask=None,
                          output_hidden_states=output_hidden_states,
                          output_attentions=output_attentions,
                          return_dict=True)

        if output_hidden_states:
            assert len(outputs.hidden_states) == len(outputs.attentions) + 1
            return {'last_hidden_state': outputs.last_hidden_state,
                    'hidden_states': outputs.hidden_states,
                    'attentions': outputs.attentions}

        else:
            return outputs.last_hidden_state

    def get_cross_embeds(self, image_embeds, image_atts, text_ids=None, text_embeds=None, text_atts=None,
                         output_hidden_states=None, output_attentions=None):
        assert text_atts is not None
        assert output_hidden_states == output_attentions

        encoder = self.text_encoder.bert if hasattr(self.text_encoder, 'bert') else self.text_encoder

        if text_embeds is not None:
            outputs = encoder(encoder_embeds=text_embeds,
                              attention_mask=text_atts,
                              encoder_hidden_states=image_embeds,
                              encoder_attention_mask=image_atts,
                              output_attentions=output_attentions,
                              output_hidden_states=output_hidden_states,
                              return_dict=True,
                              mode='fusion')

        elif text_ids is not None:
            outputs = encoder(text_ids,
                              attention_mask=text_atts,
                              encoder_hidden_states=image_embeds,
                              encoder_attention_mask=image_atts,
                              return_dict=True)

        else:
            raise ValueError

        if output_hidden_states:
            assert len(outputs.hidden_states) == len(outputs.attentions) + 1
            return {'last_hidden_state': outputs.last_hidden_state,
                    'hidden_states': outputs.hidden_states,
                    'attentions': outputs.attentions,
                    'cross_attentions': outputs.cross_attentions}

        return outputs.last_hidden_state

    def get_features(self, image_embeds=None, text_embeds=None, is_tl=False):
        if image_embeds is None:
            if is_tl:
                return F.normalize(self.tl_text_proj(text_embeds[:, 0, :]), dim=-1)
            else:
                return F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)
        elif text_embeds is None:
            return F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)
        else:
            return F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1), \
                F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)

    def get_contrastive_loss(self, image_feat, text_feat, idx=None):
        """
        Args:
            image_feat, text_feat: normalized

        Returns: contrastive loss

        """
        assert image_feat.size(-1) == self.embed_dim
        assert text_feat.size(-1) == self.embed_dim

        image_feat_all = allgather(image_feat, torch.distributed.get_rank(), torch.distributed.get_world_size())
        text_feat_all = allgather(text_feat, torch.distributed.get_rank(), torch.distributed.get_world_size())
        logits = image_feat_all @ text_feat_all.t() / self.temp

        bsz = image_feat_all.shape[0]

        if idx is None:
            labels = torch.arange(bsz, device=image_feat.device)
            loss_i2t = F.cross_entropy(logits, labels)
            loss_t2i = F.cross_entropy(logits.t(), labels)

        else:
            idx = idx.view(-1, 1)
            assert idx.size(0) == image_feat.size(0)
            idx_all = allgather(idx, torch.distributed.get_rank(), torch.distributed.get_world_size())
            pos_idx = torch.eq(idx_all, idx_all.t()).float()
            labels = pos_idx / pos_idx.sum(1, keepdim=True)
            loss_i2t = -torch.sum(F.log_softmax(logits, dim=1) * labels, dim=1).mean()
            loss_t2i = -torch.sum(F.log_softmax(logits.t(), dim=1) * labels, dim=1).mean()

        return (loss_i2t + loss_t2i) / 2

    def process_attn(self, attn):
        return (attn * self.mim_scale).softmax(dim=-1)

    def get_kd_loss(self, tea_attn, tea_cross_attn, stu_attn, stu_cross_attn):
        stu_attn = self.process_attn(stu_attn)
        stu_cross_attn = self.process_attn(stu_cross_attn)
        tea_attn = self.process_attn(tea_attn)
        tea_cross_attn = self.process_attn(tea_cross_attn)
        loss_attn = nn.KLDivLoss(reduction="none")(stu_attn.log(), tea_attn).sum(-1)
        loss_cross_attn = nn.KLDivLoss(reduction="none")(stu_cross_attn.log(), tea_cross_attn).sum(-1)
        loss_attn = loss_attn.mean()
        loss_cross_attn = loss_cross_attn.mean()
        return loss_attn + loss_cross_attn

    def get_d_mim_loss(self, tea_feats, stu_feats, loss_type='cos'):
        if loss_type == 'smooth_l1':
            tea_feats = F.layer_norm(tea_feats, (tea_feats.shape[-1],))
            criterion = nn.SmoothL1Loss(beta=0.5, reduction='mean')
            stu_feats = self.mim_head(stu_feats)
            loss = criterion(tea_feats, stu_feats)
        elif loss_type == 'cos':
            tea_feats = tea_feats / tea_feats.norm(dim=-1, keepdim=True)
            stu_feats = self.mim_head(stu_feats)
            stu_feats = stu_feats / stu_feats.norm(dim=-1, keepdim=True)
            loss = 1 - (stu_feats * tea_feats).sum(dim=-1)
            loss = loss.mean()
        else:
            raise NotImplementedError("{} is not implemented".format(loss_type))
        return loss

    def get_hard_negatives(self, image_feat, text_feat, idx=None):
        bs = image_feat.size(0)
        with torch.no_grad():
            sim_i2t = image_feat @ text_feat.t() / self.temp
            sim_t2i = text_feat @ image_feat.t() / self.temp

            weights_i2t = F.softmax(sim_i2t, dim=1) + 1e-5
            weights_t2i = F.softmax(sim_t2i, dim=1) + 1e-5

            if idx is None:
                weights_i2t.fill_diagonal_(0)
                weights_t2i.fill_diagonal_(0)
            else:
                idx = idx.view(-1, 1)
                assert idx.size(0) == bs
                mask = torch.eq(idx, idx.t())
                weights_i2t.masked_fill_(mask, 0)
                weights_t2i.masked_fill_(mask, 0)

        image_neg_idx = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_neg_idx.append(neg_idx)

        text_neg_idx = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_neg_idx.append(neg_idx)

        return image_neg_idx, text_neg_idx

    def get_neg_embeddings(self, image_embeds, image_atts, image_feat, text_embeds, text_atts, text_feat, idx=None):
        image_neg_idx, text_neg_idx = self.get_hard_negatives(image_feat, text_feat, idx=idx)

        bs = image_feat.size(0)
        image_embeds_neg = []
        image_atts_neg = []
        for b in range(bs):
            neg_idx = image_neg_idx[b]
            image_embeds_neg.append(image_embeds[neg_idx])
            image_atts_neg.append(image_atts[neg_idx])

        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)
        image_atts_neg = torch.stack(image_atts_neg, dim=0)

        text_embeds_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = text_neg_idx[b]
            text_embeds_neg.append(text_embeds[neg_idx])
            text_atts_neg.append(text_atts[neg_idx])

        text_embeds_neg = torch.stack(text_embeds_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)
        return image_embeds_neg, image_atts_neg, text_embeds_neg, text_atts_neg

    def get_matching_loss(self, image_embeds, image_atts, image_feat, text_embeds, text_atts, text_feat, idx=None):
        """
        Matching Loss with hard negatives
        """
        bs = image_feat.size(0)
        image_embeds_neg, image_atts_neg, text_embeds_neg, text_atts_neg = (
            self.get_neg_embeddings(image_embeds, image_atts, image_feat, text_embeds, text_atts, text_feat, idx))

        text_embeds_all = torch.cat([text_embeds, text_embeds_neg], dim=0)
        text_atts_all = torch.cat([text_atts, text_atts_neg], dim=0)
        image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)
        image_atts_all = torch.cat([image_atts_neg, image_atts], dim=0)

        cross_pos = self.get_cross_embeds(image_embeds, image_atts, text_embeds=text_embeds, text_atts=text_atts)[:, 0, :]
        cross_neg = self.get_cross_embeds(image_embeds_all, image_atts_all, text_embeds=text_embeds_all,
                                          text_atts=text_atts_all)[:, 0, :]

        output = self.itm_head(torch.cat([cross_pos, cross_neg], dim=0))
        itm_labels = torch.cat([torch.ones(bs, dtype=torch.long),
                                torch.zeros(2 * bs, dtype=torch.long)], dim=0).to(image_embeds.device)

        return F.cross_entropy(output, itm_labels)



    def get_matching_loss_hard(self, image_embeds, image_atts, image_embeds_hard, image_atts_hard,
                               text_embeds, text_atts, text_embeds_hard, text_atts_hard):
        bs = image_embeds.size(0)
        image_embeds = torch.cat([image_embeds, image_embeds_hard], dim=0)
        image_atts = torch.cat([image_atts, image_atts_hard], dim=0)
        text_embeds = torch.cat([text_embeds_hard, text_embeds], dim=0)
        text_atts = torch.cat([text_atts_hard, text_atts], dim=0)

        cross_hard = self.get_cross_embeds(image_embeds, image_atts, text_embeds=text_embeds,
                                          text_atts=text_atts)[:, 0, :]
        output = self.itm_head(cross_hard)

        # Matching Loss with only hard negatives for pab
        itm_labels = torch.cat([torch.zeros(bs, dtype=torch.long),
                                torch.zeros(bs, dtype=torch.long)], dim=0).to(image_embeds.device)
        itm_loss = F.cross_entropy(output, itm_labels)
        return itm_loss



    def get_mlm_loss(self, text_ids_masked, text_atts, image_embeds, image_atts, masked_pos, masked_ids):
        return self.text_encoder(text_ids_masked,
                                 attention_mask=text_atts,
                                 encoder_hidden_states=image_embeds,
                                 encoder_attention_mask=image_atts,
                                 return_dict=True,
                                 labels=masked_ids,
                                 masked_pos=masked_pos).loss

    def get_sdm_loss(self, image_feats, text_feats, idx, logit_scale, image_id=None, factor=0.3, epsilon=1e-8):
        """
        Similarity Distribution Matching
        """
        batch_size = image_feats.shape[0]
        idx = idx.reshape((batch_size, 1))  # make sure pid size is [batch_size, 1]
        idx_dist = idx - idx.t()
        labels = (idx_dist == 0).float()

        if image_id != None:
            # print("Mix PID and ImageID to create soft label.")
            image_id = image_id.reshape((-1, 1))
            image_id_dist = image_id - image_id.t()
            image_id_mask = (image_id_dist == 0).float()
            labels = (labels - image_id_mask) * factor + image_id_mask
            # labels = (labels + image_id_mask) / 2

        image_norm = image_feats / image_feats.norm(dim=1, keepdim=True)
        text_norm = text_feats / text_feats.norm(dim=1, keepdim=True)

        t2i_cosine_theta = text_norm @ image_norm.t()
        i2t_cosine_theta = t2i_cosine_theta.t()

        text_proj_image = logit_scale * t2i_cosine_theta
        image_proj_text = logit_scale * i2t_cosine_theta

        # normalize the true matching distribution
        labels_distribute = labels / labels.sum(dim=1)

        i2t_pred = F.softmax(image_proj_text, dim=1)
        i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
        t2i_pred = F.softmax(text_proj_image, dim=1)
        t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))

        loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

        return loss

    def get_id_loss(self, image_feats, text_feats, tl_text_feats, labels):
        """
        Instance loss proposed at http://arxiv.org/abs/1711.05535
        """
        image_logits = self.classifier(image_feats)
        text_logits = self.classifier(text_feats)
        tl_text_logits = self.classifier(tl_text_feats)
        criterion = nn.CrossEntropyLoss(reduction="mean")
        loss = criterion(image_logits, labels) + criterion(text_logits, labels) + criterion(tl_text_logits, labels)
        return loss / 3


class XVLMPlusBase(XVLMBase):
    """
    Separate text encoder and cross encoder, making the text encoder easier to be replaced.
    Re-implement func build_text_encoder to support any type of text encoder
    """

    def __init__(self, config, load_vision_params=False, load_text_params=False, load_cross_params=False,
                 use_contrastive_loss=False, use_matching_loss=False, use_mlm_loss=False, use_bbox_loss=False,
                 pretraining=False):

        super().__init__(config, load_vision_params=load_vision_params, load_text_params=load_text_params,
                         load_cross_params=load_cross_params,
                         use_contrastive_loss=use_contrastive_loss, use_matching_loss=use_matching_loss,
                         use_mlm_loss=False, use_bbox_loss=use_bbox_loss,
                         config_text=None, pretraining=pretraining)

        self.use_mlm_loss = use_mlm_loss
        if use_mlm_loss:
            from models.xbert import BertOnlyMLMHead
            self.mlm_head = BertOnlyMLMHead(self.text_encoder.config)
            self.update_init_params(['mlm_head.' + n for (n, _) in self.mlm_head.named_parameters()])
            # self.tie_text_and_cross_wordemb()

        if pretraining:
            print("### Train From Scratch: ", sorted(self.init_params))
        else:
            self.init_params = []

    def build_text_encoder(self, config, vision_width, load_text_params=False, use_mlm_loss=False, config_text=None):
        config_text = get_bert_config(config['text_encoder'], num_hidden_layers=config['text_num_hidden_layers'],
                                      cross_start_at=config['text_num_hidden_layers'])

        text_encoder, missing_keys = build_text_encoder(config, vision_width, load_text_params=load_text_params,
                                                        use_mlm_loss=use_mlm_loss, config_text=config_text)

        # set attrs
        self.vocab_size = text_encoder.config.vocab_size
        self.num_text_layers = text_encoder.config.fusion_layer
        self.text_width = text_encoder.config.hidden_size  # i.e. cross_width
        print("### X-VLM, num_text_layers: ", self.num_text_layers, flush=True)

        return text_encoder, missing_keys

    def build_cross_encoder(self, config, config_text, load_cross_params=False):
        config_cross = get_bert_config(config['cross_encoder'], num_hidden_layers=config['cross_num_hidden_layers'],
                                       cross_start_at=0)

        if config_text.hidden_size != config_cross.hidden_size:
            raise ValueError

        config_cross.pad_token_id = config_text.pad_token_id
        config_cross.vocab_size = config_text.vocab_size
        config_cross.embedding_dim = config_text.embedding_dim
        config_cross.encoder_width = config_text.encoder_width

        assert 'cross_drop_path_rate' not in config, "notimplemented"

        cross_encoder = BertModel(config=config_cross, add_pooling_layer=False, add_embeddings_layer=False)

        self.num_cross_layers = cross_encoder.config.num_hidden_layers
        self.cross_width = cross_encoder.config.hidden_size
        print("### X-VLM, num_cross_layers: ", self.num_cross_layers, flush=True)

        missing_keys = []
        if load_cross_params:
            print("### Initializing cross encoder from ", os.path.join(config['cross_encoder'], 'pytorch_model.bin'))
            state_dict = torch.load(os.path.join(config['cross_encoder'], 'pytorch_model.bin'), map_location="cpu")
            if 'model' in state_dict.keys():
                state_dict = state_dict['model']

            state_dict = {k.replace('bert.', ''): v for k, v in state_dict.items()}
            prefix = "encoder.layer"

            if 'bert-base-uncased' in config['cross_encoder']:
                rename_tf_layernorm(state_dict)
                if config_cross.num_hidden_layers == 6:
                    mapper = {6: 0, 7: 1, 8: 2, 9: 3, 10: 4, 11: 5}
                    load_params_choose_layers(prefix, state_dict, mapper)
                else:
                    raise NotImplementedError

            elif 'bert-large-uncased-12l' in config['cross_encoder']:
                if config_cross.num_hidden_layers == 6:
                    mapper = {6: 0, 7: 1, 8: 2, 9: 3, 10: 4, 11: 5}
                    load_params_choose_layers(prefix, state_dict, mapper)
                else:
                    raise NotImplementedError

            else:
                raise NotImplementedError

            # del
            for k in list(state_dict.keys()):
                if 'word_embeddings' in k:
                    del state_dict[k]
                elif k == 'cls.predictions.decoder.weight':
                    del state_dict[k]
                elif k == 'cls.predictions.bias':
                    del state_dict[k]

            msg = cross_encoder.load_state_dict(state_dict, strict=False)
            print("missing_keys: ", msg.missing_keys, flush=True)
            print("unexpected_keys: ", msg.unexpected_keys, flush=True)

            missing_keys = [f'cross_encoder.{k}' for k in msg.missing_keys]

        return cross_encoder, missing_keys

    def tie_text_and_cross_wordemb(self):
        # # tie word embeddings
        # self.mlm_head.predictions.decoder.weight = self.text_encoder.embeddings.word_embeddings.weight
        # self.mlm_head.predictions.decoder.bias = self.text_encoder.embeddings.word_embeddings.bias
        # self.init_params.remove('mlm_head.predictions.decoder.weight')
        # self.init_params.remove('mlm_head.predictions.decoder.bias')
        raise NotImplementedError("implement it in the domain pretraining model")

    def load_pretrained_xvlm(self, xvlm_ckpt_rpath, config, is_eval=False):
        print('### Loading X-VLM checkpoint from %s' % xvlm_ckpt_rpath, flush=True)
        state_dict = load_pretrained(self, xvlm_ckpt_rpath, config, is_eval=is_eval, load_text=True)

        if hasattr(self, 'absolute_frame_pos_embed') and ('absolute_frame_pos_embed' in state_dict.keys()):
            pretrained = state_dict['absolute_frame_pos_embed']
            if pretrained.shape != self.absolute_frame_pos_embed.shape:
                frame_len = min(pretrained.shape[1], self.absolute_frame_pos_embed.shape[1])
                self.absolute_frame_pos_embed.data[:, :frame_len, :, :] = pretrained.data[:, :frame_len, :, :]
                print(
                    f"load absolute_frame_pos_embed[:{frame_len}] ({pretrained.shape[1]}/{self.absolute_frame_pos_embed.shape[1]})",
                    flush=True)
                del state_dict['absolute_frame_pos_embed']

        replace_text_encoder = config.get('replace_text_encoder', False)
        num_text_layers = config['xvlm_ckpt_text_num_hidden_layers']
        if not is_eval:
            for k in list(state_dict.keys()):
                if k.startswith('text_encoder.'):
                    if 'layer' in k:
                        encoder_keys = k.split('.')
                        layer_num = int(encoder_keys[3])
                        if layer_num < num_text_layers:
                            if replace_text_encoder:
                                del state_dict[k]  # del text encoder
                        else:
                            new_k = k.replace('text_encoder.', 'cross_encoder.')
                            new_k = new_k.replace(f'layer.{layer_num}.', f'layer.{layer_num - num_text_layers}.')
                            state_dict[new_k] = state_dict[k]
                            del state_dict[k]
                    else:

                        if replace_text_encoder:
                            if 'embeddings' in k:
                                del state_dict[k]
                                continue
                            elif k.startswith('text_encoder.cls.predictions.decoder'):
                                del state_dict[k]
                                continue
                            elif k.startswith('text_encoder.cls.predictions.bias'):
                                del state_dict[k]
                                continue

                        if k.startswith('text_encoder.cls.'):
                            new_k = k.replace('text_encoder.cls.', 'mlm_head.').strip()
                            state_dict[new_k] = state_dict[k]
                            del state_dict[k]

        return state_dict

    def load_pretrained_text(self, text_ckpt_rpath, config, is_eval=False):
        assert os.path.exists(text_ckpt_rpath)
        print('### Loading Text-Encoder checkpoint from %s' % text_ckpt_rpath)
        checkpoint = torch.load(text_ckpt_rpath, map_location='cpu')
        text_enc_state_dict = checkpoint['model'] if 'model' in checkpoint.keys() else checkpoint
        rename_tf_layernorm(text_enc_state_dict)

        for k in list(text_enc_state_dict.keys()):
            if k.startswith('roberta.') or k.startswith('bert.'):
                encoder_key = 'text_encoder.' + k.replace('roberta.', '').replace('bert.', '').strip()

                text_enc_state_dict[encoder_key] = text_enc_state_dict[k]
                if encoder_key != k:
                    del text_enc_state_dict[k]

        return text_enc_state_dict

    def load_pretrained(self, xvlm_ckpt_rpath, config, is_eval=False, text_ckpt_rpath=''):
        """
        xvlm_ckpt_rpath: of XVLMBase
        """
        if config.get('is_xvlm_ckpt', False):
            state_dict = self.load_pretrained_xvlm(xvlm_ckpt_rpath, config, is_eval=is_eval)
        else:
            state_dict = load_pretrained(self, xvlm_ckpt_rpath, config, is_eval=is_eval, load_text=True)

        if config.get('replace_text_encoder', False):
            text_enc_state_dict = self.load_pretrained_text(text_ckpt_rpath, config, is_eval=is_eval)
            for k, v in text_enc_state_dict.items():
                state_dict[k] = v

        msg = self.load_state_dict(state_dict, strict=False)
        print("unexpected_keys: ", msg.unexpected_keys)
        missing_keys = [p for p in msg.missing_keys]  #  if 'vision_encoder' not in p
        self.update_init_params(missing_keys)
        print("train from scratch: ", sorted(self.init_params))

    def get_text_embeds(self, text_ids, text_atts, output_hidden_states=None, output_attentions=None):
        assert output_hidden_states == output_attentions

        outputs = self.text_encoder(text_ids, attention_mask=text_atts, return_dict=True,
                                    output_hidden_states=output_hidden_states, output_attentions=output_attentions)

        if output_hidden_states:
            assert len(outputs.hidden_states) == len(outputs.attentions) + 1
            return {'last_hidden_state': outputs.last_hidden_state,
                    'hidden_states': outputs.hidden_states,
                    'attentions': outputs.attentions}

        return outputs.last_hidden_state

    def get_text_embeds_12L(self, text_ids, text_atts, output_hidden_states=None, output_attentions=None):
        return self.get_text_embeds(text_ids, text_atts, output_hidden_states=output_hidden_states,
                                    output_attentions=output_attentions)

    def get_cross_embeds(self, image_embeds, image_atts, text_ids=None, text_embeds=None, text_atts=None,
                         output_hidden_states=None, output_attentions=None):
        assert text_atts is not None
        assert output_hidden_states == output_attentions

        if text_embeds is None:
            assert text_ids is not None
            assert not output_hidden_states, "please manually split get_text_embeds and get_cross_embeds"
            text_embeds = self.get_text_embeds(text_ids, text_atts)

        outputs = self.cross_encoder(encoder_embeds=text_embeds,
                                     attention_mask=text_atts,
                                     encoder_hidden_states=image_embeds,
                                     encoder_attention_mask=image_atts,
                                     return_dict=True,
                                     mode='fusion',
                                     output_hidden_states=output_hidden_states,
                                     output_attentions=output_attentions)

        if output_hidden_states:
            assert len(outputs.hidden_states) == len(outputs.attentions) + 1
            return {'last_hidden_state': outputs.last_hidden_state,
                    'hidden_states': outputs.hidden_states,
                    'attentions': outputs.attentions}

        return outputs.last_hidden_state

    def get_mlm_loss(self, text_ids_masked, text_atts, image_embeds, image_atts, masked_pos, masked_ids):
        def gather_seq_out_by_pos(seq, pos):
            return torch.gather(seq, 1, pos.unsqueeze(2).expand(-1, -1, seq.size(-1)))

        sequence_output = self.get_cross_embeds(image_embeds, image_atts, text_ids=text_ids_masked, text_atts=text_atts)

        # sequence_output, (bs, len, 768)
        # masked_pos, (bs, n_mask)
        sequence_output = gather_seq_out_by_pos(sequence_output, masked_pos)
        # sequence_output, (bs, n_mask, 768)

        prediction_scores = self.mlm_head(sequence_output)

        loss_fct = CrossEntropyLoss()  # -100 index = padding token
        masked_lm_loss = loss_fct(prediction_scores.view(-1, self.vocab_size), masked_ids.view(-1))

        return masked_lm_loss


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

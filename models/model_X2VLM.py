import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from models.xvlm import XVLMBase, XVLMPlusBase
from models.xvlm import QuickGELU, LayerNorm
from models.pose import Block, ConvExpandReduce


class Search(XVLMBase):
    def __init__(self, config, num_classes=None):
        # Initialize the base XVLM model for parallel data
        super().__init__(config, num_classes=num_classes, load_vision_params=False, load_text_params=False,
                         use_contrastive_loss=True, use_matching_loss=True, use_mlm_loss=config['mlm']['is_mlm'],
                         use_bbox_loss=False)
        # self.itm_weight = config.itm.weight
        # self.mim_weight = config.mim.weight
        # self.num_attention_heads = self.text_encoder.config.num_attention_heads
        self.init_params = []
        self.action_list = config.get('action_list', [])
        self.has_initialized_codebook = False

        self.be_hard = config.get('be_hard', False)
        self.be_pose_img = config.get('be_pose_img', False)
        self.be_pose_conv = config.get('pose_conv', False)
        if self.be_pose_img:
            self.pose_block = Block()
            self.init_params.extend(['pose_block.' + n for n, _ in self.pose_block.named_parameters()])
            if self.be_pose_conv:
                print('pose_conv')
                self.pose_conv = ConvExpandReduce()
                self.init_params.extend(['pose_conv.' + n for n, _ in self.pose_conv.named_parameters()])
        self.t = 0.1
        self.local_weight_text = nn.Parameter(torch.tensor([1.0 for _ in range(77)]))
        self.local_weight_image = nn.Parameter(torch.tensor([1.0 for _ in range(50)]))
        
        self.itm_weight = config.get('itm_weight', 4.0)
        self.cls_weight = config.get('cls_weight', 1.0)
        self.cls_target = config.get('cls_target', 'query')
        if self.cls_target not in {'codebook', 'query'}:
            raise ValueError("cls_target must be either 'codebook' or 'query'")
    
    def _check_and_init_codebook(self, device):
        if not self.has_initialized_codebook and hasattr(self, 'action_codebook'):
            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                tokenizer = self.tokenizer

            inputs = tokenizer(self.action_list, padding=True, return_tensors='pt').to(device)
            
            self.init_action_codebook(inputs.input_ids, inputs.attention_mask)
            
            self.has_initialized_codebook = True

    def get_image_feat(self, image_embeds):
        # Evaluation must use the same LightAIR representation optimized during training.
        return self.decouple_features(image_embeds)[0]


    def forward(self, image, text_ids, text_atts, text_ids_masked=None, masked_pos=None, masked_ids=None,
                idx=None, text_ids_eda=None, text_atts_eda=None,
                pose=None, hard_i=None, hard_i_pose=None, hard_text_ids=None, hard_text_atts=None,
                action_labels=None,
                ):

        image_embeds, image_atts = self.get_vision_embeds(image)
        text_embeds = self.get_text_embeds(text_ids, text_atts)
        self._check_and_init_codebook(image.device)

        if self.be_pose_img:
            if self.be_pose_conv:
                pose = self.pose_conv(pose)

            pose, _ = self.get_vision_embeds(pose)
            image_embeds = self.pose_block(image_embeds, pose)
            
        z_final, z_id, z_act_recon, v_proj, logits = self.decouple_features(image_embeds)
        text_feat = self.get_text_feat(text_embeds)
        
        loss_cls = torch.tensor(0., device=image.device)
        if action_labels is not None:
            valid_action_mask = (action_labels >= 0) & (action_labels < self.action_codebook.size(0))
            if valid_action_mask.any():
                if self.cls_target == 'query':
                    t_y = text_feat[valid_action_mask].detach()
                else:
                    t_y = self.action_codebook[action_labels[valid_action_mask]].detach()
                loss_cls = 1.0 - F.cosine_similarity(z_act_recon[valid_action_mask], t_y, dim=-1).mean()

        loss_itc = self.get_contrastive_loss(z_final, text_feat, idx=idx)
        loss_itm = self.get_matching_loss(image_embeds, image_atts, z_final,
                                          text_embeds, text_atts, text_feat, idx=idx)

        # eda
        text_embeds_eda = self.get_text_embeds(text_ids_eda, text_atts_eda)
        text_feat_eda = self.get_text_feat(text_embeds_eda)
        
        loss_itc_eda = self.get_contrastive_loss(z_final, text_feat_eda, idx=idx)
        loss_itm_eda = self.get_matching_loss(image_embeds, image_atts, z_final,
                                              text_embeds_eda, text_atts_eda, text_feat_eda, idx=idx, )
        loss_itc = loss_itc + 0.8 * loss_itc_eda
        loss_itm = loss_itm + 0.8 * loss_itm_eda

        loss_mlm = self.get_mlm_loss(text_ids_masked, text_atts, image_embeds, image_atts,
                                     masked_pos, masked_ids, )

        if self.be_hard:
            image_embeds_hard, image_atts_hard = self.get_vision_embeds(hard_i)
            z_final_hard, z_id_hard, _, _, _ = self.decouple_features(image_embeds_hard)
            text_embeds_hard = self.get_text_embeds(hard_text_ids, hard_text_atts)

            if self.be_pose_img:
                if self.be_pose_conv:
                    hard_i_pose = self.pose_conv(hard_i_pose)

                hard_pose, _ = self.get_vision_embeds(hard_i_pose)
                image_embeds_hard = self.pose_block(image_embeds_hard, hard_pose)

            loss_itm_hard = self.get_matching_loss_hard(image_embeds, image_atts, image_embeds_hard, image_atts_hard,
                                                        text_embeds, text_atts, text_embeds_hard, hard_text_atts)
            loss_itm = loss_itm + loss_itm_hard
        loss_mlm = loss_mlm

        return loss_itc + self.cls_weight * loss_cls, loss_itm * self.itm_weight, loss_mlm

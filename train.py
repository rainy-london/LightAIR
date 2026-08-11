import math

import torch

import utils
from thop import profile
import copy

import os
import csv
import json


def _get_amp_dtype(device):
    device = torch.device(device)
    if device.type != 'cuda':
        return torch.float32

    major, _ = torch.cuda.get_device_capability(device)
    native_bf16 = major >= 8 and torch.cuda.is_bf16_supported()
    return torch.bfloat16 if native_bf16 else torch.float16

def mlm(text, text_input, tokenizer, device, mask_generator, config):
    text_masked = tokenizer(text, padding='max_length', truncation=True, max_length=config['max_tokens'],
                            return_tensors="pt").to(device)
    text_ids_masked = text_masked.input_ids
    masked_pos = torch.empty((text_ids_masked.shape[0], config['max_masks']), dtype=torch.int64, device=device)
    masked_ids = torch.empty((text_ids_masked.shape[0], config['max_masks']), dtype=torch.long, device=device)
    for index, text_id in enumerate(text_ids_masked):
        text_ids_masked_, masked_pos_ = mask_generator(text_id)
        masked_ids_ = [text_input.input_ids[index][p].item() for p in masked_pos_]
        n_pad = config['max_masks'] - len(masked_ids_)
        masked_pos_ = masked_pos_ + [0] * n_pad
        masked_pos_ = torch.tensor(masked_pos_, dtype=torch.int64).to(device)
        masked_ids_ = masked_ids_ + [-100] * n_pad
        masked_ids_ = torch.tensor(masked_ids_, dtype=torch.long).to(device)
        masked_pos[index] = masked_pos_
        masked_ids[index] = masked_ids_
    return text_ids_masked, masked_pos, masked_ids


def train_model(model, data_loader, optimizer, scaler, tokenizer, epoch, device, scheduler, config, mask_generator, max_epoch=None):
    model.train()
    model_without_ddp = model.module if hasattr(model, 'module') else model
    device_type = torch.device(device).type
    amp_dtype = _get_amp_dtype(device)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_itc', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_mlm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50

    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
        if len(batch) == 9:
            image, text, text_eda, idx, pose, hard_i, hard_i_pose, hard_caption, action_labels = batch
        else:
            image, text, text_eda, idx, pose, hard_i, hard_i_pose, hard_caption = batch
            action_labels = None

        image = image.to(device, non_blocking=True)
        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=config['max_tokens'],
                               return_tensors="pt").to(device)
        text_ids_masked, masked_pos, masked_ids = mlm(text, text_input, tokenizer, device, mask_generator, config)

        text_input_eda = tokenizer(text_eda, padding='max_length', truncation=True, max_length=config['max_tokens'],
                                   return_tensors="pt").to(device)
        text_ids_eda = text_input_eda.input_ids
        text_atts_eda = text_input_eda.attention_mask

        idx = idx.to(device, non_blocking=True)
        if action_labels is not None:
            action_labels = action_labels.to(device, non_blocking=True)

        if config.get('be_hard', False):
            hard_i = hard_i.to(device, non_blocking=True)
            if config.get('be_pose_img', False):
                hard_i_pose = hard_i_pose.to(device, non_blocking=True)
            else:
                hard_i_pose = None

            hard_text_input = tokenizer(hard_caption, padding='max_length', truncation=True,
                                        max_length=config['max_tokens'], return_tensors="pt").to(device)
            hard_text_ids = hard_text_input.input_ids
            hard_text_atts = hard_text_input.attention_mask

        else:
            hard_i, hard_i_pose, hard_text_ids, hard_text_atts = None, None, None, None

        if config.get('be_pose_img', False):
            pose = pose.to(device, non_blocking=True)
        else:
            pose = None
            
        with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=device_type == 'cuda'):
            loss_itc, loss_itm, loss_mlm = \
                    model(image, text_input.input_ids, text_input.attention_mask,
                      text_ids_masked=text_ids_masked, masked_pos=masked_pos, masked_ids=masked_ids,
                      idx=idx, text_ids_eda=text_ids_eda, text_atts_eda=text_atts_eda,
                      pose=pose, hard_i=hard_i, hard_i_pose=hard_i_pose,
                      hard_text_ids=hard_text_ids, hard_text_atts=hard_text_atts,
                      action_labels=action_labels
                      )
            loss = loss_itc + loss_itm #+ loss_mlm
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scale = scaler.get_scale()
        scaler.update()
        skip_lr_sched = (scale > scaler.get_scale())
        if not skip_lr_sched:
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            scheduler.step()
        optimizer.zero_grad()

        metric_logger.update(loss_itc=loss_itc.item())
        metric_logger.update(loss_itm=loss_itm.item())
        metric_logger.update(loss_mlm=loss_mlm.item())
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.5f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}

def train_model_TPS(model, data_loader, optimizer, scaler, tokenizer, epoch, device, scheduler, config, mask_generator):
    model.train()
    device_type = torch.device(device).type
    amp_dtype = _get_amp_dtype(device)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_itc', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_mlm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50

    for i, (image, text, text_eda, idx) in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)):

        image = image.to(device, non_blocking=True)
        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=config['max_tokens'],
                               return_tensors="pt").to(device)
        text_ids_masked, masked_pos, masked_ids = mlm(text, text_input, tokenizer, device, mask_generator, config)

        text_input_eda = tokenizer(text_eda, padding='max_length', truncation=True, max_length=config['max_tokens'],
                                   return_tensors="pt").to(device)
        text_ids_eda = text_input_eda.input_ids
        text_atts_eda = text_input_eda.attention_mask

        idx = idx.to(device, non_blocking=True)

        hard_i, hard_i_pose, hard_text_ids, hard_text_atts = None, None, None, None

        pose = None

        with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=device_type == 'cuda'):
            loss_itc, loss_itm, loss_mlm = \
                model(image, text_input.input_ids, text_input.attention_mask,
                      text_ids_masked=text_ids_masked, masked_pos=masked_pos, masked_ids=masked_ids,
                      idx=idx, text_ids_eda=text_ids_eda, text_atts_eda=text_atts_eda,
                      pose=pose, hard_i=hard_i, hard_i_pose=hard_i_pose,
                      hard_text_ids=hard_text_ids, hard_text_atts=hard_text_atts,
                      )
            loss = loss_itc + loss_itm + loss_mlm #+ 0.1 * loss_kl

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scale = scaler.get_scale()
        scaler.update()
        skip_lr_sched = (scale > scaler.get_scale())
        if not skip_lr_sched:
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            scheduler.step()
        optimizer.zero_grad()

        metric_logger.update(loss_itc=loss_itc.item())
        metric_logger.update(loss_itm=loss_itm.item())
        metric_logger.update(loss_mlm=loss_mlm.item())
        # metric_logger.update(loss_kl=loss_kl.item())
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.5f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


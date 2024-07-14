#!/usr/bin/env python
import torch

torch.backends.cudnn.deterministic = True

import argparse
import numpy as np
import os
import shutil
import time
import queue
import logging
from torch.optim import Adam
import torch.nn.functional as F
import tensorflow as tf
import torch
import onnx
from onnx_tf.backend import prepare

from kd-mc.model import build_model
from kd-mc.batch import Batch
from kd-mc.helpers import (
    log_data_info,
    load_config,
    log_cfg,
    load_checkpoint,
    make_model_dir,
    make_logger,
    set_seed,
    symlink_update,
)
from kd-mc.model import SignModel
from kd-mc.data import load_data, make_data_iter
from kd-mc.builders import build_optimizer, build_scheduler, build_gradient_clipper
from kd-mc.metrics import bleu, chrf, rouge, wer_list
from torch.utils.tensorboard import SummaryWriter
from torchtext.data import Dataset
from typing import List, Dict
from kd-mc.vocabulary import (
    TextVocabulary,
    GlossVocabulary,
    PAD_TOKEN,
    EOS_TOKEN,
    BOS_TOKEN,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(ckpt_path, cfg, gls_vocab, txt_vocab):
    model_checkpoint = load_checkpoint(ckpt_path, torch.cuda.is_available())
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    model = build_model(
        cfg=cfg["model"],
        gls_vocab=gls_vocab,
        txt_vocab=txt_vocab,
        sgn_dim=sum(cfg["data"]["feature_size"])
        if isinstance(cfg["data"]["feature_size"], list)
        else cfg["data"]["feature_size"],
        do_recognition=True,
        do_translation=True,
    ).to(device)

    model.load_state_dict(model_checkpoint["model_state"])
    
    return model

def convertModel(cfg_file, ckpt):
    cfg = load_config(cfg_file)
    output_path=cfg["training"]["model_dir"]
    
    _, dev_data, _, gls_vocab, txt_vocab = load_data(data_cfg=cfg["data"])
    dev_iter = make_data_iter(dev_data, batch_size=cfg["training"]["batch_size"], batch_type="sentence", shuffle=False)
    
    model = load_model(ckpt, cfg, gls_vocab, txt_vocab)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    txt_pad_index = txt_vocab.stoi[PAD_TOKEN]

    batch = next(iter(dev_iter))
    batch = Batch(
        is_train=False,
        torch_batch=batch,
        txt_pad_index=txt_pad_index,
        sgn_dim=cfg["data"]["feature_size"],
        use_cuda=torch.cuda.is_available()
    )

    sgn, txt = batch.sgn, batch.txt
    txt_input = txt[:, :-1]
    sgn_lengths, txt_lengths = batch.sgn_lengths, batch.txt_lengths
    sgn_mask = (sgn != torch.zeros(batch.sgn_dim, device=device))[..., 0].unsqueeze(1)
    txt_mask = (txt_input != txt_pad_index).unsqueeze(1)
    
    # Export the best model to ONNX
    onnx_model_path = f"{output_path}/best_model.onnx"
    torch.onnx.export(
        model,
        (sgn, sgn_mask, sgn_lengths, txt_input, txt_mask),
        onnx_model_path,
        export_params=True,
        opset_version=11,
        input_names=['sgn', 'sgn_mask', 'sgn_lengths', 'txt_input', 'txt_mask'],
        output_names=['decoder_outputs', 'gloss_probabilities'],
        dynamic_axes={
            'sgn': {0: 'batch_size', 1: 'seq_length'},
            'sgn_mask': {0: 'batch_size', 1: 'seq_length'},
            'sgn_lengths': {0: 'batch_size'},
            'txt_input': {0: 'batch_size', 1: 'txt_seq_length'},
            'txt_mask': {0: 'batch_size', 1: 'txt_seq_length'},
            'decoder_outputs': {0: 'batch_size', 1: 'txt_seq_length'},
            'gloss_probabilities': {1: 'batch_size'}
        }
    )
    logger.info("Exported the model to ONNX format.")
    
    # Convert ONNX Model to TensorFlow
    onnx_model = onnx.load(onnx_model_path)
    tf_rep = prepare(onnx_model)
    tf_model_dir = f"{output_path}/best_model_tf"
    tf_rep.export_graph(tf_model_dir)
    logger.info("Converted the model to TensorFlow format.")

    # Convert TensorFlow Model to TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_dir)
    converter.target_spec.supported_ops = [
      tf.lite.OpsSet.TFLITE_BUILTINS,  # Enable TensorFlow Lite ops.
      tf.lite.OpsSet.SELECT_TF_OPS  # Enable TensorFlow ops.
    ]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    tflite_model_path = f"{output_path}/best_model.tflite"
    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)
    logger.info(f"Converted the model to TFLite format and saved in {output_path}.")
    
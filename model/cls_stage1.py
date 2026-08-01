
import torch
import torch.nn as nn
import random
import numpy as np

from .tf_blocks import CustomEncoderLayer, CustomQKVTransformerEncoder



class MPartyCLSSeqCrossPFlatTriModalFirstStageMean(nn.Module):
    def __init__(self, out_class_num, input_seq_len=6):
        super().__init__()
        self.num_heads = 6
        self.num_enc_layers = 1
        self.num_cross_enc_layers = 2
        self.modality_enc_out_dim = 300
        self.tf_dropout = 0.1
        self.seq_len = input_seq_len

        self.posi_modality         = nn.Parameter(torch.randn(1, self.seq_len, self.modality_enc_out_dim))
        self.posi_cross_p_plus_cls = nn.Parameter(torch.randn(1, 1 + 3 * 3, self.modality_enc_out_dim))
        self.cross_p_seq_cls = nn.Parameter(torch.randn(1, 1, self.modality_enc_out_dim))

        self.proj_marlin    = nn.Conv1d(768,  self.modality_enc_out_dim, kernel_size=1, padding=0)
        self.proj_trillsson = nn.Conv1d(1024, self.modality_enc_out_dim, kernel_size=1, padding=0)
        self.proj_roberta   = nn.Conv1d(768,  self.modality_enc_out_dim, kernel_size=1, padding=0)


        face_enc_layer  = CustomEncoderLayer(self.modality_enc_out_dim, nhead=self.num_heads, dim_feedforward=2048, dropout=self.tf_dropout)
        audio_enc_layer = CustomEncoderLayer(self.modality_enc_out_dim, nhead=self.num_heads, dim_feedforward=2048, dropout=self.tf_dropout)
        lang_enc_layer  = CustomEncoderLayer(self.modality_enc_out_dim, nhead=self.num_heads, dim_feedforward=2048, dropout=self.tf_dropout)
        self.face_enc  = CustomQKVTransformerEncoder(encoder_layer=face_enc_layer,  num_layers=self.num_enc_layers)
        self.audio_enc = CustomQKVTransformerEncoder(encoder_layer=audio_enc_layer, num_layers=self.num_enc_layers)
        self.lang_enc  = CustomQKVTransformerEncoder(encoder_layer=lang_enc_layer,  num_layers=self.num_enc_layers)

        enc_cross_layer = CustomEncoderLayer(self.modality_enc_out_dim, nhead=self.num_heads, dim_feedforward=2048, dropout=self.tf_dropout)
        self.encblock_cross = CustomQKVTransformerEncoder(encoder_layer=enc_cross_layer, num_layers=self.num_cross_enc_layers)

        self.cls_layer = nn.Linear(self.modality_enc_out_dim, out_class_num)

    def _project(self, proj_layer, features):
        x = proj_layer(features.transpose(1, 2)).transpose(1, 2)
        return x + self.posi_modality

    def forward(self, batch):
        x_t_marlin,   x_p1_marlin,   x_p2_marlin   = batch['inputs_face_p1'].float(),  batch['inputs_face_p2'].float(),  batch['inputs_face_p3'].float()
        x_t_trillsson, x_p1_trillsson, x_p2_trillsson = batch['inputs_audio_p1'].float(), batch['inputs_audio_p2'].float(), batch['inputs_audio_p3'].float()

        x_t_lang  = batch['inputs_lang_p1'].float()
        x_p1_lang = batch['inputs_lang_p2'].float()
        x_p2_lang = batch['inputs_lang_p3'].float()

        x_t_face    = self._project(self.proj_marlin,    x_t_marlin)
        x_p1_face   = self._project(self.proj_marlin,    x_p1_marlin)
        x_p2_face   = self._project(self.proj_marlin,    x_p2_marlin)
        x_t_audio   = self._project(self.proj_trillsson, x_t_trillsson)
        x_p1_audio  = self._project(self.proj_trillsson, x_p1_trillsson)
        x_p2_audio  = self._project(self.proj_trillsson, x_p2_trillsson)
        x_t_lang_proj  = self._project(self.proj_roberta, x_t_lang)
        x_p1_lang_proj = self._project(self.proj_roberta, x_p1_lang)
        x_p2_lang_proj = self._project(self.proj_roberta, x_p2_lang)

        enc_t_face   = self.face_enc(x_t_face,   x_t_face,   x_t_face)
        enc_p1_face  = self.face_enc(x_p1_face,  x_p1_face,  x_p1_face)
        enc_p2_face  = self.face_enc(x_p2_face,  x_p2_face,  x_p2_face)
        enc_t_audio  = self.audio_enc(x_t_audio,  x_t_audio,  x_t_audio)
        enc_p1_audio = self.audio_enc(x_p1_audio, x_p1_audio, x_p1_audio)
        enc_p2_audio = self.audio_enc(x_p2_audio, x_p2_audio, x_p2_audio)

        enc_t_lang  = self.lang_enc(x_t_lang_proj,  x_t_lang_proj,  x_t_lang_proj)
        enc_p1_lang = self.lang_enc(x_p1_lang_proj, x_p1_lang_proj, x_p1_lang_proj)
        enc_p2_lang = self.lang_enc(x_p2_lang_proj, x_p2_lang_proj, x_p2_lang_proj)

        cls_t_face   = enc_t_face.mean(dim=1).unsqueeze(1);   cls_t_audio  = enc_t_audio.mean(dim=1).unsqueeze(1)
        cls_p1_face  = enc_p1_face.mean(dim=1).unsqueeze(1);  cls_p1_audio = enc_p1_audio.mean(dim=1).unsqueeze(1)
        cls_p2_face  = enc_p2_face.mean(dim=1).unsqueeze(1);  cls_p2_audio = enc_p2_audio.mean(dim=1).unsqueeze(1)
        cls_t_lang  = enc_t_lang.mean(dim=1).unsqueeze(1)
        cls_p1_lang = enc_p1_lang.mean(dim=1).unsqueeze(1)
        cls_p2_lang = enc_p2_lang.mean(dim=1).unsqueeze(1)

        cross_p_cls = self.cross_p_seq_cls.expand(x_t_marlin.shape[0], -1, -1)
        cls_seq = torch.cat((cross_p_cls,
            cls_t_face,  cls_t_audio,  cls_t_lang,
            cls_p1_face, cls_p1_audio, cls_p1_lang,
            cls_p2_face, cls_p2_audio, cls_p2_lang,
        ), dim=1)
        cls_seq = cls_seq + self.posi_cross_p_plus_cls

        cross_out = self.encblock_cross(cls_seq, cls_seq, cls_seq)
        return self.cls_layer(cross_out[:, 0])
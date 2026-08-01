
import torch
import torch.nn as nn
import random
import numpy as np

from .tf_blocks import CustomEncoderLayer, CustomQKVTransformerEncoder



class CrossPMAllFlatLangPerTrack(nn.Module):
    def __init__(self, out_class_num, input_seq_len=6):
        super().__init__()
        self.num_heads = 6
        self.num_enc_layers = 2
        self.modality_enc_out_dim = 300
        self.tf_dropout = 0.1
        self.seq_len = input_seq_len

        flat_total_len = 9 * input_seq_len
        self.posi_flat = nn.Parameter(torch.randn(1, flat_total_len, self.modality_enc_out_dim))


        self.proj_marlin    = nn.Conv1d(768,  self.modality_enc_out_dim, kernel_size=1, padding=0)
        self.proj_trillsson = nn.Conv1d(1024, self.modality_enc_out_dim, kernel_size=1, padding=0)
        self.proj_roberta   = nn.Conv1d(768,  self.modality_enc_out_dim, kernel_size=1, padding=0)


        enc_layer = CustomEncoderLayer(self.modality_enc_out_dim, nhead=self.num_heads, dim_feedforward=2048, dropout=self.tf_dropout)
        self.flat_enc = CustomQKVTransformerEncoder(encoder_layer=enc_layer, num_layers=self.num_enc_layers)

        self.cls_layer = nn.Linear(self.modality_enc_out_dim, out_class_num)

    def _project(self, proj_layer, features):
        """Conv1d projection -> (batch, seq_len, dim)."""
        return proj_layer(features.transpose(1, 2)).transpose(1, 2)

    def forward(self, batch):
        """
        Args:
            batch:
            inputs_lang shape: torch.Size([1, seq_len, 768]): (batch, seq_len, feature_dim)
            inputs_audio_p1 shape: torch.Size([1, seq_len, 1024])
            inputs_face_p1 shape: torch.Size([1, seq_len, 768])
            inputs_audio_p2 shape: torch.Size([1, seq_len, 1024])
            inputs_face_p2 shape: torch.Size([1, seq_len, 768])
            inputs_audio_p3 shape: torch.Size([1, seq_len, 1024])
            inputs_face_p3 shape: torch.Size([1, seq_len, 768])

        Returns:
            outputs: (batch, out_class_num)
        """

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
        x_t_lang = self._project(self.proj_roberta,   x_t_lang)
        x_p1_lang = self._project(self.proj_roberta,   x_p1_lang)
        x_p2_lang = self._project(self.proj_roberta,   x_p2_lang)


        flat_seq = torch.cat((x_t_face, x_t_audio, x_t_lang, x_p1_face, x_p1_audio, x_p1_lang, x_p2_face, x_p2_audio, x_p2_lang), dim=1)
        flat_seq = flat_seq + self.posi_flat


        enc_out = self.flat_enc(flat_seq, flat_seq, flat_seq)

        pooled = enc_out.mean(dim=1)

        outputs = self.cls_layer(pooled)
        return outputs


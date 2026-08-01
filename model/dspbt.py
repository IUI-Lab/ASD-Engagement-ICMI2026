
import torch
import torch.nn as nn
from itertools import combinations

from .tf_blocks import CustomEncoderLayer
from .model_dspbt import AttentionPooling


class DSPBTTriModal(nn.Module):
    def __init__(self, out_class_num=2, input_seq_len=15, ffn_activation='relu'):
        super().__init__()
        self.d_model = 256
        self.num_heads = 4
        self.num_gtl_layers = 4          
        self.bottleneck_len = 4          
        self.tf_dropout = 0.1
        self.seq_len = input_seq_len     

        _act_map = {'relu': nn.ReLU(), 'gelu': nn.GELU()}
        if ffn_activation not in _act_map:
            raise ValueError(f"ffn_activation must be 'relu' or 'gelu', got '{ffn_activation}'")
        self._ffn_activation = _act_map[ffn_activation]


        self.stream_names = [
            "face_p1", "face_p2", "face_p3",
            "audio_p1", "audio_p2", "audio_p3",
            "lang_p1", "lang_p2", "lang_p3",
        ]
        self.num_streams = len(self.stream_names)

        self.pairs = list(combinations(range(self.num_streams), 2))


        self.proj_marlin = nn.Conv1d(768, self.d_model, kernel_size=1, padding=0)
        self.proj_trillsson = nn.Conv1d(1024, self.d_model, kernel_size=1, padding=0)
        self.proj_roberta = nn.Conv1d(768, self.d_model, kernel_size=1, padding=0)


        self.posi_embedding = nn.Parameter(torch.randn(1, self.seq_len, self.d_model))


        self.global_token_init = nn.Parameter(torch.randn(1, self.bottleneck_len, self.d_model))


        self.gtl_layers = nn.ModuleList([
            nn.ModuleList([
                CustomEncoderLayer(self.d_model, nhead=self.num_heads, dim_feedforward=2048, dropout=self.tf_dropout, activation=self._ffn_activation)
                for _ in range(self.num_streams)
            ])
            for _ in range(self.num_gtl_layers)
        ])


        self.attention_pooling = AttentionPooling(self.d_model)
        self.cls_layer = nn.Linear(self.d_model, out_class_num)

    def _project_and_position(self, proj_layer, features):

        x = proj_layer(features.transpose(1, 2)).transpose(1, 2)
        return x + self.posi_embedding

    def _pair_key(self, m, n):
        return (m, n) if m < n else (n, m)

    def forward(self, batch):
        x_face_p1  = batch['inputs_face_p1'].float()
        x_face_p2  = batch['inputs_face_p2'].float()
        x_face_p3  = batch['inputs_face_p3'].float()
        x_audio_p1 = batch['inputs_audio_p1'].float()
        x_audio_p2 = batch['inputs_audio_p2'].float()
        x_audio_p3 = batch['inputs_audio_p3'].float()
        x_lang_p1  = batch['inputs_lang_p1'].float()
        x_lang_p2  = batch['inputs_lang_p2'].float()
        x_lang_p3  = batch['inputs_lang_p3'].float()

        batch_size = x_face_p1.shape[0]


        H = [
            self._project_and_position(self.proj_marlin,    x_face_p1),   
            self._project_and_position(self.proj_marlin,    x_face_p2),  
            self._project_and_position(self.proj_marlin,    x_face_p3),   
            self._project_and_position(self.proj_trillsson, x_audio_p1),  
            self._project_and_position(self.proj_trillsson, x_audio_p2),  
            self._project_and_position(self.proj_trillsson, x_audio_p3),  
            self._project_and_position(self.proj_roberta,   x_lang_p1),   
            self._project_and_position(self.proj_roberta,   x_lang_p2),   
            self._project_and_position(self.proj_roberta,   x_lang_p3),  
        ]


        pair_tokens = {
            pair: self.global_token_init.expand(batch_size, -1, -1)
            for pair in self.pairs
        }

        T = self.seq_len
        B_tok = self.bottleneck_len

        for layer_idx in range(self.num_gtl_layers):
            new_H = [None] * self.num_streams
            dir_tokens = {}

            for m in range(self.num_streams):
                partners = [n for n in range(self.num_streams) if n != m]
                g_cat = torch.cat([pair_tokens[self._pair_key(m, n)] for n in partners], dim=1)
                seq_in = torch.cat([H[m], g_cat], dim=1)

                seq_out = self.gtl_layers[layer_idx][m](seq_in, seq_in, seq_in)

                new_H[m] = seq_out[:, :T, :]
                g_out = seq_out[:, T:, :]
                for i, n in enumerate(partners):
                    dir_tokens[(m, n)] = g_out[:, i * B_tok:(i + 1) * B_tok, :]

            pair_tokens = {
                pair: dir_tokens[(pair[0], pair[1])] + dir_tokens[(pair[1], pair[0])]
                for pair in self.pairs
            }
            H = new_H


        target_seq = torch.cat([H[0], H[3], H[6]], dim=1)
        pooled = self.attention_pooling(target_seq)
        return self.cls_layer(pooled)

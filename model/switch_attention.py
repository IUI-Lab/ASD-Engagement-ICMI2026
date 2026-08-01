import torch
import torch.nn as nn

from .layers import TransformerEncoder


class MultipartyTransformer(nn.Module):
    def __init__(self, out_class_num=2, input_seq_len=15, behavior_dims=192,
                 switch_attn_mode: str = "split"):
        super().__init__()
        assert switch_attn_mode in ("split", "mask", "archive"), \
            f"Please specify 'split', 'mask', or 'archive' for switch_attn_mode: {switch_attn_mode}"
        self.switch_attn_mode = switch_attn_mode

        self.behavior_dims = behavior_dims
        proj_dim = behavior_dims // 3 

        self.num_heads = 4
        self.layers    = 4
        self.attn_dropout   = 0.25
        self.relu_dropout   = 0.25
        self.res_dropout    = 0.25
        self.embed_dropout  = 0.25
        self.attn_mask      = True

        self.proj_marlin    = nn.Conv1d(768,  proj_dim, kernel_size=1, padding=0)
        self.proj_trillsson = nn.Conv1d(1024, proj_dim, kernel_size=1, padding=0)
        self.proj_roberta   = nn.Conv1d(768,  proj_dim, kernel_size=1, padding=0)

        self.trans             = self._get_enc(behavior_dims)
        self.trans_t1_speaker  = self._get_enc(behavior_dims)
        self.trans_t1_listener = self._get_enc(behavior_dims)

        self.trans_t1_mem = nn.LSTM(3 * behavior_dims, 3 * behavior_dims, 1, batch_first=True)

        self.cls_layer = nn.Linear(3 * behavior_dims, out_class_num)

    def _get_enc(self, embed_dim):
        return TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=self.num_heads,
            layers=self.layers,
            attn_dropout=self.attn_dropout,
            relu_dropout=self.relu_dropout,
            res_dropout=self.res_dropout,
            embed_dropout=self.embed_dropout,
            attn_mask=self.attn_mask,
        )

    def _proj(self, proj_layer, x):
        return proj_layer(x.transpose(1, 2))

    def forward(self, batch):
        x_t1_marlin    = batch['inputs_face_p1'].float()
        x_t1_trillsson = batch['inputs_audio_p1'].float()
        x_p1_marlin    = batch['inputs_face_p2'].float()
        x_p1_trillsson = batch['inputs_audio_p2'].float()
        x_p2_marlin    = batch['inputs_face_p3'].float()
        x_p2_trillsson = batch['inputs_audio_p3'].float()

        x_t1_lang = batch['inputs_lang_p1'].float()
        x_p1_lang = batch['inputs_lang_p2'].float()
        x_p2_lang = batch['inputs_lang_p3'].float()

        p_t1_m  = self._proj(self.proj_marlin,    x_t1_marlin)
        p_t1_a  = self._proj(self.proj_trillsson, x_t1_trillsson)
        p_t1_l  = self._proj(self.proj_roberta,   x_t1_lang)
        p_p1_m  = self._proj(self.proj_marlin,    x_p1_marlin)
        p_p1_a  = self._proj(self.proj_trillsson, x_p1_trillsson)
        p_p1_l  = self._proj(self.proj_roberta,   x_p1_lang)
        p_p2_m  = self._proj(self.proj_marlin,    x_p2_marlin)
        p_p2_a  = self._proj(self.proj_trillsson, x_p2_trillsson)
        p_p2_l  = self._proj(self.proj_roberta,   x_p2_lang)

        proj_t1 = torch.cat([p_t1_m, p_t1_a, p_t1_l], dim=1).permute(2, 0, 1)
        proj_p1 = torch.cat([p_p1_m, p_p1_a, p_p1_l], dim=1).permute(2, 0, 1)
        proj_p2 = torch.cat([p_p2_m, p_p2_a, p_p2_l], dim=1).permute(2, 0, 1)

        h_self = self.trans(proj_t1, proj_t1, proj_t1)

        vad = batch['speaker']

        if self.switch_attn_mode == "split":
            t1_listener_idx = [i for i, j in enumerate(vad) if j != 'P0']
            t1_speaker_idx  = [i for i, j in enumerate(vad) if j == 'P0']

            if len(t1_listener_idx) > 1:
                if len(t1_speaker_idx) > 1:
                    proj_t1_listener = proj_t1[:, t1_listener_idx, :]
                    proj_t1_speaker  = proj_t1[:, t1_speaker_idx, :]
                    proj_p1_listener = proj_p1[:, t1_listener_idx, :]
                    proj_p1_speaker  = proj_p1[:, t1_speaker_idx, :]
                    proj_p2_listener = proj_p2[:, t1_listener_idx, :]
                    proj_p2_speaker  = proj_p2[:, t1_speaker_idx, :]

                    h_t1_with_p1_speaker  = self.trans_t1_speaker(proj_p1_speaker, proj_t1_speaker, proj_t1_speaker)
                    h_t1_with_p1_listener = self.trans_t1_listener(proj_t1_listener, proj_p1_listener, proj_p1_listener)
                    h_t1_with_p2_speaker  = self.trans_t1_speaker(proj_p2_speaker, proj_t1_speaker, proj_t1_speaker)
                    h_t1_with_p2_listener = self.trans_t1_listener(proj_t1_listener, proj_p2_listener, proj_p2_listener)

                    speaker_idx_t  = torch.tensor(t1_speaker_idx,  device=proj_t1.device)
                    listener_idx_t = torch.tensor(t1_listener_idx, device=proj_t1.device)

                    h_p1 = torch.empty(proj_t1.shape[0], proj_t1.shape[1], h_t1_with_p1_speaker.shape[2],
                                       dtype=h_t1_with_p1_speaker.dtype, device=h_t1_with_p1_speaker.device)
                    h_p1[:, speaker_idx_t, :]  = h_t1_with_p1_speaker
                    h_p1[:, listener_idx_t, :] = h_t1_with_p1_listener

                    h_p2 = torch.empty_like(h_p1)
                    h_p2[:, speaker_idx_t, :]  = h_t1_with_p2_speaker
                    h_p2[:, listener_idx_t, :] = h_t1_with_p2_listener
                else:
                    h_p1 = self.trans_t1_listener(proj_t1, proj_p1, proj_p1)
                    h_p2 = self.trans_t1_listener(proj_t1, proj_p2, proj_p2)
            else:
                h_p1 = self.trans_t1_speaker(proj_p1, proj_t1, proj_t1)
                h_p2 = self.trans_t1_speaker(proj_p2, proj_t1, proj_t1)

        elif self.switch_attn_mode == "archive":
            t1_listener_idx = [i for i, j in enumerate(vad) if j != 'P0']
            t1_speaker_idx  = [i for i, j in enumerate(vad) if j == 'P0']

            if len(t1_listener_idx) > 1:
                if len(t1_speaker_idx) > 1:
                    proj_t1_listener = proj_t1[:, t1_listener_idx, :]
                    proj_t1_speaker  = proj_t1[:, t1_speaker_idx, :]
                    proj_p1_listener = proj_p1[:, t1_listener_idx, :]
                    proj_p1_speaker  = proj_p1[:, t1_speaker_idx, :]
                    proj_p2_listener = proj_p2[:, t1_listener_idx, :]
                    proj_p2_speaker  = proj_p2[:, t1_speaker_idx, :]

                    h_t1_with_p1_speaker  = self.trans_t1_speaker(proj_p1_speaker, proj_t1_speaker, proj_t1_speaker)
                    h_t1_with_p1_listener = self.trans_t1_listener(proj_t1_listener, proj_p1_listener, proj_p1_listener)
                    h_t1_with_p2_speaker  = self.trans_t1_speaker(proj_p2_speaker, proj_t1_speaker, proj_t1_speaker)
                    h_t1_with_p2_listener = self.trans_t1_listener(proj_t1_listener, proj_p2_listener, proj_p2_listener)

                    h_p1 = torch.cat([h_t1_with_p1_speaker, h_t1_with_p1_listener], dim=1)
                    h_p2 = torch.cat([h_t1_with_p2_speaker, h_t1_with_p2_listener], dim=1)
                else:
                    h_p1 = self.trans_t1_listener(proj_t1, proj_p1, proj_p1)
                    h_p2 = self.trans_t1_listener(proj_t1, proj_p2, proj_p2)
            else:
                h_p1 = self.trans_t1_speaker(proj_p1, proj_t1, proj_t1)
                h_p2 = self.trans_t1_speaker(proj_p2, proj_t1, proj_t1)

        else:
            is_speaker = torch.tensor([j == 'P0' for j in vad],
                                       dtype=torch.bool, device=proj_t1.device)
            n_spk = int(is_speaker.sum())
            n_lis = int((~is_speaker).sum())

            h_p1_spk = self.trans_t1_speaker(proj_p1, proj_t1, proj_t1)
            h_p1_lis = self.trans_t1_listener(proj_t1, proj_p1, proj_p1)
            h_p2_spk = self.trans_t1_speaker(proj_p2, proj_t1, proj_t1)
            h_p2_lis = self.trans_t1_listener(proj_t1, proj_p2, proj_p2)

            mask = is_speaker.view(1, -1, 1)
            if n_lis > 1 and n_spk > 1:
                h_p1 = torch.where(mask, h_p1_spk, h_p1_lis)
                h_p2 = torch.where(mask, h_p2_spk, h_p2_lis)
            elif n_lis > 1:
                h_p1 = h_p1_lis
                h_p2 = h_p2_lis
            else:
                h_p1 = h_p1_spk
                h_p2 = h_p2_spk

        h_cat = torch.cat([h_self, h_p1, h_p2], dim=2)

        h_cat = h_cat.permute(1, 0, 2)
        h_lstm, _ = self.trans_t1_mem(h_cat)
        last_h = h_lstm[:, -1]

        return self.cls_layer(last_h)
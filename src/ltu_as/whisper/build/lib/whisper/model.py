from dataclasses import dataclass
from typing import Dict
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn

from .transcribe import transcribe as transcribe_function
from .transcribe import transcribe_audio as transcribe_audio_function
from .decoding import detect_language as detect_language_function, decode as decode_function


@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int


class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x, self.weight.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype)
        )


class Conv1d(nn.Conv1d):
    def _conv_forward(self, x: Tensor, weight: Tensor, bias: Optional[Tensor]) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None):
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]
        qk = qk.float()

        w = F.softmax(qk, dim=-1).to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = MultiHeadAttention(n_state, n_head) if cross_attention else None
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state))
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x


class AudioEncoder(nn.Module):
    def __init__(self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int):
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.n_layer = n_layer
        if n_layer > 0:
            self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
                [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
            )

        self.ln_post = LayerNorm(n_state)

    def forward(self, x: Tensor, o_layer='last'):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)

        if self.n_layer == 0:
            return x

        x_len = x.shape[1]
        x = (x + self.positional_embedding[:x_len, :]).to(x.dtype)

        # this certainly is correct, but yields double ln (mlp also has ln)
        # remove ln
        if o_layer == 'last':
            for block in self.blocks:
                x = block(x)
            #x = self.ln_post(x)
            return x

        elif o_layer == 'all':
            all_x = []
            for block in self.blocks:
                # append the output of cnn
                all_x.append(x.clone().detach())
                x = block(x)
            #x = self.ln_post(x) # skip the layer norm, make it fair for all layers
            all_x.append(x.clone().detach())
            return all_x

        # save the mean pooled version of x
        elif o_layer == 'all_pool':
            all_x = []
            for block in self.blocks:
                # append the output of cnn
                x_mean = torch.mean(x, dim=1)
                all_x.append(x_mean)
                x = block(x) # x in shape [B, audio_len, feat_dim], after pooling in shape [B, feat_dim]
            x_mean = torch.mean(x, dim=1)
            all_x.append(x_mean)
            all_x = torch.stack(all_x, dim=2) # should be in shape (B, feat_dim, num_layer)
            return all_x

        elif o_layer == 'all_nopool':
            all_x = []
            for block in self.blocks:
                all_x.append(x)
                x = block(x) # x in shape [B, audio_len, feat_dim], after pooling in shape [B, feat_dim]
            all_x.append(x)
            all_x = torch.stack(all_x, dim=3) # should be in shape (B, audio_len, feat_dim, num_layer)
            return all_x


class AudioEncoder_Ori(nn.Module):
    def __init__(self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int):
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)

    def forward(self, x: Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)

        assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        x = (x + self.positional_embedding).to(x.dtype)

        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)
        return x


class TextDecoder(nn.Module):
    def __init__(self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head, cross_attention=True) for _ in range(n_layer)]
        )
        self.ln = LayerNorm(n_state)

        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: Tensor, xa: Tensor, kv_cache: Optional[dict] = None):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_mels, n_audio_ctx)
            the encoded audio features to be attended on
        """
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = self.token_embedding(x) + self.positional_embedding[offset : offset + x.shape[-1]]
        x = x.to(xa.dtype)

        for block in self.blocks:
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache)

        x = self.ln(x)
        logits = (x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)).float()

        return logits

class Whisper(nn.Module):
    def __init__(self, dims: ModelDimensions, label_dim=1, cla='mlp_1'):
        super().__init__()
        self.dims = dims
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
        )

        self.cla = cla
        if 'mlp' in self.cla:
            self.mlp_layer = int(self.cla.split('_')[-1])
            if self.mlp_layer == 1:
                self.mlp_head = nn.Sequential(nn.LayerNorm(self.dims.n_audio_state), nn.Linear(self.dims.n_audio_state, label_dim))
                print('now use mlp head classifier with {:d} layers'.format(self.mlp_layer))
            elif self.mlp_layer == 2:
                self.mlp_head = nn.Sequential(nn.LayerNorm(self.dims.n_audio_state),
                                              nn.Linear(self.dims.n_audio_state, 768),
                                              nn.ReLU(),
                                              nn.LayerNorm(768),
                                              nn.Linear(768, label_dim)
                                              )
                print('now use mlp head classifier with {:d} layers'.format(self.mlp_layer))
            elif self.mlp_layer == 3:
                self.mlp_head = nn.Sequential(nn.LayerNorm(self.dims.n_audio_state),
                                              nn.Linear(self.dims.n_audio_state, 768),
                                              nn.ReLU(),
                                              nn.LayerNorm(768),
                                              nn.Linear(768, 768),
                                              nn.ReLU(),
                                              nn.LayerNorm(768),
                                              nn.Linear(768, label_dim)
                                              )
                print('now use mlp head classifier with {:d} layers'.format(self.mlp_layer))

        # different in order of mlp head
        elif 'trans' in self.cla:
            self.num_trans = int(self.cla.split('_')[-2])
            self.num_heads = int(self.cla.split('_')[-1])
            self.a_trans = nn.ModuleList(
                [ResidualAttentionBlock(self.dims.n_audio_state, self.num_heads) for _ in range(self.num_trans)]
            )
            self.mlp_head = nn.Sequential(nn.LayerNorm(self.dims.n_audio_state), nn.Linear(self.dims.n_audio_state, label_dim))

        elif 'tranx' in self.cla:
            self.num_trans = int(self.cla.split('_')[-1])
            self.mlp_head = nn.Sequential(nn.LayerNorm(self.dims.n_audio_state), nn.Linear(self.dims.n_audio_state, label_dim))
            if label_dim == 527:
                self.a_trans = nn.ModuleList(
                    [ResidualAttentionBlock(label_dim, 17) for _ in range(self.num_trans)]
                )
            elif label_dim == 50:
                self.a_trans = nn.ModuleList(
                    [ResidualAttentionBlock(label_dim, 5) for _ in range(self.num_trans)]
                )

        elif 'trandx' in self.cla:
            self.num_trans = int(self.cla.split('_')[-1])
            self.mlp_head1 = nn.Sequential(nn.LayerNorm(self.dims.n_audio_state), nn.Linear(self.dims.n_audio_state, 768))
            self.a_trans = nn.ModuleList(
                [ResidualAttentionBlock(768, 12) for _ in range(self.num_trans)]
            )
            self.mlp_head2 = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, label_dim))

        elif 'trancx' in self.cla:
            def _no_grad_trunc_normal_(tensor, mean, std, a, b):
                def norm_cdf(x):
                    # Computes standard normal cumulative distribution function
                    return (1. + math.erf(x / math.sqrt(2.))) / 2.

                if (mean < a - 2 * std) or (mean > b + 2 * std):
                    warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                                  "The distribution of values may be incorrect.",
                                  stacklevel=2)

            def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
                return _no_grad_trunc_normal_(tensor, mean, std, a, b)

            self.a_trans_cls = nn.Parameter(torch.zeros(1, 1, 768))
            trunc_normal_(self.a_trans_cls, std=.02)

            self.num_trans = int(self.cla.split('_')[-1])
            self.mlp_head1 = nn.Sequential(nn.LayerNorm(self.dims.n_audio_state), nn.Linear(self.dims.n_audio_state, 768))
            self.a_trans = nn.ModuleList(
                [ResidualAttentionBlock(768, 12) for _ in range(self.num_trans)]
            )
            self.mlp_head2 = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, label_dim))

        elif 'layertr' in self.cla:
            self.num_att_head = int(self.cla.split('_')[-1])
            self.inter_rep_dim = int(self.cla.split('_')[-2])

            self.mlp_layers = nn.ModuleList([nn.Sequential(nn.LayerNorm(self.dims.n_audio_state), nn.Linear(self.dims.n_audio_state, self.inter_rep_dim)) for _ in range(self.dims.n_audio_layer)])
            self.tr_layers = nn.ModuleList([ResidualAttentionBlock(self.inter_rep_dim, self.num_att_head) for _ in range(self.dims.n_audio_layer)])
            self.layer_tr = ResidualAttentionBlock(self.inter_rep_dim, self.num_att_head)
            self.layer_weight = torch.nn.Parameter(torch.tensor([1 / self.n_layer] * self.n_layer))
            self.cla_layer = nn.Sequential(nn.LayerNorm(self.inter_rep_dim), nn.Linear(self.inter_rep_dim, self.label_dim))

        # for weight average
        self.layer_weight = torch.nn.Parameter(torch.tensor([1/(self.dims.n_audio_layer + 1)] * (self.dims.n_audio_layer + 1)))
        print('Now using modified audio whisper model')
        print(self.layer_weight)

    def embed_audio(self, mel: torch.Tensor, o_layer='last'):
        return self.encoder(mel, o_layer)

    def forward(self, mel, mode='last'):
        if mode == 'last':
            audio_rep = self.encoder(mel)
            if 'mlp' in self.cla:
                audio_rep = torch.mean(audio_rep, dim=1)
                label = self.mlp_head(audio_rep)
                return label
            elif 'trans' in self.cla:
                for block in self.a_trans:
                    audio_rep = block(audio_rep)
                audio_rep = torch.mean(audio_rep, dim=1)
                label = self.mlp_head(audio_rep)
                return label
            elif 'tranx' in self.cla:
                audio_rep = self.mlp_head(audio_rep)
                for block in self.a_trans:
                    audio_rep = block(audio_rep)
                label = torch.mean(audio_rep, dim=1)
                return label
            elif 'trandx' in self.cla:
                audio_rep = self.mlp_head1(audio_rep)
                for block in self.a_trans:
                    audio_rep = block(audio_rep)
                audio_rep = torch.mean(audio_rep, dim=1)
                label = self.mlp_head2(audio_rep)
                return label
            elif 'trancx' in self.cla:
                B = audio_rep.shape[0]
                audio_rep = self.mlp_head1(audio_rep)
                cls_tokens = self.a_trans_cls.expand(B, -1, -1)
                audio_rep = torch.cat((cls_tokens, audio_rep), dim=1)
                for block in self.a_trans:
                    audio_rep = block(audio_rep)
                audio_rep = audio_rep[:, 0]
                label = self.mlp_head2(audio_rep)
                return label

        # average all layers
        elif mode == 'avg_layer':
            if 'mlp' in self.cla:
                # audio_rep in shape(B, feat_dim, num_layer)
                audio_rep = self.encoder(mel, o_layer='all_pool')
                # now audio_rep in shape (B, feat_dim)
                audio_rep = (audio_rep@self.layer_weight)/self.layer_weight.sum()
                #print(self.layer_weight), no mean necessary
                label = self.mlp_head(audio_rep)
                return label
            elif 'trans' in self.cla:
                # audio_rep in shape(B, feat_dim, num_layer)
                audio_rep = self.encoder(mel, o_layer='all_nopool')
                # now audio_rep in shape (B, feat_dim)
                audio_rep = (audio_rep@self.layer_weight)/self.layer_weight.sum()
                #print(self.layer_weight)
                for block in self.a_trans:
                    audio_rep = block(audio_rep)
                audio_rep = torch.mean(audio_rep, dim=1)
                label = self.mlp_head(audio_rep)
                return label
            elif 'tranx' in self.cla:
                # audio_rep in shape(B, feat_dim, num_layer)
                audio_rep = self.encoder(mel, o_layer='all_nopool')
                # now audio_rep in shape (B, feat_dim)
                audio_rep = (audio_rep@self.layer_weight)/self.layer_weight.sum()
                audio_rep = self.mlp_head(audio_rep)
                # print(self.layer_weight)
                for block in self.a_trans:
                    audio_rep = block(audio_rep)
                label = torch.mean(audio_rep, dim=1)
                return label
            elif 'trandx' in self.cla:
                # audio_rep in shape(B, feat_dim, num_layer)
                audio_rep = self.encoder(mel, o_layer='all_nopool')
                # now audio_rep in shape (B, feat_dim)
                audio_rep = (audio_rep@self.layer_weight)/self.layer_weight.sum()
                audio_rep = self.mlp_head1(audio_rep)
                # print(self.layer_weight)
                for block in self.a_trans:
                    audio_rep = block(audio_rep)
                audio_rep = torch.mean(audio_rep, dim=1)
                label = self.mlp_head2(audio_rep)
                return label
            elif 'trancx' in self.cla:
                # audio_rep in shape(B, feat_dim, num_layer)
                audio_rep = self.encoder(mel, o_layer='all_nopool')
                # now audio_rep in shape (B, feat_dim)
                audio_rep = (audio_rep@self.layer_weight)/self.layer_weight.sum()
                audio_rep = self.mlp_head1(audio_rep)
                B = audio_rep.shape[0]
                cls_tokens = self.a_trans_cls.expand(B, -1, -1)
                audio_rep = torch.cat((cls_tokens, audio_rep), dim=1)
                # print(self.layer_weight)
                for block in self.a_trans:
                    audio_rep = block(audio_rep)
                audio_rep = audio_rep[:, 0]
                label = self.mlp_head2(audio_rep)
                return label


    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_multilingual(self):
        return self.dims.n_vocab == 51865

    transcribe_audio = transcribe_audio_function

# # original whisper model
# class Whisper_Ori(nn.Module):
#     def __init__(self, dims: ModelDimensions):
#         super().__init__()
#         self.dims = dims
#         self.encoder = AudioEncoder(
#             self.dims.n_mels,
#             self.dims.n_audio_ctx,
#             self.dims.n_audio_state,
#             self.dims.n_audio_head,
#             self.dims.n_audio_layer,
#         )
#         self.decoder = TextDecoder(
#             self.dims.n_vocab,
#             self.dims.n_text_ctx,
#             self.dims.n_text_state,
#             self.dims.n_text_head,
#             self.dims.n_text_layer,
#         )
#
#     def embed_audio(self, mel: torch.Tensor):
#         return self.encoder(mel)
#
#     def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor):
#         return self.decoder(tokens, audio_features)
#
#     def forward(self, mel: torch.Tensor, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
#         return self.decoder(tokens, self.encoder(mel))
#
#     @property
#     def device(self):
#         return next(self.parameters()).device
#
#     @property
#     def is_multilingual(self):
#         return self.dims.n_vocab == 51865
#
#     def install_kv_cache_hooks(self, cache: Optional[dict] = None):
#         """
#         The `MultiHeadAttention` module optionally accepts `kv_cache` which stores the key and value
#         tensors calculated for the previous positions. This method returns a dictionary that stores
#         all caches, and the necessary hooks for the key and value projection modules that save the
#         intermediate tensors to be reused during later calculations.
#
#         Returns
#         -------
#         cache : Dict[nn.Module, torch.Tensor]
#             A dictionary object mapping the key/value projection modules to its cache
#         hooks : List[RemovableHandle]
#             List of PyTorch RemovableHandle objects to stop the hooks to be called
#         """
#         cache = {**cache} if cache is not None else {}
#         hooks = []
#
#         def save_to_cache(module, _, output):
#             if module not in cache or output.shape[1] > self.decoder.positional_embedding.shape[0]:
#                 cache[module] = output  # save as-is, for the first token or cross attention
#             else:
#                 cache[module] = torch.cat([cache[module], output], dim=1).detach()
#             return cache[module]
#
#         def install_hooks(layer: nn.Module):
#             if isinstance(layer, MultiHeadAttention):
#                 hooks.append(layer.key.register_forward_hook(save_to_cache))
#                 hooks.append(layer.value.register_forward_hook(save_to_cache))
#
#         self.decoder.apply(install_hooks)
#         return cache, hooks
#
#     detect_language = detect_language_function
#     transcribe = transcribe_function
#     transcribe_audio = transcribe_audio_function
#     decode = decode_function

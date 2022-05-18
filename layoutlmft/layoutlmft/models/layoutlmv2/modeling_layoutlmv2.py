# coding=utf-8
import math

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

import detectron2
from detectron2.modeling import META_ARCH_REGISTRY
from transformers import PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    TokenClassifierOutput,
)
from transformers.modeling_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.models.layoutlm.modeling_layoutlm import LayoutLMIntermediate as LayoutLMv2Intermediate
from transformers.models.layoutlm.modeling_layoutlm import LayoutLMOutput as LayoutLMv2Output
from transformers.models.layoutlm.modeling_layoutlm import LayoutLMPooler as LayoutLMv2Pooler
from transformers.models.layoutlm.modeling_layoutlm import LayoutLMSelfOutput as LayoutLMv2SelfOutput
from transformers.utils import logging

from ...modules.decoders.re import REDecoder
from ...utils import ReOutput
from .configuration_layoutlmv2 import LayoutLMv2Config
from .detectron2_config import add_layoutlmv2_config


logger = logging.get_logger(__name__)

LAYOUTLMV2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "layoutlmv2-base-uncased",
    "layoutlmv2-large-uncased",
]


LayoutLMv2LayerNorm = torch.nn.LayerNorm


class LayoutLMv2Embeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super(LayoutLMv2Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        self.x_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.coordinate_size)
        self.y_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.coordinate_size)  # 1024(we used 1000 at the moment) * 128
        self.h_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.shape_size)  # 1024(we used 1000 at the moment) * 128
        self.w_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.shape_size)  # 1024(we used 1000 at the moment) * 128
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)  # 2(two token types) * 768

        self.LayerNorm = LayoutLMv2LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def _cal_spatial_position_embeddings(self, bbox):
        try:
            left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])  # get embedding for top left x. (batch_size, 512) -> (batch_size, 512, 128). each value is represented by a 128 vector.
            upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
            right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
            lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])
        except IndexError as e:
            raise IndexError("The :obj:`bbox`coordinate values should be within 0-1000 range.") from e
        # x_position_embeddings(1024, 128), y_position_embeddings(1024, 128), h_position_embeddings(1024, 128), w_position_embeddings(1024, 128)
        h_position_embeddings = self.h_position_embeddings(bbox[:, :, 3] - bbox[:, :, 1])  # height embedding (batch_size, 512) -> (batch_size, 512, 128)
        w_position_embeddings = self.w_position_embeddings(bbox[:, :, 2] - bbox[:, :, 0])  # width embedding
        # concat to spatial_position_embeddings(batch_size, 512, 768(128 * 6))
        spatial_position_embeddings = torch.cat(
            [
                left_position_embeddings,
                upper_position_embeddings,
                right_position_embeddings,
                lower_position_embeddings,
                h_position_embeddings,
                w_position_embeddings,
            ],
            dim=-1,
        )
        return spatial_position_embeddings


class LayoutLMv2SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.fast_qkv = config.fast_qkv
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.has_relative_attention_bias = config.has_relative_attention_bias
        self.has_spatial_attention_bias = config.has_spatial_attention_bias

        if config.fast_qkv:
            self.qkv_linear = nn.Linear(config.hidden_size, 3 * self.all_head_size, bias=False)
            self.q_bias = nn.Parameter(torch.zeros(1, 1, self.all_head_size))
            self.v_bias = nn.Parameter(torch.zeros(1, 1, self.all_head_size))
        else:
            self.query = nn.Linear(config.hidden_size, self.all_head_size)
            self.key = nn.Linear(config.hidden_size, self.all_head_size)
            self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):  # x(batch, 561, 768), 561 consists of 512 text tokens and 49 visually cut bboxes. 768 is combined embeddings.
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)  # new_x_shape(batch, 561, num_attention_heads=12, attention_head_size=64)
        x = x.view(*new_x_shape)  # x(8, 561, num_attention_heads=12, attention_head_size=64)
        return x.permute(0, 2, 1, 3)  # x(8, num_attention_heads=12, 561, attention_head_size=64)

    def compute_qkv(self, hidden_states):
        # hidden_states - final_emb(batch, 561=49+512, 768) combines text_layout_emb and visual_emb.
        if self.fast_qkv:
            # qkv(batch, 561, 768*3) = hidden_states(batch, 561, 768) * W^T(768, 768*3).
            # We project text_layout_visual_embedding into the row spaces of query, key and value.
            qkv = self.qkv_linear(hidden_states)
            # We cut qkv(batch, 561, 768*3) along last dim into 3 same-sized(batch, 561, 768) projected matrices: query, key, value.
            q, k, v = torch.chunk(qkv, 3, dim=-1)
            if q.ndimension() == self.q_bias.ndimension():
                q = q + self.q_bias
                v = v + self.v_bias
            else:
                _sz = (1,) * (q.ndimension() - 1) + (-1,)
                q = q + self.q_bias.view(*_sz)
                v = v + self.v_bias.view(*_sz)
        else:
            q = self.query(hidden_states)
            k = self.key(hidden_states)
            v = self.value(hidden_states)
        # q - query(batch, 561, 768), k - key(batch, 561, 768), v - value(batch, 561, 768)
        return q, k, v

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        rel_pos=None,
        rel_2d_pos=None,
    ):
        # hidden_states - final_emb(batch, 561=512+49, 768) combines text_layout_emb and visual_emb. 561 consists of 512 text tokens and 49 visually cut bboxes. 768 is combined embeddings.
        # attention_mask - extended_attention_mask(8, 1, 1, 561) contains 0 for attended word token and 49 visually cutted bboxes, -10000 for padding and others.
        # rel_pos(batch, num_attention_heads=12, 561, 561) relative position information, or neighbour information, is calculated for input positions and visual bboxes positions.
        # rel_2d_pos(batch, num_attention_heads=12, 561, 561) relative 2d position information, or neighbour information, is calculated for input bboxes and visually cut 49 bboxes.

        # projected query, key and value matrices.
        # q - query(batch, 561, 768), k - key(batch, 561, 768), v - value(batch, 561, 768)
        # each of query, key and value matrices is a concatenation of 12 head projection matrices.
        q, k, v = self.compute_qkv(hidden_states)

        # (B, L, H*D) -> (B, H, L, D)
        query_layer = self.transpose_for_scores(q)  # query_layer(batch, num_attention_heads=12, 561, attention_head_size=64)
        key_layer = self.transpose_for_scores(k)  # key_layer(batch, num_attention_heads=12, 561, attention_head_size=64)
        value_layer = self.transpose_for_scores(v)  # value_layer(batch, num_attention_heads=12, 561, attention_head_size=64)

        # this is a practice from transformer. Dividing the square root of the dimension of the key vectors used in the paper – 64.
        # This leads to having more stable gradients. There could be other possible values here, but this is the default.
        query_layer = query_layer / math.sqrt(self.attention_head_size)
        # [BSZ, NAT, L, L]
        # attention_scores(batch, num_attention_heads=12, 561, 561) means we have 561 tokens,
        # each tokens has attention score to all the 561 tokens. We have 12 attention head to express different attention needs.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        if self.has_relative_attention_bias:
            attention_scores += rel_pos  # we mixed in position ids relative position attention score.
        if self.has_spatial_attention_bias:
            attention_scores += rel_2d_pos  # we mixed in bbox 2d relative position attention score.
        attention_scores = attention_scores.float().masked_fill_(attention_mask.to(torch.bool), float("-inf"))  # replace padding tokens with -inf.
        # attention_probs(batch, num_attention_heads=12, 561, 561) softmax to get attention probabilities.
        attention_probs = F.softmax(attention_scores, dim=-1, dtype=torch.float32).type_as(value_layer)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        # context_layer(batch, num_attention_heads=12, 561, 64). Each token of 561 is weighted sum(attention prob) of 561 64-valued vectors.
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # context_layer(batch, num_attention_heads=12, 561, 64) -> (batch, 561, num_attention_heads=12, 64)
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)  # new_context_layer_shape(batch, 561, 768)
        context_layer = context_layer.view(*new_context_layer_shape)  # context_layer(batch, 561, 768) concatenate value vectors from 12 heads together.

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class LayoutLMv2Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = LayoutLMv2SelfAttention(config)
        self.output = LayoutLMv2SelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        rel_pos=None,
        rel_2d_pos=None,
    ):
        # hidden_states - final_emb(batch, 561=49+512, 768) combines text_layout_emb and visual_emb.
        # attention_mask - extended_attention_mask(8, 1, 1, 561) contains 0 for attended word token and 49 visually cutted bboxes, -10000 for padding and others.
        # rel_pos(batch, num_attention_heads=12, 561, 561) relative position information, or neighbour information, is calculated for input positions and visual bboxes positions.
        # rel_2d_pos(batch, num_attention_heads=12, 561, 561) relative 2d position information, or neighbour information, is calculated for input bboxes and visually cut 49 bboxes.

        # self_outputs[0](batch, 561, 768) concatenated value vectors from 12 heads together, with relative 1d and 2d position attention score mixed in.
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
            rel_pos=rel_pos,
            rel_2d_pos=rel_2d_pos,
        )
        # apply dense + dropout + layernorm(with residual connection, namely input hidden_state is added as the final input to layernorm).
        attention_output = self.output(self_outputs[0], hidden_states)  # attention_output(batch, 561, 768)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class LayoutLMv2Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = LayoutLMv2Attention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            assert self.is_decoder, f"{self} should be used as a decoder model if cross attention is added"
            self.crossattention = LayoutLMv2Attention(config)
        self.intermediate = LayoutLMv2Intermediate(config)
        self.output = LayoutLMv2Output(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        rel_pos=None,
        rel_2d_pos=None,
    ):
        # hidden_states - final_emb(batch, 561=49+512, 768) combines text_layout_emb and visual_emb.
        # attention_mask - extended_attention_mask(8, 1, 1, 561) contains 0 for attended word token and 49 visually cutted bboxes, -10000 for padding and others.
        # rel_pos(batch, num_attention_heads=12, 561, 561) relative position information, or neighbour information, is calculated for input positions and visual bboxes positions.
        # rel_2d_pos(batch, num_attention_heads=12, 561, 561) relative 2d position information, or neighbour information, is calculated for input bboxes and visually cut 49 bboxes.

        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
            rel_pos=rel_pos,
            rel_2d_pos=rel_2d_pos,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            assert hasattr(
                self, "crossattention"
            ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value
        # feed forward layers.
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


def relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
    ret = 0  # relative_position(batch, 561=512+49, 561) store relative positions with neighbors for all the 561 tokens. num_buckets=32, max_distance=128
    if bidirectional:
        num_buckets //= 2  # num_buckets=32//2=16
        ret += (relative_position > 0).long() * num_buckets  # (relative_position > 0).long() to indicate right neighbours. ret(batch, 561=512+49, 561) with right neighbours set to 16 and others set to 0.
        n = torch.abs(relative_position)  # n(batch, 561=512+49, 561), after abs, we cannot tell left or right neighbour with n.
    else:
        n = torch.max(-relative_position, torch.zeros_like(relative_position))
    # now n is in the range [0, inf)

    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2  # max_exact = 16 // 2 = 8
    is_small = n < max_exact  # is_small(batch, 561=512+49, 561) is to indicate the left and right neighbors that are within 8 steps. True for neighbors within 8 steps and False for others.

    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    # log(n/8) / log(128/8) * (32 - 8)
    # val_if_large(batch, 561=512+49, 561), for abs neighbors that are more than 8 steps away,
    # the relative position grow slowly. Namely, we don't differentiate much.
    val_if_large = max_exact + (
        torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
    ).to(torch.long)
    # we make sure the maximum relative position <= 15=num_buckets - 1.
    val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
    # ret(batch, 561=512+49, 561) with near neighbors(within 8 steps) described exactly and far away neighbors(more than 8 steps) described logarithmically.
    # left neighbors are in the range of (1 ~ 15), self is 0, and right neighbors are in the range of (17~31).
    ret += torch.where(is_small, n, val_if_large)
    return ret


class LayoutLMv2Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([LayoutLMv2Layer(config) for _ in range(config.num_hidden_layers)])

        self.has_relative_attention_bias = config.has_relative_attention_bias
        self.has_spatial_attention_bias = config.has_spatial_attention_bias

        if self.has_relative_attention_bias:
            self.rel_pos_bins = config.rel_pos_bins
            self.max_rel_pos = config.max_rel_pos
            self.rel_pos_onehot_size = config.rel_pos_bins
            self.rel_pos_bias = nn.Linear(self.rel_pos_onehot_size, config.num_attention_heads, bias=False)

        if self.has_spatial_attention_bias:
            self.max_rel_2d_pos = config.max_rel_2d_pos
            self.rel_2d_pos_bins = config.rel_2d_pos_bins
            self.rel_2d_pos_onehot_size = config.rel_2d_pos_bins
            self.rel_pos_x_bias = nn.Linear(self.rel_2d_pos_onehot_size, config.num_attention_heads, bias=False)
            self.rel_pos_y_bias = nn.Linear(self.rel_2d_pos_onehot_size, config.num_attention_heads, bias=False)

    def _cal_1d_pos_emb(self, hidden_states, position_ids):
        # hidden_states - final_emb(batch, 561=49+512, 768) combines text_layout_emb and visual_emb.
        # position_ids - final_position_ids(batch_size, 561 = 512 + 49) combines input position ids[0, 1, ..., 510, 511] and visual bboxes position ids [0, 1, ..., 47, 48]

        # rel_pos_mat(batch, 561 = 512 + 49, 561) = position_ids.unsqueeze(-2)(batch, 1, 561) - position_ids.unsqueeze(-1)(batch, 561, 1)
        # rel_pos_mat[0, 0, :] = [ 0,  1, ..., 510, 511,  0,  1, ..., 47, 48]
        # rel_pos_mat[0, 1, :] = [-1,  0, ..., 509, 510, -1,  0, ..., 46, 47]
        # rel_pos_mat[0, 2, :] = [-2, -1, ..., 508, 509, -2, -1, ..., 45, 46]
        # rel_pos_mat store relative positions with neighbors for all the 561 tokens.
        rel_pos_mat = position_ids.unsqueeze(-2) - position_ids.unsqueeze(-1)
        # rel_pos(batch, 561=512+49, 561) with near neighbors(within 8 steps) described exactly and far away neighbors(more than 8 steps) described logarithmically.
        # left neighbors are in the range of (1 ~ 15), self is 0, and right neighbors are in the range of (17~31).
        # rel_pos basically to describe near-by neighbour exactly and far-away neighbour briefly.
        rel_pos = relative_position_bucket(
            rel_pos_mat,
            num_buckets=self.rel_pos_bins,
            max_distance=self.max_rel_pos,
        )
        # rel_pos(batch, 561=512+49, 561, 32) encoded as one-hot representation.
        rel_pos = F.one_hot(rel_pos, num_classes=self.rel_pos_onehot_size).type_as(hidden_states)
        # project rel_pos into attention space. 32 one-hot values of each token is projected as an attention score for each attention head,
        # we have 12 attention score for 12 attention heads. Each token has attention scores to all other tokens.
        # These relative position attention scores will be combined in attention layer.
        # rel_pos(batch, 561=512+49, 561, 32) -> self.rel_pos_bias(rel_pos)(batch, 561, 561, num_attention_heads=12)
        #   -> permute(0, 3, 1, 2)(batch, num_attention_heads=12, 561, 561)
        rel_pos = self.rel_pos_bias(rel_pos).permute(0, 3, 1, 2)
        rel_pos = rel_pos.contiguous()  # rel_pos(batch, num_attention_heads=12, 561, 561)
        return rel_pos

    def _cal_2d_pos_emb(self, hidden_states, bbox):
        # hidden_states - final_emb(batch, 561=49+512, 768) combines text_layout_emb and visual_emb.
        # bbox - final_bbox(batch_size, 561(512 + 49), 4) contains input bboxes and visually cut 49 bboxes.

        position_coord_x = bbox[:, :, 0]  # position_coord_x(batch, 561) is top left x positions.
        position_coord_y = bbox[:, :, 3]  # position_coord_y(batch, 561) is bottom right y positions.
        # rel_pos_x_2d_mat store relative positions of neighbors for all the 561 tokens' top left x.
        rel_pos_x_2d_mat = position_coord_x.unsqueeze(-2) - position_coord_x.unsqueeze(-1)
        # rel_pos_y_2d_mat store relative positions of neighbors for all the 561 tokens' bottom right y.
        rel_pos_y_2d_mat = position_coord_y.unsqueeze(-2) - position_coord_y.unsqueeze(-1)
        # rel_pos_x/rel_pos_y(batch, 561=512+49, 561) with near neighbors(within 16 steps) described exactly and far away neighbors(more than 16 steps) described logarithmically.
        # left neighbors are in the range of (1 ~ 31), self is 0, and right neighbors are in the range of (33~63).
        # rel_pos_x/rel_pos_y basically to describe near-by neighbour exactly and far-away neighbour briefly.
        rel_pos_x = relative_position_bucket(
            rel_pos_x_2d_mat,
            num_buckets=self.rel_2d_pos_bins,
            max_distance=self.max_rel_2d_pos,
        )
        rel_pos_y = relative_position_bucket(
            rel_pos_y_2d_mat,
            num_buckets=self.rel_2d_pos_bins,
            max_distance=self.max_rel_2d_pos,
        )
        # rel_pos_x/rel_pos_y(batch, 561=512+49, 561, 64) encoded as one-hot representation.
        rel_pos_x = F.one_hot(rel_pos_x, num_classes=self.rel_2d_pos_onehot_size).type_as(hidden_states)
        rel_pos_y = F.one_hot(rel_pos_y, num_classes=self.rel_2d_pos_onehot_size).type_as(hidden_states)
        # project rel_pos_x/rel_pos_y into attention space. 64 one-hot values of each token is projected as an attention score for each attention head,
        # we have 12 attention score for 12 attention heads. Each token has attention scores to all other tokens.
        # These relative position attention scores will be combined in attention layer.
        # rel_pos_x/rel_pos_y(batch, 561=512+49, 561, 64) -> self.rel_pos_bias(rel_pos)(batch, 561, 561, num_attention_heads=12)
        #   -> permute(0, 3, 1, 2)(batch, num_attention_heads=12, 561, 561)
        rel_pos_x = self.rel_pos_x_bias(rel_pos_x).permute(0, 3, 1, 2)
        rel_pos_y = self.rel_pos_y_bias(rel_pos_y).permute(0, 3, 1, 2)
        # rel_pos_x/rel_pos_y(batch, num_attention_heads=12, 561, 561)
        rel_pos_x = rel_pos_x.contiguous()
        rel_pos_y = rel_pos_y.contiguous()
        rel_2d_pos = rel_pos_x + rel_pos_y  # combine x and y.
        return rel_2d_pos

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        bbox=None,
        position_ids=None,
    ):
        # hidden_states - final_emb(batch, 561=49+512, 768) combines text_layout_emb and visual_emb.
        # attention_mask - extended_attention_mask(8, 1, 1, 561) contains 0 for attended word token and 49 visually cutted bboxes, -10000 for padding and others.
        # bbox - final_bbox(batch_size, 561(512 + 49), 4) contains input bboxes and visually cut 49 bboxes.
        # position_ids - final_position_ids(batch_size, 561 = 512 + 49) combines input position ids[1, 2, ..., 510, 511] and visual bboxes position ids [1, 2, ..., 47, 48]
        # head_mask list of 12 None for 12 layers.
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        # rel_pos(batch, num_attention_heads=12, 561, 561) relative position information, or neighbour information, is calculated for input positions and visual bboxes positions.
        rel_pos = self._cal_1d_pos_emb(hidden_states, position_ids) if self.has_relative_attention_bias else None
        # rel_2d_pos(batch, num_attention_heads=12, 561, 561) relative 2d position information, or neighbour information, is calculated for input bboxes and visually cut 49 bboxes.
        rel_2d_pos = self._cal_2d_pos_emb(hidden_states, bbox) if self.has_spatial_attention_bias else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                if use_cache:
                    logger.warn(
                        "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                        "`use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    rel_pos=rel_pos,
                    rel_2d_pos=rel_2d_pos,
                )
            else:
                # hidden_states - final_emb(batch, 561=49+512, 768) combines text_layout_emb and visual_emb.
                # attention_mask - extended_attention_mask(8, 1, 1, 561) contains 0 for attended word token and 49 visually cutted bboxes, -10000 for padding and others.
                # rel_pos(batch, num_attention_heads=12, 561, 561) relative position information, or neighbour information, is calculated for input positions and visual bboxes positions.
                # rel_2d_pos(batch, num_attention_heads=12, 561, 561) relative 2d position information, or neighbour information, is calculated for input bboxes and visually cut 49 bboxes.
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                    rel_pos=rel_pos,
                    rel_2d_pos=rel_2d_pos,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class LayoutLMv2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = LayoutLMv2Config
    pretrained_model_archive_map = LAYOUTLMV2_PRETRAINED_MODEL_ARCHIVE_LIST
    base_model_prefix = "layoutlmv2"
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, LayoutLMv2LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


def my_convert_sync_batchnorm(module, process_group=None):
    # same as `nn.modules.SyncBatchNorm.convert_sync_batchnorm` but allowing converting from `detectron2.layers.FrozenBatchNorm2d`
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        return nn.modules.SyncBatchNorm.convert_sync_batchnorm(module, process_group)
    module_output = module
    if isinstance(module, detectron2.layers.FrozenBatchNorm2d):
        module_output = torch.nn.SyncBatchNorm(
            num_features=module.num_features,
            eps=module.eps,
            affine=True,
            track_running_stats=True,
            process_group=process_group,
        )
        module_output.weight = torch.nn.Parameter(module.weight)
        module_output.bias = torch.nn.Parameter(module.bias)
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = torch.tensor(0, dtype=torch.long, device=module.running_mean.device)
    for name, child in module.named_children():
        module_output.add_module(name, my_convert_sync_batchnorm(child, process_group))
    del module
    return module_output


class VisualBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cfg = detectron2.config.get_cfg()
        add_layoutlmv2_config(self.cfg)
        meta_arch = self.cfg.MODEL.META_ARCHITECTURE
        model = META_ARCH_REGISTRY.get(meta_arch)(self.cfg)
        assert isinstance(model.backbone, detectron2.modeling.backbone.FPN)
        self.backbone = model.backbone
        if (
            config.convert_sync_batchnorm
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_rank() > -1
        ):
            self_rank = torch.distributed.get_rank()
            node_size = torch.cuda.device_count()
            world_size = torch.distributed.get_world_size()
            assert world_size % node_size == 0

            node_global_ranks = [
                list(range(i * node_size, (i + 1) * node_size)) for i in range(world_size // node_size)
            ]
            sync_bn_groups = [
                torch.distributed.new_group(ranks=node_global_ranks[i]) for i in range(world_size // node_size)
            ]
            node_rank = self_rank // node_size
            assert self_rank in node_global_ranks[node_rank]

            self.backbone = my_convert_sync_batchnorm(self.backbone, process_group=sync_bn_groups[node_rank])

        assert len(self.cfg.MODEL.PIXEL_MEAN) == len(self.cfg.MODEL.PIXEL_STD)
        num_channels = len(self.cfg.MODEL.PIXEL_MEAN)
        self.register_buffer(
            "pixel_mean",
            torch.Tensor(self.cfg.MODEL.PIXEL_MEAN).view(num_channels, 1, 1),
        )  # pixel_mean is assigned here.
        self.register_buffer("pixel_std", torch.Tensor(self.cfg.MODEL.PIXEL_STD).view(num_channels, 1, 1))  # pixel_std is assigned here.
        self.out_feature_key = "p2"
        if torch.is_deterministic():
            logger.warning("using `AvgPool2d` instead of `AdaptiveAvgPool2d`")
            input_shape = (224, 224)
            backbone_stride = self.backbone.output_shape()[self.out_feature_key].stride
            self.pool = nn.AvgPool2d(
                (
                    math.ceil(math.ceil(input_shape[0] / backbone_stride) / config.image_feature_pool_shape[0]),
                    math.ceil(math.ceil(input_shape[1] / backbone_stride) / config.image_feature_pool_shape[1]),
                )
            )
        else:
            self.pool = nn.AdaptiveAvgPool2d(config.image_feature_pool_shape[:2])
        if len(config.image_feature_pool_shape) == 2:
            config.image_feature_pool_shape.append(self.backbone.output_shape()[self.out_feature_key].channels)
        assert self.backbone.output_shape()[self.out_feature_key].channels == config.image_feature_pool_shape[2]

    def forward(self, images):  # images(batch, channels, width, height)
        images_input = ((images if torch.is_tensor(images) else images.tensor) - self.pixel_mean) / self.pixel_std  # standardized pixels.
        features = self.backbone(images_input)
        features = features[self.out_feature_key]  # features(batch, 256, 56, 56)
        features = self.pool(features).flatten(start_dim=2).transpose(1, 2).contiguous()  # self.pool(features)(batch, 256, 7, 7) -> flatten(start_dim=2)(batch, 256, 49) -> transpose(1, 2)(batch, 49, 256) -> contiguous() remain the same.
        return features


class LayoutLMv2Model(LayoutLMv2PreTrainedModel):
    def __init__(self, config):
        super(LayoutLMv2Model, self).__init__(config)
        self.config = config
        self.has_visual_segment_embedding = config.has_visual_segment_embedding
        self.embeddings = LayoutLMv2Embeddings(config)

        self.visual = VisualBackbone(config)
        self.visual_proj = nn.Linear(config.image_feature_pool_shape[-1], config.hidden_size)
        if self.has_visual_segment_embedding:
            self.visual_segment_embedding = nn.Parameter(nn.Embedding(1, config.hidden_size).weight[0])
        self.visual_LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.visual_dropout = nn.Dropout(config.hidden_dropout_prob)

        self.encoder = LayoutLMv2Encoder(config)
        self.pooler = LayoutLMv2Pooler(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def _calc_text_embeddings(self, input_ids, bbox, position_ids, token_type_ids):  # input_ids(batch_size, 512), bbox(batch_size, 512, 4), position_ids(batch_size, 512), token_type_ids(batch_size, 512)
        seq_length = input_ids.size(1)  # seq_length(512)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.embeddings.word_embeddings(input_ids)  # (batch_size, 512) -> (batch_size, 512, 768), taking embedding by word index.
        position_embeddings = self.embeddings.position_embeddings(position_ids)  # (batch_size, 512) -> (batch_size, 512, 768), taking embedding by position index.
        spatial_position_embeddings = self.embeddings._cal_spatial_position_embeddings(bbox)  # spatial_position_embeddings(batch_size, 512, 768(128 * 6)), which combines x1, y1, x2, y2, width and height embeddings.
        token_type_embeddings = self.embeddings.token_type_embeddings(token_type_ids)  # self.embeddings.token_type_embeddings(2, 768). token_type_ids(batch_size, 512) -> (batch_size, 768).
        embeddings = words_embeddings + position_embeddings + spatial_position_embeddings + token_type_embeddings  # combine 4 embedding vectors to produce final embedding vector: word vector, position vector, bounding box vector and token type vector.
        embeddings = self.embeddings.LayerNorm(embeddings)  # normalize by mean and std on the last embedding dim(768).
        embeddings = self.embeddings.dropout(embeddings)
        return embeddings

    def _calc_img_embeddings(self, image, bbox, position_ids):
        # bbox(batch, 49, 4) is visual bbox cutting 0 to 1000 area into 49 boxes.
        # position_ids(batch, 49) is visual position ids for 49 visual boxes.
        # image(batch, channel, width, height)

        # image(batch, channel, width, height) -> self.visual(image) to apply visual cnn to get (batch, 49, 256)
        #   -> self.visual_proj() project image to match text embedding dim to get (batch, 49, 768).
        visual_embeddings = self.visual_proj(self.visual(image))
        # mapping to its position embedding by index to get position_embeddings(batch, 49, 768).
        position_embeddings = self.embeddings.position_embeddings(position_ids)
        # spatial_position_embeddings(batch_size, 49, 768(128 * 6)),
        # which combines x1, y1, x2, y2, width and height embeddings for 49 visual bboxes.
        spatial_position_embeddings = self.embeddings._cal_spatial_position_embeddings(bbox)
        # embeddings(batch, 49, 768) combines 3 pieces of information: image representation vector,
        # visually cutted 49 bboxes vector, visually cutted 49 bboxes position vector.
        embeddings = visual_embeddings + position_embeddings + spatial_position_embeddings
        if self.has_visual_segment_embedding:  # we don't use has_visual_segment_embedding?
            embeddings += self.visual_segment_embedding
        embeddings = self.visual_LayerNorm(embeddings)  # normalize by mean and std on the last embedding dim(768).
        embeddings = self.visual_dropout(embeddings)
        return embeddings

    def forward(
        self,
        input_ids=None,
        bbox=None,
        image=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):  # attention_mask(batch, 512) contains 1 for word token to attend and 0 for padding or others.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        visual_shape = list(input_shape)  # input_shape(batch_size, 512)
        visual_shape[1] = self.config.image_feature_pool_shape[0] * self.config.image_feature_pool_shape[1]  # 7 * 7 = 49
        visual_shape = torch.Size(visual_shape)  # (batch_size, 7 * 7 = 49)
        final_shape = list(input_shape)
        final_shape[1] += visual_shape[1]
        final_shape = torch.Size(final_shape)  # batch_size * 561(512 + 49), which merges input_shape and visual_shape.

        visual_bbox_x = (
            torch.arange(
                0,
                1000 * (self.config.image_feature_pool_shape[1] + 1),
                1000,
                device=device,
                dtype=bbox.dtype,
            )
            // self.config.image_feature_pool_shape[1]
        )  # tensor([   0,  142,  285,  428,  571,  714,  857, 1000])
        visual_bbox_y = (
            torch.arange(
                0,
                1000 * (self.config.image_feature_pool_shape[0] + 1),
                1000,
                device=device,
                dtype=bbox.dtype,
            )
            // self.config.image_feature_pool_shape[0]
        )  # tensor([   0,  142,  285,  428,  571,  714,  857, 1000])
        visual_bbox = torch.stack(
            [
                visual_bbox_x[:-1].repeat(self.config.image_feature_pool_shape[0], 1),
                visual_bbox_y[:-1].repeat(self.config.image_feature_pool_shape[1], 1).transpose(0, 1),
                visual_bbox_x[1:].repeat(self.config.image_feature_pool_shape[0], 1),
                visual_bbox_y[1:].repeat(self.config.image_feature_pool_shape[1], 1).transpose(0, 1),
            ],
            dim=-1,
        ).view(-1, bbox.size(-1))  # shape is (49, 4). All combinations of visual_bbox_x and visual_bbox_y. The first box is [0, 0, 142, 142] and the last box is [ 857,  857, 1000, 1000]. Basically to cut 0 to 1000 area into 49 boxes.
        visual_bbox = visual_bbox.repeat(final_shape[0], 1, 1)  # replicate visual_bbox batch_size times. The shape becomes (batch_size, 49, 4).
        final_bbox = torch.cat([bbox, visual_bbox], dim=1)  # final_bbox(batch_size, 561(512 + 49), 4) contains input bboxes and visually cutted 49 bboxes.

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)

        visual_attention_mask = torch.ones(visual_shape, device=device)  # (batch_size, 49)
        final_attention_mask = torch.cat([attention_mask, visual_attention_mask], dim=1)  # final_attention_mask(batch_size, 561(512 + 49)) combines input text attention and 49 visually cutted bboxes attention.

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if position_ids is None:
            seq_length = input_shape[1]
            position_ids = self.embeddings.position_ids[:, :seq_length]  # shape (1, 512) and value [[0, 1, ..., 510, 511]]
            position_ids = position_ids.expand_as(input_ids)  # expand as the same as input_ids, which is of shape (batch_size, 512). Therefore, position_ids is replicated batch_size times.

        visual_position_ids = torch.arange(0, visual_shape[1], dtype=torch.long, device=device).repeat(
            input_shape[0], 1
        )  # shape (batch_size, 49) and value of visual_position_ids[0] is [0, 1, ..., 47, 48], visual position ids for 49 visual boxes.
        final_position_ids = torch.cat([position_ids, visual_position_ids], dim=1)  # final_position_ids(batch_size, 561 = 512 + 49) combines input position ids[1, 2, ..., 510, 511] and visual bboxes position ids [1, 2, ..., 47, 48]

        if bbox is None:
            bbox = torch.zeros(tuple(list(input_shape) + [4]), dtype=torch.long, device=device)
        # text_layout_emb(batch, 512, 768) combines 4 pieces of information: word vector, position vector,
        # bounding box vector and token type vector.
        text_layout_emb = self._calc_text_embeddings(
            input_ids=input_ids,
            bbox=bbox,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        # visual_emb(batch, 49, 768) combines 3 pieces of information: 49 convolved image representation vector,
        # visually cut 49 bboxes vector, visually cut 49 bboxes position vector.
        visual_emb = self._calc_img_embeddings(
            image=image,
            bbox=visual_bbox,
            position_ids=visual_position_ids,
        )
        final_emb = torch.cat([text_layout_emb, visual_emb], dim=1)  # final_emb(batch, 561=49+512, 768) combines text_layout_emb and visual_emb.

        extended_attention_mask = final_attention_mask.unsqueeze(1).unsqueeze(2)  # final_attention_mask(batch, 561=512+49) -> extended_attention_mask(batch, 1, 1, 561)

        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0  # extended_attention_mask contains 0 for attended word token and 49 visually cutted bboxes, -10000 for padding and others.

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)
        else:
            head_mask = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            final_emb,
            extended_attention_mask,
            bbox=final_bbox,
            position_ids=final_position_ids,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class LayoutLMv2ForTokenClassification(LayoutLMv2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.layoutlmv2 = LayoutLMv2Model(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def get_input_embeddings(self):
        return self.layoutlmv2.embeddings.word_embeddings

    def forward(
        self,
        input_ids=None,
        bbox=None,
        image=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.layoutlmv2(
            input_ids=input_ids,
            bbox=bbox,
            image=image,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        seq_length = input_ids.size(1)
        sequence_output, image_output = outputs[0][:, :seq_length], outputs[0][:, seq_length:]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()

            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class LayoutLMv2ForRelationExtraction(LayoutLMv2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.layoutlmv2 = LayoutLMv2Model(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.extractor = REDecoder(config)
        self.init_weights()

    def forward(
        self,
        input_ids,
        bbox,
        labels=None,
        image=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        entities=None,
        relations=None,
    ):
        outputs = self.layoutlmv2(
            input_ids=input_ids,
            bbox=bbox,
            image=image,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )

        seq_length = input_ids.size(1)
        sequence_output, image_output = outputs[0][:, :seq_length], outputs[0][:, seq_length:]
        sequence_output = self.dropout(sequence_output)
        loss, pred_relations = self.extractor(sequence_output, entities, relations)

        return ReOutput(
            loss=loss,
            entities=entities,
            relations=relations,
            pred_relations=pred_relations,
            hidden_states=outputs[0],
        )

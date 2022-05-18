# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch LayoutLMv3 model. """
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers import apply_chunking_to_forward
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
    TokenClassifierOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.models.roberta.modeling_roberta import (
    RobertaIntermediate,
    RobertaLMHead,
    RobertaOutput,
    RobertaSelfOutput,
)
from transformers.utils import logging

from .configuration_layoutlmv3 import LayoutLMv3Config
from timm.models.layers import to_2tuple


logger = logging.get_logger(__name__)


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        # The following variables are used in detection mycheckpointer.py
        self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.num_patches_w = self.patch_shape[0]
        self.num_patches_h = self.patch_shape[1]

    def forward(self, x, position_embedding=None):
        # x(batch_size, channel_size, height, width) -> x(batch_size, emd_size(# filters), height, width)
        x = self.proj(x)

        if position_embedding is not None:
            # interpolate the position embedding to the corresponding size
            position_embedding = position_embedding.view(1, self.patch_shape[0], self.patch_shape[1], -1).permute(0, 3, 1, 2)
            Hp, Wp = x.shape[2], x.shape[3]
            position_embedding = F.interpolate(position_embedding, size=(Hp, Wp), mode='bicubic')
            x = x + position_embedding
        # x(batch_size, n_patches, emd_size)
        # x is 2d convolution result,which convolves patch by batch. Now each patch is represented by #emb_size numbers.
        x = x.flatten(2).transpose(1, 2)
        return x

class LayoutLMv3Embeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.__init__
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

        # End copy
        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

        self.x_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.coordinate_size)
        self.y_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.coordinate_size)
        self.h_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.shape_size)
        self.w_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.shape_size)

    def _calc_spatial_position_embeddings(self, bbox):
        # bbox(batch_size, seq_len, 4)
        try:
            assert torch.all(0 <= bbox) and torch.all(bbox <= 1023)
            # left_position_embeddings(batch_size, seq_len, 128); All possible x values from 0 to 1000(max 1024)
            # are embedded into 128 dimensional space.
            left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
            upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
            right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
            lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])
        except IndexError as e:
            raise IndexError("The :obj:`bbox` coordinate values should be within 0-1000 range.") from e

        h_position_embeddings = self.h_position_embeddings(torch.clip(bbox[:, :, 3] - bbox[:, :, 1], 0, 1023))
        w_position_embeddings = self.w_position_embeddings(torch.clip(bbox[:, :, 2] - bbox[:, :, 0], 0, 1023))
        # spatial_position_embeddings(batch_size, seq_len, emd_size) combines left, right, upper, lower, height
        # and width information from bbox.
        # below is the difference between LayoutLMEmbeddingsV2 (torch.cat) and LayoutLMEmbeddingsV1 (add)
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

    def create_position_ids_from_input_ids(self, input_ids, padding_idx, past_key_values_length=0):
        """
        Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
        are ignored. This is modified from fairseq's `utils.make_positions`.

        Args:
            x: torch.Tensor x:

        Returns: torch.Tensor
        """
        # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
        mask = input_ids.ne(padding_idx).int()
        incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
        return incremental_indices.long() + padding_idx

    def forward(
        self,
        input_ids=None,
        bbox=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        past_key_values_length=0,
    ):
        # LayoutLMv3Embeddings is to create text embeddings.

        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                # position_ids(batch_size, seq_len)[[2, 3, 4, ..., 1], ...]
                # position id starts from 2 with padding id 1.
                position_ids = self.create_position_ids_from_input_ids(
                    input_ids, self.padding_idx, past_key_values_length).to(input_ids.device)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            # input_ids(batch_size, seq_len); inputs_embeds(batch_size, seq_len, emd_size)
            inputs_embeds = self.word_embeddings(input_ids)
        # token_type_ids(batch_size, seq_len); token_type_embeddings(batch_size, seq_len, emd_size)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        # embeddings(batch_size, seq_len, emd_size).
        # embeddings = inputs_embeds + token_type_embeddings + position_embeddings + spatial_position_embeddings.
        # spatial_position_embeddings(batch_size, seq_len, emd_size) combines left, right, upper, lower, height and
        # width information from bbox.
        embeddings = inputs_embeds + token_type_embeddings
        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings

        spatial_position_embeddings = self._calc_spatial_position_embeddings(bbox)

        embeddings = embeddings + spatial_position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor≈

        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)


class LayoutLMv3PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = LayoutLMv3Config
    base_model_prefix = "layoutlmv3"

    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
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
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class LayoutLMv3SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.has_relative_attention_bias = config.has_relative_attention_bias
        self.has_spatial_attention_bias = config.has_spatial_attention_bias

    def transpose_for_scores(self, x):
        # We project x together for all attention heads instead of one by one. Each input token is described be
        # #proj_size values. Namely, each head has #head_proj_size=#proj_size/#n_attention_heads values.
        # x(batch_size, input_patch_len, proj_size)
        #   -> x.view(*new_x_shape)(batch_size, input_patch_len, n_attention_heads, head_proj_size)
        #   -> x.permute(0, 2, 1, 3)(batch_size, n_attention_heads, input_patch_len, head_proj_size)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def cogview_attn(self, attention_scores, alpha=32):
        '''
        https://arxiv.org/pdf/2105.13290.pdf
        Section 2.4 Stabilization of training: Precision Bottleneck Relaxation (PB-Relax).
        A replacement of the original nn.Softmax(dim=-1)(attention_scores)
        Seems the new attention_probs will result in a slower speed and a little bias
        Can use torch.allclose(standard_attention_probs, cogview_attention_probs, atol=1e-08) for comparison
        The smaller atol (e.g., 1e-08), the better.
        '''
        scaled_attention_scores = attention_scores / alpha
        max_value = scaled_attention_scores.amax(dim=(-1)).unsqueeze(-1)
        # max_value = scaled_attention_scores.amax(dim=(-2, -1)).unsqueeze(-1).unsqueeze(-1)
        new_attention_scores = (scaled_attention_scores - max_value) * alpha
        return nn.Softmax(dim=-1)(new_attention_scores)

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
        # hidden_states(batch_size, input_patch_len, emd_size) contains both textual embedding and visual embedding.
        # attention_mask(batch_size, 1, 1, seq_len+n_batches+1) contains value of 0.0 to attend
        #   and -10000(big negative value) to ignore. as these values will be added to raw value before softmax.
        # rel_pos(batch, num_attention_heads=12, input_patch_len, input_patch_len) relative position information
        #   or neighbour information. It is calculated for input positions and patch positions.
        # rel_2d_pos(batch, num_attention_heads=12, input_patch_len, input_patch_len) relative 2d position
        #   information or neighbour information. It is calculated for input bboxes and patch bboxes.

        # mixed_query_layer(batch_size, input_patch_len, emd_size) compute projected query matrix.
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            # We project hidden_states into key/value space together for all attention heads instead of one by one.
            # Each input token is described be #proj_size values. Namely, each head
            # has #head_proj_size=#proj_size/#n_attention_heads values, which is done by self.transpose_for_scores.
            #
            # self.key/value(hidden_states)(batch_size, input_patch_len, emd_size)
            #   -> self.transpose_for_scores(...)(batch_size, n_attn_heads, input_patch_len, head_proj_size)
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
        # Same as key_layer/value_layer.
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # The attention scores QT K/√d could be significantly larger than input elements, and result in overflow.
        # Changing the computational order into QT(K/√d) alleviates the problem. (https://arxiv.org/pdf/2105.13290.pdf)
        # query_layer/key_layer(batch_size, n_attn_heads, input_patch_len, head_proj_size).
        # Each matrix(input_patch_len, head_proj_size) is representation of all tokens for an attention head.
        # When we dot product [matrix(input_patch_len, head_proj_size) dot matrix(head_proj_size, input_patch_len)],
        # we get correlation between tokens(or covariance with different in magnitude), which can be interpreted as
        # raw attention scores.
        # attention_scores(batch_size, n_attn_heads, input_patch_len, input_patch_len)
        attention_scores = torch.matmul(query_layer / math.sqrt(self.attention_head_size), key_layer.transpose(-1, -2))

        if self.has_relative_attention_bias and self.has_spatial_attention_bias:
            # we mix in position ids and bbox 2d relative position attention score.
            # rel_pos contains raw attention scores of relative position(or relative position correlation information).
            attention_scores += (rel_pos + rel_2d_pos) / math.sqrt(self.attention_head_size)
        elif self.has_relative_attention_bias:
            attention_scores += rel_pos / math.sqrt(self.attention_head_size)

        # if self.has_relative_attention_bias:
        #     attention_scores += rel_pos
        # if self.has_spatial_attention_bias:
        #     attention_scores += rel_2d_pos

        # attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
            # attention_mask is applied to raw attention_scores before softmax, so that -10,000 for masked value will
            # be equivalent to mask out the field. attention_scores(
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        # attention_probs = nn.Softmax(dim=-1)(attention_scores)  # comment the line below and use this line for speedup
        # attention_probs(batch_size, n_attn_heads, input_patch_len, input_patch_len) softmax to get attention probabilities.
        attention_probs = self.cogview_attn(attention_scores)  # to stablize training
        # assert torch.allclose(attention_probs, nn.Softmax(dim=-1)(attention_scores), atol=1e-8)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        # context_layer(batch, n_attention_heads=12, input_patch_len, head_proj_size).
        # Each token is weighted sum of all 64-valued token vectors, by row dot matrix view of the two matrices below.
        # [attention_probs(..., input_patch_len, input_patch_len) dot value_layer (...,input_patch_len,head_proj_size)]
        context_layer = torch.matmul(attention_probs, value_layer)
        # context_layer(batch_size, input_patch_len, n_attention_heads, head_proj_size).
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        # context_layer(batch_size, input_patch_len, emd_size)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class LayoutLMv3Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = LayoutLMv3SelfAttention(config)
        self.output = RobertaSelfOutput(config)
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
        attention_output = self.output(self_outputs[0], hidden_states)  # hidden_states is for skip connection.
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class LayoutLMv3Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = LayoutLMv3Attention(config)
        assert not config.is_decoder and not config.add_cross_attention, \
            "This version do not support decoder. Please refer to RoBERTa for implementation of is_decoder."
        self.intermediate = RobertaIntermediate(config)
        self.output = RobertaOutput(config)

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
        # hidden_states(batch_size, input_patch_len, emd_size) contains both textual embedding and visual embedding.
        # attention_mask(batch_size, 1, 1, seq_len+n_batches+1) contains value of 0.0 to attend
        #   and -10000(big negative value) to ignore. as these values will be added to raw value before softmax.
        # rel_pos(batch, num_attention_heads=12, input_patch_len, input_patch_len) relative position information
        #   or neighbour information. It is calculated for input positions and patch positions.
        # rel_2d_pos(batch, num_attention_heads=12, input_patch_len, input_patch_len) relative 2d position
        #   information or neighbour information. It is calculated for input bboxes and patch bboxes.

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

        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights
        # feed forward layers.
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class LayoutLMv3Encoder(nn.Module):
    def __init__(self, config, detection=False, out_features=None):
        super().__init__()
        self.config = config
        self.detection = detection
        self.layer = nn.ModuleList([LayoutLMv3Layer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

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

        if self.detection:
            self.gradient_checkpointing = True
            embed_dim = self.config.hidden_size
            self.out_features = out_features
            self.out_indices = [int(name[5:]) for name in out_features]
            self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                # nn.SyncBatchNorm(embed_dim),
                nn.BatchNorm2d(embed_dim),
                nn.GELU(),
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )

            self.fpn2 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )

            self.fpn3 = nn.Identity()

            self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]

    def relative_position_bucket(self, relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        # relative_position(batch_size, seq_len+n_patches+1, seq_len+n_patches+1) store relative positions with
        # neighbors for all the (seq_len+n_patches+1) tokens.
        ret = 0
        if bidirectional:
            num_buckets //= 2  # num_buckets=32//2=16
            # (relative_position > 0).long() to indicate right neighbours.
            # ret(batch_size, seq_len+n_patches+1, seq_len+n_patches+1) with right neighbours set to 16
            # and others set to 0.
            ret += (relative_position > 0).long() * num_buckets
            # n(batch_size, seq_len+n_patches+1, seq_len+n_patches+1), after abs, we cannot tell left or
            # right neighbour with n.
            n = torch.abs(relative_position)
        else:
            n = torch.max(-relative_position, torch.zeros_like(relative_position))
        # now n is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2  # max_exact = 16 // 2 = 8
        # is_small(batch, seq_len+n_patches+1, seq_len+n_patches+1) is to indicate the left and right neighbors
        # that are within 8 steps. True for neighbors within 8 steps and False for others.
        is_small = n < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        val_if_large = max_exact + (
                torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).to(torch.long)
        # log(n/8) / log(128/8) * (32 - 8)
        # val_if_large(batch, seq_len+n_patches+1, seq_len+n_patches+1), for abs neighbors that are more than 8 steps
        # away, the relative position grow slowly. Namely, we don't differentiate much.
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        # ret(batch, 561=512+49, 561) with near neighbors(within 8 steps) described exactly(by n) and
        # far away neighbors(more than 8 steps) described logarithmically (by val_if_large).
        # left neighbors are in the range of (1 ~ 15), self is 0, and right neighbors are in the range of (17~31).
        ret += torch.where(is_small, n, val_if_large)
        return ret

    def _cal_1d_pos_emb(self, hidden_states, position_ids):
        # hidden_states(batch_size, seq_len+n_patches+1, emd_size) contains both textual embedding and visual embedding.
        # position_ids(batch_size, seq_len+n_patches+1)[[0,1,2...511,0,1, 196], ...]

        # rel_pos_mat(batch_size, seq_len+n_patches+1, seq_len+n_patches+1) =
        #       position_ids.unsqueeze(-2)(batch_size, 1, seq_len+n_patches+1)
        #       - position_ids.unsqueeze(-1)(batch_size, seq_len+n_patches+1, 1)
        # Each matrix in rel_pos_mat is (seq_len+n_patches+1, seq_len+n_patches+1). rel_pos_mat[0, 0, :]
        # represents difference between all positions to the first position. rel_pos_mat[0, 1, :] represents difference
        # between all the positions to the second position.
        # rel_pos_mat[0, 0, 1] will have opposite value to rel_pos_mat[0, 1, 0], as rel_pos_mat[0, 0, 1] represents
        # difference between position 1 and position 0, and rel_pos_mat[0, 1, 0] represents difference between
        # position 0 and position 1.
        # rel_pos_mat[0, 0, :] = [ 0,  1, ..., 510, 511,  0,  1, ..., 195, 196]
        # rel_pos_mat[0, 1, :] = [-1,  0, ..., 509, 510, -1,  0, ..., 194, 195]
        # rel_pos_mat[0, 2, :] = [-2, -1, ..., 508, 509, -2, -1, ..., 193, 194]
        # rel_pos_mat store relative positions with neighbors for all the (seq_len+n_patches+1) tokens.
        rel_pos_mat = position_ids.unsqueeze(-2) - position_ids.unsqueeze(-1)
        # rel_pos(batch_size, seq_len+n_patches+1, seq_len+n_patches+1) with near neighbors(within 8 steps) described
        # exactly and far away neighbors(more than 8 steps) described logarithmically.
        # left neighbors are in the range of (1 ~ 15), self is 0, and right neighbors are in the range of (17~31).
        # rel_pos basically to describe near-by neighbour exactly and far-away neighbour briefly.
        rel_pos = self.relative_position_bucket(
            rel_pos_mat,
            num_buckets=self.rel_pos_bins,
            max_distance=self.max_rel_pos,
        )
        # rel_pos(batch, seq_len+n_patches+1, seq_len+n_patches+1, 32) encoded as one-hot representation.
        rel_pos = F.one_hot(rel_pos, num_classes=self.rel_pos_onehot_size).type_as(hidden_states)
        # Project relative position(or relative position correlation information), rel_pos, into attention space.
        # 32 one-hot values of each token is projected as multiple values, each value being a raw attention score for
        # each attention head. We have 12 attention score for 12 attention heads. Each token has attention scores to
        # all other tokens. These relative position attention scores will be combined in attention layer.
        # rel_pos(batch_size, seq_len+n_patches+1, seq_len+n_patches+1, 32)
        #   -> self.rel_pos_bias(rel_pos)(batch_size, seq_len+n_patches+1, seq_len+n_patches+1, num_attention_heads=12)
        #   -> permute(0, 3, 1, 2)(batch_size, num_attention_heads=12, seq_len+n_patches+1, seq_len+n_patches+1)
        rel_pos = self.rel_pos_bias(rel_pos).permute(0, 3, 1, 2)
        rel_pos = rel_pos.contiguous()
        return rel_pos

    def _cal_2d_pos_emb(self, hidden_states, bbox):
        # hidden_states(batch_size, input_patch_len, emd_size) contains both textual embedding and visual embedding.
        # bbox(batch_size, input_patch_len, 4) contains both input bboxes and patch bboxes.

        position_coord_x = bbox[:, :, 0]  # position_coord_x(batch_size, input_patch_len) is top left x positions.
        position_coord_y = bbox[:, :, 3]  # position_coord_y(batch_size, input_patch_len) is bottom right y positions.
        # rel_pos_x_2d_mat store relative positions of neighbors for all the #input_patch_len tokens' top left x.
        rel_pos_x_2d_mat = position_coord_x.unsqueeze(-2) - position_coord_x.unsqueeze(-1)
        # rel_pos_y_2d_mat store relative positions of neighbors for all the #input_patch_len tokens' bottom right y.
        rel_pos_y_2d_mat = position_coord_y.unsqueeze(-2) - position_coord_y.unsqueeze(-1)
        # rel_pos_x/rel_pos_y(batch_size, input_patch_len, input_patch_len) with near neighbors(within 16 steps)
        # described exactly and far away neighbors(more than 16 steps) described logarithmically.
        # left neighbors are in the range of (1 ~ 31), self is 0, and right neighbors are in the range of (33~63).
        # rel_pos_x/rel_pos_y basically to describe near-by neighbour exactly and far-away neighbour briefly.
        rel_pos_x = self.relative_position_bucket(
            rel_pos_x_2d_mat,
            num_buckets=self.rel_2d_pos_bins,
            max_distance=self.max_rel_2d_pos,
        )
        rel_pos_y = self.relative_position_bucket(
            rel_pos_y_2d_mat,
            num_buckets=self.rel_2d_pos_bins,
            max_distance=self.max_rel_2d_pos,
        )
        # rel_pos_x/rel_pos_y(batch_size, input_patch_len, input_patch_len, 64) encoded as one-hot representation.
        rel_pos_x = F.one_hot(rel_pos_x, num_classes=self.rel_2d_pos_onehot_size).type_as(hidden_states)
        rel_pos_y = F.one_hot(rel_pos_y, num_classes=self.rel_2d_pos_onehot_size).type_as(hidden_states)
        # Project rel_pos_x/rel_pos_y into attention space. 64 one-hot values of each token is projected as
        # an attention score for each attention head. We have 12 attention score for 12 attention heads.
        # Each token has attention scores to all other tokens. These relative position attention scores will be
        # combined in attention layer.
        # rel_pos_x/rel_pos_y(batch_size, input_patch_len, input_patch_len, 64)
        #   -> self.rel_pos_bias(rel_pos_x/rel_pos_y)(batch_size,input_patch_len,input_patch_len,num_attention_heads=12)
        #   -> permute(0, 3, 1, 2)(batch_size, num_attention_heads=12, input_patch_len, input_patch_len)
        rel_pos_x = self.rel_pos_x_bias(rel_pos_x).permute(0, 3, 1, 2)
        rel_pos_y = self.rel_pos_y_bias(rel_pos_y).permute(0, 3, 1, 2)
        # rel_pos_x/rel_pos_y(batch, num_attention_heads=12, input_patch_len, input_patch_len)
        rel_pos_x = rel_pos_x.contiguous()
        rel_pos_y = rel_pos_y.contiguous()
        rel_2d_pos = rel_pos_x + rel_pos_y  # combine x and y.
        return rel_2d_pos

    def forward(
        self,
        hidden_states,
        bbox=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        position_ids=None,
        Hp=None,
        Wp=None
    ):
        # hidden_states(batch_size, seq_len+n_patches+1, emd_size) contains both textual embedding and visual embedding.
        # bbox(batch_size, seq_len+n_batches+1, 4) contains bounding boxes, cls_visual_box and patch boxes.
        # attention_mask(batch_size, 1, 1, seq_len+n_batches+1) contains value of 0.0 to attend
        #           and -10000(big negative value) to ignore. as these values will be added to raw value before softmax.
        # position_ids(batch_size, seq_len+n_patches+1)[[0,1,2...511,0,1, 196], ...]

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        # rel_pos(batch_size, num_attention_heads=12, seq_len+n_patches+1, seq_len+n_patches+1) is relative position
        # information or neighbour information, is calculated for input positions and patch bboxes positions.
        rel_pos = self._cal_1d_pos_emb(hidden_states, position_ids) if self.has_relative_attention_bias else None
        # rel_2d_pos(batch, num_attention_heads=12, seq_len+n_patches+1, seq_len+n_patches+1) is relative 2d position
        # information or neighbour information, is calculated for input bboxes and patch bboxes.
        rel_2d_pos = self._cal_2d_pos_emb(hidden_states, bbox) if self.has_spatial_attention_bias else None

        if self.detection:
            feat_out = {}
            j = 0

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                        # return module(*inputs, past_key_value, output_attentions, rel_pos, rel_2d_pos)
                        # The above line will cause error:
                        # RuntimeError: Trying to backward through the graph a second time
                        # (or directly access saved tensors after they have already been freed).
                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                    rel_pos,
                    rel_2d_pos
                )
            else:
                # hidden_states(batch_size, input_patch_len, emd_size) contains both textual embedding and visual embedding.
                # attention_mask(batch_size, 1, 1, seq_len+n_batches+1) contains value of 0.0 to attend
                #   and -10000(big negative value) to ignore. as these values will be added to raw value before softmax.
                # rel_pos(batch, num_attention_heads=12, input_patch_len, input_patch_len) relative position information
                #   or neighbour information. It is calculated for input positions and patch positions.
                # rel_2d_pos(batch, num_attention_heads=12, input_patch_len, input_patch_len) relative 2d position
                #   information or neighbour information. It is calculated for input bboxes and patch bboxes.
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

            if self.detection and i in self.out_indices:
                xp = hidden_states[:, -Hp*Wp:, :].permute(0, 2, 1).reshape(len(hidden_states), -1, Hp, Wp)
                feat_out[self.out_features[j]] = self.ops[j](xp.contiguous())
                j += 1

        if self.detection:
            return feat_out

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


class LayoutLMv3Model(LayoutLMv3PreTrainedModel):
    """
    """

    _keys_to_ignore_on_load_missing = [r"position_ids"]

    # Copied from transformers.models.bert.modeling_bert.BertModel.__init__ with Bert->Roberta
    def __init__(self, config, detection=False, out_features=None, image_only=False):
        super().__init__(config)
        self.config = config
        assert not config.is_decoder and not config.add_cross_attention, \
            "This version do not support decoder. Please refer to RoBERTa for implementation of is_decoder."
        self.detection = detection
        if not self.detection:
            self.image_only = False
        else:
            assert config.visual_embed
            self.image_only = image_only

        if not self.image_only:
            self.embeddings = LayoutLMv3Embeddings(config)
        self.encoder = LayoutLMv3Encoder(config, detection=detection, out_features=out_features)

        if config.visual_embed:
            embed_dim = self.config.hidden_size
            # use the default pre-training parameters for fine-tuning (e.g., input_size)
            # when the input_size is larger in fine-tuning, we will interpolate the position embedding in forward
            self.patch_embed = PatchEmbed(embed_dim=embed_dim)

            patch_size = 16
            size = int(self.config.input_size / patch_size)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_embed = nn.Parameter(torch.zeros(1, size * size + 1, embed_dim))
            self.pos_drop = nn.Dropout(p=0.)

            self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.dropout = nn.Dropout(config.hidden_dropout_prob)

            if self.config.has_relative_attention_bias or self.config.has_spatial_attention_bias:
                self._init_visual_bbox(img_size=(size, size))

            from functools import partial
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
            self.norm = norm_layer(embed_dim)

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

    def _init_visual_bbox(self, img_size=(14, 14), max_len=1000):
        visual_bbox_x = torch.div(torch.arange(0, max_len * (img_size[1] + 1), max_len),
                                  img_size[1], rounding_mode='trunc')
        visual_bbox_y = torch.div(torch.arange(0, max_len * (img_size[0] + 1), max_len),
                                  img_size[0], rounding_mode='trunc')
        visual_bbox = torch.stack(
            [
                visual_bbox_x[:-1].repeat(img_size[0], 1),
                visual_bbox_y[:-1].repeat(img_size[1], 1).transpose(0, 1),
                visual_bbox_x[1:].repeat(img_size[0], 1),
                visual_bbox_y[1:].repeat(img_size[1], 1).transpose(0, 1),
            ],
            dim=-1,
        ).view(-1, 4)

        cls_token_box = torch.tensor([[0 + 1, 0 + 1, max_len - 1, max_len - 1]])
        self.visual_bbox = torch.cat([cls_token_box, visual_bbox], dim=0)

    def _calc_visual_bbox(self, device, dtype, bsz):  # , img_size=(14, 14), max_len=1000):
        # visual_box(n_batches+1, 4) represents equal size patch boxes(left, top, right, bottom).
        # There is a visual cls_token_box is appended in the front.
        # -> visual_box(batch_size, n_batches+1, 4)
        visual_bbox = self.visual_bbox.repeat(bsz, 1, 1)
        visual_bbox = visual_bbox.to(device).type(dtype)
        return visual_bbox

    def forward_image(self, x):
        if self.detection:
            # x(batch_size, n_patches, emd_size), each patch represented by an embedding of size #emd_size.
            x = self.patch_embed(x, self.pos_embed[:, 1:, :] if self.pos_embed is not None else None)
        else:
            x = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.pos_embed is not None and self.detection:
            cls_tokens = cls_tokens + self.pos_embed[:, :1, :]
        # cls tokens embedding is appended in the front.
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None and not self.detection:
            x = x + self.pos_embed # add positional information to image embedding.
        x = self.pos_drop(x)

        x = self.norm(x)
        return x

    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    def forward(
        self,
        input_ids=None,
        bbox=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        images=None,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        use_cache = False

        # if input_ids is not None and inputs_embeds is not None:
        #     raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
            device = input_ids.device
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
            device = inputs_embeds.device
        elif images is not None:
            batch_size = len(images)
            device = images.device
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds or images")

        if not self.image_only:
            # past_key_values_length
            past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

            if attention_mask is None:
                attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
            if token_type_ids is None:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        # extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        if not self.image_only:
            if bbox is None:
                bbox = torch.zeros(tuple(list(input_shape) + [4]), dtype=torch.long, device=device)
            # embedding_output(batch_size, seq_len, emd_size) is textual embedding.
            # embedding_output = input_ids_embeddings + token_type_embeddings + position_embeddings
            #                       + spatial_position_embeddings(bbox)
            embedding_output = self.embeddings(
                input_ids=input_ids,
                bbox=bbox,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                inputs_embeds=inputs_embeds,
                past_key_values_length=past_key_values_length,
            )

        final_bbox = final_position_ids = None
        Hp = Wp = None
        if images is not None:
            patch_size = 16
            Hp, Wp = int(images.shape[2] / patch_size), int(images.shape[3] / patch_size)
            visual_emb = self.forward_image(images) # visual_emb(batch_size, n_patches+1, emd_size). visual_emb contains image patch embeddings and its positional embeddings.
            if self.detection:
                visual_attention_mask = torch.ones((batch_size, visual_emb.shape[1]), dtype=torch.long, device=device)
                if self.image_only:
                    attention_mask = visual_attention_mask
                else:
                    attention_mask = torch.cat([attention_mask, visual_attention_mask], dim=1)
            elif self.image_only:
                attention_mask = torch.ones((batch_size, visual_emb.shape[1]), dtype=torch.long, device=device)

            if self.config.has_relative_attention_bias or self.config.has_spatial_attention_bias:
                if self.config.has_spatial_attention_bias:
                    visual_bbox = self._calc_visual_bbox(device, dtype=torch.long, bsz=batch_size) # visual_box(batch_size, n_batches+1, 4)
                    if self.image_only:
                        final_bbox = visual_bbox
                    else:
                        # final_bbox(batch_size, seq_len+n_batches+1, 4) contains bounding boxes, cls_visual_box
                        # and patch boxes.
                        final_bbox = torch.cat([bbox, visual_bbox], dim=1)
                # visual_position_ids(batch_size, n_patches+1)[[0, 1, 2, ...],...]
                visual_position_ids = torch.arange(0, visual_emb.shape[1], dtype=torch.long, device=device).repeat(
                    batch_size, 1)
                if self.image_only:
                    final_position_ids = visual_position_ids
                else:
                    position_ids = torch.arange(0, input_shape[1], device=device).unsqueeze(0)
                    position_ids = position_ids.expand_as(input_ids)
                    # final_position_ids(batch_size, seq_len+n_patches+1)[[0,1,2...511,0,1, 196], ...]
                    final_position_ids = torch.cat([position_ids, visual_position_ids], dim=1)

            if self.image_only:
                embedding_output = visual_emb
            else:
                # embedding_output(batch_size, seq_len+n_patches+1, emd_size) contains both textual embedding
                # and visual embedding.
                embedding_output = torch.cat([embedding_output, visual_emb], dim=1)
            embedding_output = self.LayerNorm(embedding_output)
            embedding_output = self.dropout(embedding_output)
        elif self.config.has_relative_attention_bias or self.config.has_spatial_attention_bias:
            if self.config.has_spatial_attention_bias:
                final_bbox = bbox
            if self.config.has_relative_attention_bias:
                position_ids = self.embeddings.position_ids[:, :input_shape[1]]
                position_ids = position_ids.expand_as(input_ids)
                final_position_ids = position_ids
        # Turn 1.0 in attention_mask to 0, and 0.0 in attention_mask to -10000(big negative number),
        # so that when we add the value to softmax, it is equivalent to ignoring masked values,
        # as the value becomes so small.
        # attention_mask(batch_size, seq_len+n_patches+1) -> extended_attention_mask(batch_size, 1, 1, seq_len+n_patches+1)
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, None, device)

        encoder_outputs = self.encoder(
            embedding_output,
            bbox=final_bbox,
            position_ids=final_position_ids,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            Hp=Hp,
            Wp=Wp,
        )

        if self.detection:
            return encoder_outputs

        sequence_output = encoder_outputs[0]
        pooled_output = None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class LayoutLMv3ClassificationHead(nn.Module):
    """
    Head for sentence-level classification tasks.
    Reference: RobertaClassificationHead
    """

    def __init__(self, config, pool_feature=False):
        super().__init__()
        self.pool_feature = pool_feature
        if pool_feature:
            self.dense = nn.Linear(config.hidden_size*3, config.hidden_size)
        else:
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x):
        # x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class LayoutLMv3ForTokenClassification(LayoutLMv3PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.layoutlmv3 = LayoutLMv3Model(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if config.num_labels < 10:
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        else:
            self.classifier = LayoutLMv3ClassificationHead(config, pool_feature=False)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        bbox=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        images=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.layoutlmv3(
            input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            images=images,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
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


class LayoutLMv3ForQuestionAnswering(LayoutLMv3PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.layoutlmv3 = LayoutLMv3Model(config)
        # self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.qa_outputs = LayoutLMv3ClassificationHead(config, pool_feature=False)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        bbox=None,
        images=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.layoutlmv3(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            bbox=bbox,
            images=images,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class LayoutLMv3ForSequenceClassification(LayoutLMv3PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.layoutlmv3 = LayoutLMv3Model(config)
        self.classifier = LayoutLMv3ClassificationHead(config, pool_feature=False)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        bbox=None,
        images=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.layoutlmv3(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            bbox=bbox,
            images=images,
        )

        sequence_output = outputs[0][:, 0, :]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

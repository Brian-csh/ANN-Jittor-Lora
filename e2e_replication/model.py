# jittor version of GPT-2
# variable names are consisted with the pretrained checkpoint file

import jittor as jt
from jittor import nn

import logging
import math
import os
from collections import OrderedDict 
import copy
import math

import sys
sys.path.append('.')
from loralib import mylora_layers


# non-linearities
def gelu(x):
    return 0.5 * x * (1 + jt.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * jt.pow(x, 3))))

def swish(x):
    return x * jt.sigmoid(x)


class CrossEntropyLoss(jt.Module):
    def __init__(self, weight=None, ignore_index=None):
        self.weight = weight
        self.ignore_index = ignore_index
        
    def execute(self, output, target):
        return nn.cross_entropy_loss(output, target, self.weight, self.ignore_index, None)


# config class
class GPT2Config(object):
    def __init__(
        self,
        vocab_size_or_config_json_file=50257,
        n_positions=1024,
        n_ctx=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        lora_attn_dim=0,
        lora_attn_alpha=128,
        lora_dropout=0.0,
        lora_r_dropout=0.0,
        fix_dropout=0.0,
    ):
        self.vocab_size = vocab_size_or_config_json_file
        self.n_ctx = n_ctx
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.lora_attn_dim = lora_attn_dim
        self.lora_attn_alpha = lora_attn_alpha
        self.lora_dropout = lora_dropout
        self.lora_r_dropout = lora_r_dropout
        self.fix_dropout = fix_dropout


class Conv1D(jt.nn.Module):
    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        self.weight = jt.normal(0, 0.02, size=(nx, nf))
        self.bias = jt.zeros(nf)

    def execute(self, x):
        shape_out = x.shape[:-1] + (self.nf,)
        x = x.reshape(-1, x.shape[-1]) @ self.weight + self.bias
        x = x.reshape(shape_out)
        return x


class Attention(jt.nn.Module):
    def __init__(self, nx, n_ctx, config, scale=False):
        super().__init__()
        n_state = nx  # nx is embedding dimensions

        assert n_state % config.n_head == 0
        self.bias = jt.tril(jt.ones((n_ctx, n_ctx))).view(1, 1, n_ctx, n_ctx)
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale
        # add lora
        # print("Before entering self.c_attn in model.py: ")
        # print(f"nx: {nx}  n_state * 3: {n_state * 3}")
        self.c_attn = mylora_layers.MergedLinear(
            nx, n_state * 3, 
            r=config.lora_attn_dim, 
            lora_alpha=config.lora_attn_alpha, 
            lora_dropout=config.lora_dropout, 
            enable_lora=[True, False, True], 
            fan_in_fan_out=True,
            merge_weights=False
        )
        # print("after leaving self.c_attn=mylora...., self.weight: ")
        # print(self.c_attn.weight.size())
        self.c_proj = Conv1D(n_state, nx)
        # print("after leaving Conv1D, self.weight: ")
        # print(self.c_attn.weight.size())
        self.config = config


    # note: transpose is done before entering this function
    def _attn(self, q, k, v, len_kv=None):
        w = jt.matmul(q, k) # attention weights

        if self.scale:
            w = w / math.sqrt(v.shape[-1])

        nd, ns = w.shape[-2], w.shape[-1]
        b = self.bias[:, :, ns-nd:ns, :ns]
        w = w * b - 1e10 * (1 - b) # mask the values with negative infinity

        # q : (batch, head, q_seq_length, head_features)
        # k : (batch, head, head_features, kv_seq_length)
        # w : (batch, head, q_seq_length, kv_seq_length)
        # v : (batch, head, kv_seq_length, head_features)
        # TODO: is this syntax valid?
        if len_kv is not None:
            _len = jt.arange(k.shape[-1])
            _input_msk =  _len[None, :] >= (len_kv)[:, None]
            w = w.masked_fill(_input_msk.unsqueeze(1).unsqueeze(2), -1.0e10) 

        w = nn.softmax(w, dim=-1)
        return jt.matmul(w, v)
    

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(new_x_shape)


    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1).contiguous()  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3).contiguous()  # (batch, head, seq_length, head_features)
        
    
    def execute(self, x, history=None, layer_past=None, len_past=None):
        hidden_states = x

        x = self.c_attn(x) # attention layer, lora is expected to be integrated within the layer
        query, key, value = x.split(self.split_size, dim=2)

        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)

        #_input_msk = None

        len_kv = None

        if layer_past is not None:
            # key : (batch, head, head_features, seq_length)
            # value : (batch, head, seq_length, head_features)
            # layer_past, key : (batch, head, seq_length, head_features)
            if len_past is None:
                past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]
                key = jt.concat((past_key, key), dim=-1)
                value = jt.concat((past_value, value), dim=-2)
            else:
                key_seq = key.shape[-1]
                assert key_seq == 1

                _batch = jt.arange(0, key.shape[0], dtype=jt.int64)

                past_key, past_value = layer_past[0], layer_past[1]

                past_key[_batch,:,len_past,:] = key.squeeze(-1)
                past_value[_batch,:,len_past,:] = value.squeeze(-2)

                key = past_key.transpose(-2, -1)
                value = past_value

                len_kv = len_past + 1

        present = jt.stack((key.transpose(-2, -1), value))  # transpose to have same shapes for stacking
        a = self._attn(query, key, value, len_kv = len_kv)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        return a, present    
    

class MLP(jt.nn.Module):
    def __init__(self, n_state, config):
        super().__init__()
        nx = config.n_embd # embedding dimension
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = gelu

    def execute(self, x):
        fc_out = self.c_fc(x)
        h = self.act(fc_out)
        h2 = self.c_proj(h)
        return h2
    

class Block(jt.nn.Module):
    def __init__(self, n_ctx, config, scale=False):
        super(Block, self).__init__()
        nx = config.n_embd
        self.ln_1 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.attn = Attention(nx, n_ctx, config, scale)
        self.ln_2 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)

    def execute(self, x, layer_past=None, len_past=None):
        a, present = self.attn(self.ln_1(x), layer_past=layer_past, len_past=len_past)
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x, present
    

class GPT2Model(jt.nn.Module):
    def __init__(self, config):
        super(GPT2Model, self).__init__()
        self.n_layer = config.n_layer
        self.n_embd = config.n_embd
        self.n_vocab = config.vocab_size

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        block = Block(config.n_ctx, config, scale=True)
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.config = config


    def execute(
        self, 
        input_ids, 
        position_ids=None, 
        token_type_ids=None, 
        past=None, 
        len_past=None
    ):
        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        elif len_past is None:
            # equal size for past. []
            past_length = past[0][0].size(-2)

        if position_ids is None and len_past is None:
            position_ids = jt.arange(
                past_length, input_ids.size(-1) + past_length, 
                dtype=jt.int64
            )
            position_ids = jt.unsqueeze(position_ids, 0).expand(input_ids.shape)
        elif len_past is not None:
            position_ids = (len_past).unsqueeze(1) #.long()

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_ids.size(-1))
        position_ids = position_ids.view(-1, position_ids.size(-1))

        inputs_embeds = self.wte(input_ids)     

        position_embeds = self.wpe(position_ids)

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
            token_type_embeds = self.wte(token_type_ids)
        else:
            token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        presents = []
        for block, layer_past in zip(self.h, past):
            hidden_states, present = block(hidden_states, layer_past = layer_past, len_past=len_past)
            presents.append(present)
        hidden_states = self.ln_f(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)
        return hidden_states.view(*output_shape), presents
    

class GPT2LMHead(jt.nn.Module):
    def __init__(self, model_embeddings_weights, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.set_embeddings_weights(model_embeddings_weights)

    def set_embeddings_weights(self, model_embeddings_weights):
        embed_shape = model_embeddings_weights.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = model_embeddings_weights  # Tied weights

    def execute(self, hidden_state):
        # Truncated Language modeling logits (we remove the last token)
        # h_trunc = h[:, :-1].contiguous().view(-1, self.n_embd)
        lm_logits = self.decoder(hidden_state)
        return lm_logits
    

class GPT2LMModel(jt.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = GPT2Model(config)
        self.lm_head = GPT2LMHead(self.transformer.wte.weight, config)
        self.apply(self._init_weights)

    def set_tied(self):
        """ Make sure we are sharing the embeddings"""
        self.lm_head.set_embeddings_weights(self.transformer.wte.weight)

    def execute(
        self, 
        input_ids, 
        lm_labels=None, 
        lm_mask=None, 
        past=None, 
        len_past=None, 
        label_smooth=0.0,
        is_report_accuracy=False
    ):
        _batch, _len = input_ids.shape
        hidden_states, presents = self.transformer(input_ids, past=past, len_past=len_past)

        # batch, seq, vocab
        lm_logits = self.lm_head(hidden_states)

        if lm_labels is not None:

            if is_report_accuracy:
                _pred_token = jt.argmax(lm_logits, dim=-1)
                _hit = (_pred_token == lm_labels) * lm_mask

                _t1_acc = jt.zeros(_batch, dtype=jt.float32, device=input_ids.device) # TODO: double check datatype
                _all_acc = jt.zeros(_batch, dtype=jt.float32, device=input_ids.device)
                
                for _b in range(0, _batch):
                    for _i in range(0, _len):
                        if lm_mask[_b, _i] >= 1.0:
                            if _hit[_b, _i] > 0:
                                _t1_acc[_b] = 1.0
                            break  

                    _is_succ = True
                    for _i in range(0, _len):
                        if lm_mask[_b, _i] >= 1.0:
                            if _hit[_b, _i] <= 0:
                                _is_succ = False
                                break

                    if _is_succ:
                        _all_acc[_b] = 1.0

            if label_smooth > 0.0001:
                logprobs = nn.log_softmax(lm_logits.view(-1, lm_logits.size(-1)), dim=-1)
                nll_loss = -logprobs.gather(dim=-1, index=lm_labels.view(-1).unsqueeze(1))
                nll_loss = nll_loss.squeeze(1)
                smooth_loss = -logprobs.mean(dim=-1)
                loss = (1.0 - label_smooth) * nll_loss + label_smooth * smooth_loss
                loss = loss.view(_batch, _len)
            else:
                loss_fct = CrossEntropyLoss(ignore_index=-1)
                # import pdb; pdb.set_trace()
                loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1)).view(_batch, _len)

            if lm_mask is None:
                lm_mask = jt.ones(loss.shape, dtype=loss.dtype)
            loss = loss * lm_mask 

            loss = loss.sum() / (lm_mask.sum() + 0.0001)

            if is_report_accuracy:
                return lm_logits, loss, _t1_acc, _all_acc
            else:
                return lm_logits, loss
        return lm_logits, presents
           
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # module.weight.data.normal_(mean=0.0, std=0.02)
            jt.init.gauss_(jt.array(module.weight.data), 0.0, 0.2)
        elif isinstance(module, nn.LayerNorm):
            jt.init.zero_(jt.array(module.bias.data))
            # module.bias.data.zero_()
            jt.init.fill(jt.array(module.weight.data), 1.0)
            # module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            jt.init.zero_(jt.array(module.bias.data))
            # module.bias.data.zero_()

    def load_weight(self, state_dict):
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
    
        state_dict_tmp = copy.deepcopy(state_dict)
        old_keys = []
        new_keys = []
        for key in state_dict_tmp:
            new_key = None
            if key.endswith(".g"):
                new_key = key[:-2] + ".weight"
            elif key.endswith(".b"):
                new_key = key[:-2] + ".bias"
            elif key.endswith(".w"):
                new_key = key[:-2] + ".weight"
            
            if key.startswith("module.transformer."):
                new_key = key[len("module.transformer."):]

            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)

        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)
        
        for n, p in self.transformer.named_parameters():
            if n not in state_dict:
                state_dict[n] = p

        self.transformer.load_state_dict(state_dict)
        self.set_tied()
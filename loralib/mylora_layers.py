import jittor as jt
from jittor import nn, init
import math
from typing import Optional, List


class LoRALayer:
    def __init__(
            self,
            r: int,
            lora_alpha: int,
            lora_dropout: float,
            merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        self.merged = False
        self.merge_weights = merge_weights


class Embedding(nn.Embedding, LoRALayer):
    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            r: int = 0,
            lora_alpha: int = 1,
            merge_weights: bool = True,
            **kwargs
    ):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=0,
                           merge_weights=merge_weights)
        # lora trainable parameters
        if r > 0:
            self.lora_A = jt.nn.Parameter(jt.zeros((r, num_embeddings)))  # k = num_embeddings
            self.lora_B = jt.nn.Parameter(jt.zeros((embedding_dim, r)))  # d = embedding_dim
            self.scaling = self.lora_alpha / self.r
            self.weight.stop_grad()
        self.reset_parameters()

    def _fill_padding_idx_with_zero(self) -> None:
        if self.padding_idx is not None:
            self.weight[self.padding_idx].zero_()

    def reset_parameters(self):
        # => expected to be equivalent to 'nn.Embedding.reset_parameters(self)'
        init.gauss_(self.weight, mean=0, std=1)
        self._fill_padding_idx_with_zero()

        if hasattr(self, 'lora_A'):
            self.lora_A.assign(jt.zeros(self.lora_A.shape))
            init.gauss_(self.lora_B, mean=0, std=1)

    def train(self, mode: bool = True):
        if mode:
            self.is_train = True
            if self.merge_weights and self.merged:
                if self.r > 0:
                    self.weight.data -= (jt.matmul(self.lora_B, self.lora_A)).transpose(0, 1) * self.scaling
                self.merged = False
        else:
            self.is_train = False
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += (jt.matmul(self.lora_B, self.lora_A)).transpose(0, 1) * self.scaling
                self.merged = True

    def execute(self, x):
        # Lora is applied and not merged yet
        if self.r > 0 and not self.merged:
            result = nn.Embedding.execute(self, x)
            # conisder self.padding_idx...?
            after_A = jt.nn.embedding(x, self.lora_A.transpose(0, 1))
            result += (after_A @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        # Lora is not applied
        else:
            return nn.Embedding.execute(self, x)


class Linear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
            self,
            in_features: int,
            out_features: int,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.,
            # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
            # indicate if the weight matrix should be transposed.
            fan_in_fan_out: bool = False,
            merge_weights: bool = True,  # whether to merge Lora with the original weights during evaluating.
            **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        # print("Enter Linear, self.weight: ")
        # print(self.weight.size())
        self.fan_in_fan_out = fan_in_fan_out
        # print('self.fan_in_fan_out: ')
        # print(self.fan_in_fan_out)
        # Actual trainable parameters
        if r > 0:
            self.lora_A = jt.nn.Parameter(jt.zeros((r, in_features)))
            self.lora_B = jt.nn.Parameter(jt.zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freeze the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)
        # print("Done Initing Linear, self.weight: ")
        # print(self.weight.size())

    def reset_parameters(self):
        
        init.relu_invariant_gauss_(self.weight)
        if self.bias is not None:
            # Jittor doesn't have a direct equivalent of _calculate_fan_in_and_fan_out
            # fan_in implemented as the product of weight dimensions except the last one
            fan_in = self.weight.size().prod() / self.weight.size()[-1]
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            # Jittor uses init.uniform_ similar to PyTorch
            init.uniform_(self.bias, -bound, bound)

        if hasattr(self, 'lora_A'):
            # nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            # nn.init.zeros_(self.lora_B)
            init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            init.zero_(self.lora_B)
    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        # nn.Linear.train(self)
        if mode:
            self.is_train = True
            if self.merge_weights and self.merged:
                if self.r > 0:
                    self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
        else:
            self.is_train = False
            if self.merge_weights and not self.merged:
                if self.r > 0:
                    self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = True

    def execute(self, x):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        if self.r > 0 and not self.merged:
            result = nn.linear(x, T(self.weight), bias=self.bias)
            result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return nn.linear(x, T(self.weight), bias=self.bias)


class MergedLinear(nn.Linear, LoRALayer):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.,
            enable_lora: List[bool] = [False],
            fan_in_fan_out: bool = False,
            merge_weights: bool = True,
            **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        assert out_features % len(enable_lora) == 0, \
            'The length of enable_lora must divide out_features'
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0 and any(enable_lora):
            
            # self.lora_A = jt.nn.Parameter(jt.zeros((r * sum(enable_lora), in_features)))
            # self.lora_B = jt.nn.Parameter(jt.zeros((out_features // len(enable_lora) * sum(enable_lora), r)))
            self.lora_A = jt.zeros((r * sum(enable_lora), in_features))
            # print(self.lora_A.shape)
            self.lora_B = jt.zeros((out_features // len(enable_lora) * sum(enable_lora), r))
            # print(self.lora_B.shape)
            

            # weights for Conv1D with groups=sum(enable_lora)
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            # print(f"In MergedLinear, self.weight: \n {self.weight.size()} \n")
            # Compute the indices
            
            # self.lora_ind = self.weight.zeros((out_features,), dtype=bool).view(len(enable_lora), -1)
            self.lora_ind = jt.zeros((out_features,), dtype=jt.bool)
            # self.lora_ind = self.lora_ind.to(self.weight.device)
            self.lora_ind = self.lora_ind.view(len(enable_lora), -1)
            
            self.lora_ind[jt.array(enable_lora), :] = True
            self.lora_ind = self.lora_ind.view(-1)
            # print('after lora_ind, self.weight: ')
            # print(self.weight.size())

        self.reset_parameters()
        # print('after reset_parameters(), self.weight: ')
        # print(self.weight.size())
        if fan_in_fan_out:
            self.weight = self.weight.transpose(0, 1)
            # print(type(self.weight))
            # print(type(self.weight.data))
            # self.weight.data = self.weight.data.transpose(0, 1) 
            # print('after fan_in_fan_out, self.weight.data: ')
            # print(self.weight.shape)

    def reset_parameters(self):
        # nn.Linear.reset_parameters(self)
        
        # Not sure!
        # print('enter reset_paramters: ')
        # print(self.weight.size())
        init.relu_invariant_gauss_(self.weight)
        # print('init.relu done: ')
        # print(self.weight.size())

        if self.bias is not None:
            # Jittor doesn't have a direct equivalent of _calculate_fan_in_and_fan_out
            # fan_in implemented as the product of weight dimensions except the last one
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            # Jittor uses init.uniform_ similar to PyTorch
            init.uniform_(self.bias, -bound, bound)
        
        if hasattr(self, 'lora_A'):
            init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            init.zero_(self.lora_B)

    def zero_pad(self, x):
        # result = x.new_zeros((len(self.lora_ind), *x.shape[1:]))
        result = jt.zeros((len(self.lora_ind), *x.shape[1:]))

        result[self.lora_ind] = x
        return result

    def merge_AB(self):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        
        # input:  (mini_batch, in_channel, iW)
        # weight: (out_channels, in_channel/groups, kW)
        # By the following, we can infer in_channel = self.lora_A[0]
        # out_channels = self.lora_B[0]

        conv_layer = nn.Conv1d(in_channels=self.lora_A.shape[0],
                               out_channels=self.lora_B.shape[0],
                               kernel_size=1,
                               groups=sum(self.enable_lora),
                               bias=False)
        conv_layer.weight = self.lora_B.unsqueeze(-1)
        # print("conv_layer.weight.size(): ")
        # print(conv_layer.weight.size())
        delta_w = conv_layer(self.lora_A.unsqueeze(0)).squeeze(0)
        
        return T(self.zero_pad(delta_w))

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        
        # nn.Linear.train(self)
        if mode:
            # => Not sure
            self.is_train = True
            
            if self.merge_weights and self.merged:
                if self.r > 0 and any(self.enable_lora):
                    self.weight.data -= self.merge_AB() * self.scaling
                self.merged = False
        else:
            
            self.is_train = False
            
            if self.merge_weights and not self.merged:
                if self.r > 0 and any(self.enable_lora):
                    self.weight.data += self.merge_AB() * self.scaling
                self.merged = True

    def execute(self, x):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        if self.merged:
            return nn.linear(x, T(self.weight), bias=self.bias)
        else:
            result = nn.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                result += self.lora_dropout(x) @ T(self.merge_AB().transpose(0, 1) if self.fan_in_fan_out else self.merge_AB()) * self.scaling
            return result


class ConvLoRA(nn.Module, LoRALayer):
    def __init__(self, conv_module, in_channels, out_channels, kernel_size, r=0, lora_alpha=1, lora_dropout=0.,
                 merge_weights=True, **kwargs):
        super(ConvLoRA, self).__init__()
        self.conv = conv_module(in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        assert isinstance(kernel_size, int)
        # lora trainable parameters
        if r > 0:
            
            self.lora_A = nn.Parameter(jt.zeros((r * kernel_size, in_channels * kernel_size)))
            self.lora_B = nn.Parameter(jt.zeros((out_channels // self.conv.groups * kernel_size, r * kernel_size)))
            
            self.scaling = self.lora_alpha / self.r
            # disable gradient
            self.conv.weight.requires_grad = False
        self.reset_parameters()
        self.merged = False

    def reset_parameters(self):
        self.conv.reset_parameters()
        if hasattr(self, 'lora_A'):
            init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            init.zero_(self.lora_B)
            

    def train(self, mode=True):
        super(ConvLoRA, self).train()
        if mode:
            if self.merge_weights and self.merged:
                if self.r > 0:
                    self.conv.weight.data -= (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                if self.r > 0:
                    self.conv.weight.data += (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
                self.merged = True

    def execute(self, x):
        if self.r > 0 and not self.merged:
            return self.conv.execute(
                x,
                self.conv.weight + (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling,
                self.conv.bias
            )
        return self.conv(x)


class Conv2d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv2d, self).__init__(nn.Conv2d, *args, **kwargs)


class Conv1d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv1d, self).__init__(nn.Conv1d, *args, **kwargs)

import torch
import torch.nn as nn
import torch.nn.functional as F
from . import w8a16_gemm  # The compiled C++ extension

class Int8Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None, act_type="none"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.act_type = act_type
        self._graphed_forward = None
        self._graph_sample_shape = None
        
        # We need to register buffers for weights and scales
        # They will be populated by load_state_dict or manual assignment
        # Initial dummy values
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        # Wq: [out_features, in_features] int8
        self.register_buffer("Wq", torch.zeros((out_features, in_features), dtype=torch.int8, device=device))
        
        # scale: [out_features] float32
        self.register_buffer("scale", torch.zeros((out_features), dtype=torch.float32, device=device))
        
        if bias:
            self.register_buffer("bias", torch.zeros((out_features), dtype=dtype if dtype else torch.float32, device=device))
        else:
            self.register_buffer("bias", None)
            
        # Dummy bias for C++ binding when bias is None
        self.register_buffer("empty_bias", torch.empty(0, dtype=dtype if dtype else torch.float32, device=device))

    def enable_graph(self, sample_input):
        if sample_input.shape[-1] != self.in_features:
            raise ValueError("sample_input last dim mismatch with in_features")
        sample_input = sample_input.contiguous()
        self._graph_sample_shape = tuple(sample_input.shape)
        self._graphed_forward = torch.cuda.make_graphed_callables(self, (sample_input,))

    @classmethod
    def from_float(cls, linear_layer, act_type="none"):
        device = linear_layer.weight.device
        dtype = linear_layer.weight.dtype
        
        int8_layer = cls(linear_layer.in_features, linear_layer.out_features, 
                         bias=(linear_layer.bias is not None), device=device, dtype=dtype, act_type=act_type)
        
        W_fp = linear_layer.weight.float()
        scales = W_fp.abs().amax(dim=1) / 127.0
        scales = torch.clamp(scales, min=1e-8)
        
        W_int8 = torch.clamp((W_fp / scales[:, None]).round(), -127, 127).to(torch.int8)
        
        int8_layer.Wq = W_int8
        int8_layer.scale = scales
        if linear_layer.bias is not None:
            int8_layer.bias = linear_layer.bias.clone()
            
        return int8_layer

    def forward(self, x):
        x_shape = x.shape
        if (self._graphed_forward is not None):
            if self._graph_sample_shape is not None and tuple(x_shape) == self._graph_sample_shape:
                return self._graphed_forward(x)

        if len(x_shape) > 2:
            x = x.reshape(-1, x_shape[-1])
        x = x.contiguous()

        y = torch.empty((x.shape[0], self.out_features), device=x.device, dtype=x.dtype)

        bias_t = self.bias if self.bias is not None else self.empty_bias
        act_id = 0
        if self.act_type == "gelu":
            act_id = 1
        w8a16_gemm.forward(x, self.Wq, self.scale, bias_t, y, act_id)

        if len(x_shape) > 2:
            y = y.view(*x_shape[:-1], -1)
            
        return y

# dequant, swiglu, quant

from src.ops.base import BaseOp

class OpSwiglu(BaseOp):
    '''
    Description:
        The Swiglu operation.
        It is used to compute the Swiglu function.
    Attributes:
        m: ffn batch size.
        n: 2*moe_intermediate_size.
    '''
    def __init__(self, m, n, aichip_config, elem_size=2):
        super().__init__("Swiglu", aichip_config, elem_size)
        self.m = m
        self.n = n

    def compute_cost(self):
        # dequant: int8-->fp16
        # shape [m, n]
        dequant_flops = 4 * self.m * self.n
        # swiglu
        # Tensor w1x
        # shape [m, n/2]
        silu_flops = 6 * self.m * self.n / 2
        # quant: fp16-->int8
        # shape [m, n]
        quant_flops = 4 * self.m * self.n
        # Mul
        # (w1·x) × (w3·x)
        # [m, n/2] × [m, n/2] = [m, n/2]
        mul_flops = self.m * self.n / 2
        self.compute_flops = dequant_flops + silu_flops + quant_flops + mul_flops
        self.compute_time = self.compute_flops / self.vector_flops
        return self.compute_time

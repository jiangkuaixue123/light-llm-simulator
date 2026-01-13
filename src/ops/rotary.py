'''
ops Prototype Pattern:
torch_npu.npu_rotary_mul(Tensor x, Tensor r1, Tensor r2): -> Tensor

function:
x1, x2 = torch.chunk(x, 2, -1)
x_new = torch.cat((-x2, x1), dim=-1)
output = r1 * x + r2 * x_new

param:
Tensor x: q,k  shape:[B, N, S, D], [B, S, N, D], [S, B, N, D] dtype:bf16, fp16, fp32
Tensor r1: cos, shape:[1, 1, S, D], [1, S, 1, D], [S, 1, 1, D] dtype:bf16, fp16, fp32
Tensor r2: sin, shape:[1, 1, S, D], [1, S, 1, D], [S, 1, 1, D] dtype:bf16, fp16, fp32
'''

from src.ops.base import BaseOp

class OpRotary(BaseOp):
    '''
    Description:
        The rotary position embedding operation.
        It is used to apply the rotary position embedding to the query and key.
    Attributes:
        num_head: The number of heads.
        head_size: The head size.
    '''
    def __init__(self, name, bs, num_head, seq_len, head_size, aichip_config, elem_size=2):
        super().__init__(name, aichip_config, elem_size)
        self.bs = bs
        self.num_head = num_head
        self.seq_len = seq_len
        self.head_size = head_size

    def op_compute_dic(self):
        return 0.23

    def compute_cost(self):
        # torch.cat((-x2, x1), r1 * x, r2 * x_new, r1 * x + r2 * x_new
        self.compute_flops = 4 * self.bs * self.num_head * self.seq_len * self.head_size
        self.compute_time = self.compute_flops / self.vector_flops
        return self.compute_time

    def memory_cost(self):
        # input tensor x, output tensor, tensor cos, tensor sin
        self.bytes = (
            2 * self.bs * self.num_head * self.seq_len * self.head_size * self.elem_size +
            2 * self.seq_len * self.head_size * self.elem_size
        )
        self.memory_time = self.bytes / self.local_memory_bandwidth
        return self.memory_time

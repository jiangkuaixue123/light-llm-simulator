# Norm Cast etc.

from src.ops.base import BaseOp

class OpNorm(BaseOp):
    '''
    Description:
        The Norm operation.
        It is used to compute the Norm function.
    Attributes:
        bs: attention batch size.
    '''
    def __init__(self, bs, aichip_config, elem_size=2):
        super().__init__("Norm", aichip_config, elem_size)
        self.attn_bs = bs

    def compute_cost(self):
        self.compute_time = 60 * 1e-6
        return self.compute_time

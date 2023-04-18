from march.models.utils import *
from march.models.baseline import *


class PerfectOverfitTransformer(BaselineTransformer):
    def create_decoder_attention_mask(self, decoder_input_ids: SequenceInputIds) -> TensorType["N", "L_out"]:
        no_loss_mask: TensorType["N", "L_out"] = decoder_input_ids == -100
        return no_loss_mask

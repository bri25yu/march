from march.models.big_heads import BigHeadsTransformer, BigHeadsTransformerConfig
from march.models.baseline import BaselineTransformer, TransformerConfig 

# config = BigHeadsTransformerConfig(dim_model=306,head_scale_size=4,feedforward_scale=2)
# model = BigHeadsTransformer(config)
# print(f"{model.count_parameters():,}")


# config = BigHeadsTransformerConfig(dim_model=190,head_scale_size=8,feedforward_scale=2,dim_w_o_output_scaling=2)
# model = BigHeadsTransformer(config)
# print(f"{model.count_parameters():,}")


config = TransformerConfig()
model = BaselineTransformer(config)
print(f"{model.count_parameters():,}")

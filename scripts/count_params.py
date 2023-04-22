from march.models.big_heads import BigHeadsTransformer, BigHeadsTransformerConfig

config = BigHeadsTransformerConfig(dim_model=380,dim_qkv=190,num_heads=8)
model = BigHeadsTransformer(config)
print(f"{model.count_parameters():,}")


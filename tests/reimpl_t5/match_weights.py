from itertools import product

from torch import equal


__all__ = ["assert_property_equal", "assert_weight_equal", "assert_grad_equal"]


class ModelComponents:
    class Embedding:
        reimpl = lambda m: m.embedding
        t5 = lambda m: m.lm_head

    class SelfAttention:
        @staticmethod
        def reimpl(model, num_layer):
            self_attn = model.self_attention_layers[num_layer]
            weights = (self_attn.w_q, self_attn.w_k, self_attn.w_v, self_attn.w_o)

            if num_layer == 0:
                weights = (*weights, self_attn.relative_attention_bias)

            return weights

        @staticmethod
        def t5(model, num_layer):
            self_attn = model.block[num_layer].layer[0].SelfAttention
            weights = (self_attn.q, self_attn.k, self_attn.v, self_attn.o)

            if num_layer == 0:
                weights = (*weights, self_attn.relative_attention_bias)

            return weights

    class CrossAttention:
        @staticmethod
        def reimpl(model, num_layer):
            cross_attn = model.cross_attention_layers[num_layer]
            return (cross_attn.w_q, cross_attn.w_k, cross_attn.w_v, cross_attn.w_o)

        @staticmethod
        def t5(model, num_layer):
            cross_attn = model.block[num_layer].layer[1].EncDecAttention
            return (cross_attn.q, cross_attn.k, cross_attn.v, cross_attn.o)

    class FeedForward:
        reimpl = lambda m, i: (m.feedforward_layers[i].up_projection, m.feedforward_layers[i].down_projection)
        t5 = lambda m, i: (m.block[i].layer[-1].DenseReluDense.wi, m.block[i].layer[-1].DenseReluDense.wo)

    class LayerNorm:
        reimpl = lambda m: tuple(m.layernorms)

        @staticmethod
        def t5(model):
            num_encdec_layers = model.config.num_layers

            res = []
            for i in range(num_encdec_layers):
                block_i = model.block[i]
                for layer in block_i.layer:
                    res.append(layer.layer_norm)

            res.append(model.final_layer_norm)

            return tuple(res)


def get_components(model, model_type: str, component: str, layer_num: int=None, enc_or_dec: str=None):
    get_weight = getattr(getattr(ModelComponents, component), model_type)
    if component in "Embedding":
        output_weights = get_weight(model)
    elif component == "LayerNorm":
        model = getattr(model, enc_or_dec)
        output_weights = get_weight(model)
    else:
        model = getattr(model, enc_or_dec)
        output_weights = get_weight(model, layer_num)

    if isinstance(output_weights, tuple):
        return tuple(w.weight for w in output_weights)
    else:
        return (output_weights.weight,)


def get_matched_weights(reimpl_model, t5_model):
    num_encdec_layers = reimpl_model.config.num_layers // 2
    enc_or_dec_options = ["encoder", "decoder"]
    encdec_layer_nums = list(reversed(list(range(num_encdec_layers))))
    batch_kwargs = [
        {"component": "Embedding"},
        *[
            {"component": "SelfAttention", "layer_num": i, "enc_or_dec": enc_or_dec}
            for i, enc_or_dec in product(encdec_layer_nums, enc_or_dec_options)
        ],
        *[
            {"component": "CrossAttention", "layer_num": i, "enc_or_dec": "decoder"}
            for i in encdec_layer_nums
        ],
        *[
            {"component": "FeedForward", "layer_num": i, "enc_or_dec": enc_or_dec}
            for i, enc_or_dec in product(encdec_layer_nums, enc_or_dec_options)
        ],
        *[{"component": "LayerNorm", "enc_or_dec": enc_or_dec} for enc_or_dec in enc_or_dec_options],
    ]

    res = []
    for kwargs in batch_kwargs:
        reimpl_components = get_components(reimpl_model, "reimpl", **kwargs)
        t5_components = get_components(t5_model, "t5", **kwargs)
        assert len(reimpl_components) == len(t5_components), (kwargs, len(reimpl_components), len(t5_components))
        res.extend(zip([kwargs] * len(reimpl_components), reimpl_components, t5_components))

    return res


def assert_property_equal(reimpl_model, t5_model, property_name: str) -> int:
    matched_weights = get_matched_weights(reimpl_model, t5_model)
    num_parameters_matched = 0

    error_strs = set()
    for kwargs, reimpl_weight, t5_weight in matched_weights:
        reimpl_property = getattr(reimpl_weight, property_name)
        t5_property = getattr(t5_weight, property_name)

        num_parameters_matched += reimpl_weight.numel()

        if property_name == "grad" and reimpl_property is None and t5_property is None: continue

        # At least one but not both are None
        property_mismatch = (reimpl_property is None) ^ (t5_property is None)
        assert not property_mismatch, f"Property mismatch for {kwargs}\nReimpl\n{reimpl_property}\nT5\n{t5_property}"

        weights_equal = equal(reimpl_property, t5_property)
        if not weights_equal:
            error_strs.add(str(kwargs))

    assert not error_strs, f"Values not equal for property {property_name} for:\n\t" + "\n\t".join(error_strs)

    return num_parameters_matched


def assert_weight_equal(reimpl_model, t5_model) -> None:
    assert_property_equal(reimpl_model, t5_model, "data")


def assert_grad_equal(reimpl_model, t5_model) -> None:
    assert_property_equal(reimpl_model, t5_model, "grad")

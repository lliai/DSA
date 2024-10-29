import torch
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP
import math
import numpy as np


# layer_func: first_last_biased
def layer_func_first_last_biased(n_layers, target_sparsity):
    first_layer_sparsity = 0.2  # Lower sparsity for the first layer
    last_layer_sparsity = 0.2  # Lower sparsity for the last layer

    # Calculate the remaining sparsity for the intermediate layers
    middle_layer_sparsity = (
        n_layers * target_sparsity - first_layer_sparsity - last_layer_sparsity
    ) / (n_layers - 2)

    all_layer_ratio = (
        [1 - first_layer_sparsity]
        + [1 - middle_layer_sparsity] * (n_layers - 2)
        + [1 - last_layer_sparsity]
    )

    print(all_layer_ratio)
    print(sum(all_layer_ratio))

    # allclose
    assert math.isclose(
        round(sum(all_layer_ratio) / n_layers, 9), target_sparsity
    ), f"Sparsity mismatch! {round(sum(all_layer_ratio) / n_layers, 9)} != {target_sparsity}"

    return all_layer_ratio


# def layer_func_cos(n_layers, target_sparsity):
#     # Calculate the layer ratios using the cosine function
#     layer_ratios = [
#         0.5 * (1 + math.cos(i / n_layers * math.pi)) for i in range(n_layers)
#     ]

#     # Normalize the layer ratios to sum up to 1
#     total_ratio = sum(layer_ratios)
#     layer_ratios = [ratio / total_ratio for ratio in layer_ratios]

#     # Scale the layer ratios to match the target sparsity
#     all_layer_ratio = [ratio * target_sparsity for ratio in layer_ratios]

#     # Assert the sparsity matches the target
#     assert math.isclose(
#         round(sum(all_layer_ratio) / n_layers, 9), target_sparsity
#     ), f"Sparsity mismatch! {round(sum(all_layer_ratio) / 0.8, 9)} != {target_sparsity}"

#     # Assert each item in all_layer_ratio is between 0 and 1
#     for item in all_layer_ratio:
#         assert item < 1, f"Sparsity mismatch! {item} > 1"
#         assert item > 0, f"Sparsity mismatch! {item} < 0"

#     return all_layer_ratio


def layer_func_cos(
    n_layers,
    target_sparsity,
    max_sparsity=0.9,
    min_sparsity=0.2,
    lamda=0.08,
    shift=0.2,
):
    # Calculate the layer ratios using the cosine function
    layer_ratios = [
        0.5 * (1 + math.cos((i / (n_layers - 1) * math.pi) + shift))
        for i in range(n_layers)
    ]

    # Adjust the layer ratios to fit between min_sparsity and max_sparsity
    all_layer_ratio = [
        min(max(ratio, min_sparsity), max_sparsity) for ratio in layer_ratios
    ]

    all_layer_ratio = np.array(all_layer_ratio)
    all_layer_ratio = (all_layer_ratio - all_layer_ratio.min()) * (
        1 / (all_layer_ratio.max() - all_layer_ratio.min()) * lamda * 2
    )

    # Scale the layer ratios to match the target sparsity
    all_layer_ratio = all_layer_ratio - all_layer_ratio.mean() + target_sparsity

    print(all_layer_ratio)
    # Assert the sparsity matches the target
    assert math.isclose(
        round(sum(all_layer_ratio) / n_layers, 9), target_sparsity
    ), f"Sparsity mismatch! {round(sum(all_layer_ratio) / n_layers, 9)} != {target_sparsity}"

    # Assert each item in all_layer_ratio is between 0 and 1
    for item in all_layer_ratio:
        assert item <= 1, f"Sparsity mismatch! {item} > 1"
        assert item >= 0, f"Sparsity mismatch! {item} < 0"

    return all_layer_ratio


# layer_func: layer_type_biased
# def layer_func_layer_type_biased(layers, target_sparsity):
#     n_layers = len(layers)
#     all_layer_ratio = []

#     for layer in layers:
#         if isinstance(layer, LlamaMLP):
#             # Lower sparsity for convolutional layers
#             all_layer_ratio.append(1 - 0.2)
#         elif isinstance(layer, LlamaAttention):
#             # Higher sparsity for attention layers
#             all_layer_ratio.append(1 - 0.7)
#         else:
#             # Default sparsity for other layer types
#             all_layer_ratio.append(1 - target_sparsity)

#     # Normalize the ratios to match the target sparsity
#     sum_ratios = sum(all_layer_ratio)
#     all_layer_ratio = [
#         ratio * target_sparsity / sum_ratios for ratio in all_layer_ratio
#     ]

#     print(all_layer_ratio)
#     print(sum(all_layer_ratio))
#     assert (
#         sum(all_layer_ratio) // n_layers == target_sparsity
#     ), f"Sparsity mismatch! {sum(all_layer_ratio) / n_layers} != {target_sparsity}"

#     return all_layer_ratio


# # layer_func: magnitude_biased
# def layer_func_magnitude_biased(layers, target_sparsity):
#     n_layers = len(layers)
#     all_layer_ratio = []

#     for layer in layers:
#         weights = layer.weight.data.abs()
#         sorted_weights, _ = torch.sort(weights.flatten(), descending=True)
#         cumsum_weights = torch.cumsum(sorted_weights, dim=0)
#         target_threshold = cumsum_weights[-1] * (1 - target_sparsity)

#         layer_sparsity_ratio = (
#             1 - (cumsum_weights >= target_threshold).sum() / weights.numel()
#         )
#         all_layer_ratio.append(layer_sparsity_ratio)

#     return all_layer_ratio


_name_to_layer_func = {
    # "first_last_biased": layer_func_first_last_biased,
    # "layer_type_biased": layer_func_layer_type_biased,
    # "magnitude_biased": layer_func_magnitude_biased,
    "cos": layer_func_cos,
}


if __name__ == "__main__":
    n_layers = 5
    target_sparsity = 0.8
    result = layer_func_cos(n_layers, target_sparsity)
    print("Layer ratios:", result)

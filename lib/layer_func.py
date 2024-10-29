import torch
import math
import numpy as np
import random


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


def layer_func_symmetry(n_layers, target_sparsity, step=0.05):
    first_half = n_layers // 2

    all_sparsity_ratio = [target_sparsity for _ in range(n_layers)]

    delta_ = []

    for i in range(first_half):
        delta_.append(random.uniform(0, step))

    for i in range(first_half):
        all_sparsity_ratio[i] += delta_[i]

    # shuffle the all_sparsity_ratio list
    random.shuffle(delta_)

    for i in range(first_half + n_layers % 2, n_layers):
        all_sparsity_ratio[i] -= delta_[i - first_half - n_layers % 2]

    print(all_sparsity_ratio)
    # Assert the sparsity matches the target
    assert math.isclose(
        round(sum(all_sparsity_ratio) / n_layers, 9), target_sparsity
    ), f"Sparsity mismatch! {round(sum(all_sparsity_ratio) / n_layers, 9)} != {target_sparsity}"

    return all_sparsity_ratio


if __name__ == "__main__":
    n_layers = 11
    target_sparsity = 0.8
    # result = layer_func_cos(n_layers, target_sparsity)
    result = layer_func_symmetry(n_layers, target_sparsity)
    print("Layer ratios:", result)

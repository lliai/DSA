import math
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os

# def layer_func_cos(n_layers, target_sparsity, max_sparsity=0.8, min_sparsity=0.2):
#     # Calculate the layer ratios using the cosine function
#     layer_ratios = [
#         0.5 * (1 + math.cos(i / n_layers * math.pi)) for i in range(n_layers)
#     ]

#     # Normalize the layer ratios to sum up to 1
#     total_ratio = sum(layer_ratios)
#     layer_ratios = [ratio / total_ratio for ratio in layer_ratios]

#     # Adjust the layer ratios to achieve the target sparsity
#     adjusted_ratios = [ratio * target_sparsity for ratio in layer_ratios]

#     # Assert the sparsity matches the target
#     assert math.isclose(
#         round(sum(adjusted_ratios), 9), target_sparsity
#     ), f"Sparsity mismatch! {round(sum(adjusted_ratios), 9)} != {target_sparsity}"

#     # Assert each item in adjusted_ratios is between 0 and 1
#     for item in adjusted_ratios:
#         assert item <= max_sparsity, f"Sparsity mismatch! {item} > {max_sparsity}"
#         assert item >= min_sparsity, f"Sparsity mismatch! {item} < {min_sparsity}"

#     return adjusted_ratios


# # Example usage
# n_layers = 20
# target_sparsity = 0.8
# result = layer_func_cos(n_layers, target_sparsity)
# print("Layer ratios:", result)
# import matplotlib.pyplot as plt

# plt.plot(result)
# plt.savefig("test2.png")

import math

import math
import numpy as np


def layer_func_cos(
    n_layers,
    target_sparsity,
    max_sparsity=0.9,
    min_sparsity=0.2,
    lamda=0.08,
    phase_shift=0.2,
):
    # Calculate the layer ratios using the cosine function
    layer_ratios = [
        0.5 * (1 + math.cos((i / (n_layers - 1) * math.pi) + phase_shift))
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

    # print(all_layer_ratio)
    # Assert the sparsity matches the target
    assert math.isclose(
        round(sum(all_layer_ratio) / n_layers, 9), target_sparsity
    ), f"Sparsity mismatch! {round(sum(all_layer_ratio) / n_layers, 9)} != {target_sparsity}"

    # Assert each item in all_layer_ratio is between 0 and 1
    for item in all_layer_ratio:
        assert item <= 1, f"Sparsity mismatch! {item} > 1"
        assert item >= 0, f"Sparsity mismatch! {item} < 0"

    return all_layer_ratio


# Example usage
n_layers = 30
target_sparsity = 0.5
# result = layer_func_cos(n_layers, target_sparsity)
# print("Layer ratios:", result)

# Create a list to store the filenames of the saved plots
filenames = []
min_sparsity, max_sparsity = 9999, -1

for shift in [
    0,
    0.2,
    0.4,
    0.6,
    0.8,
    1.0,
    1.2,
    1.4,
    1.6,
    1.8,
    2.0,
    2.2,
    2.4,
    2.6,
    2.8,
    3.0,
    3.2,
    3.4,
    3.6,
    3.8,
    4.0,
    4.2,
    4.4,
    4.6,
    4.8,
    5.0,
    5.2,
    5.4,
    5.6,
    5.8,
    6.0,
    6.2,
    6.4,
    6.6,
    6.8,
    7.0,
]:
    result = layer_func_cos(n_layers, target_sparsity, phase_shift=shift, lamda=0.12)
    # print("Layer ratios:", result)

    if shift == 0.2 or shift == 6.4:
        if min_sparsity > min(result):
            min_sparsity = min(result)

        if max_sparsity < max(result):
            max_sparsity = max(result)

    # Create a new figure for each plot
    plt.figure()
    plt.plot(result)
    plt.title(f"Phase Shift: {shift}")
    plt.xlabel("Layer")
    plt.ylabel("Sparsity Ratio")

    # Save the plot as a file
    filename = f"test2_{shift}.png"
    plt.savefig(filename)
    plt.close()

    # Append the filename to the list
    filenames.append(filename)

print(min_sparsity, max_sparsity)

# Create a GIF using the saved plots
images = [imageio.imread(filename) for filename in filenames]
imageio.mimsave("test2_0.4.gif", images, duration=800)  # Adjust the duration as needed

# Clean up the temporary plot files
for filename in filenames:
    os.remove(filename)

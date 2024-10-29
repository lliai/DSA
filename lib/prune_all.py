import time
import heapq
import torch
import torch.nn as nn
from .sparsegpt import SparseGPT
from .layerwrapper import WrappedGPT
from .data import get_loaders
import numpy as np
from pdb import set_trace as st
from collections import defaultdict
from .layer_func import layer_func_cos, layer_func_symmetry
from .evo_lib import evolution_for_gene, post_adjust_list_mean
from .ompq_lib import ompq_process


def prepare_calibration_input_opt(model, dataloader, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    if "OPT" in model.__class__.__name__:
        layers = model.model.decoder.layers

    else:
        layers = model.model.layers

    # dev = model.hf_device_map["model.embed_tokens"]
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (128, model.seqlen, model.config.hidden_size), dtype=dtype, device=device
    )
    inps.requires_grad = False
    cache = {
        "i": 0,
        "attention_mask": None,
    }

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    model.config.use_cache = use_cache

    position_ids = None

    return inps, outs, attention_mask, position_ids


def find_layers(module, layers=[nn.Linear], name=""):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(
                child, layers=layers, name=name + "." + name1 if name != "" else name1
            )
        )
    return res


def check_sparsity(model):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers
    count = 0
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W == 0).sum().item()
            total_params += W.numel()

            sub_count += (W == 0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache
    return float(count) / total_params


def check_sparsity_mask(mask):

    W = mask
    count = 0
    total_params = 0
    count += (W != 0).sum().item()
    total_params += W.numel()

    print(f" density {float(count)/total_params:.6f}")


def check_outlier(mask, threshold):

    W = mask
    count = 0
    total_params = 0

    max_shred = torch.max(W) * threshold
    count += (W > max_shred).sum().item()
    total_params += W.numel()

    outlier_ratio = float(count) / total_params * 100

    return outlier_ratio


def plot_outlier(distribution: list, threshold: float, plot_name: str):
    import matplotlib.pyplot as plt

    plt.hist(distribution, bins=100)
    plt.axvline(x=threshold, color="r", linestyle="--")
    plt.tight_layout()
    plt.savefig(plot_name)


def check_outlier_mean(mask, threshold, name=""):
    W = mask
    count = 0
    total_params = 0

    max_shred = torch.mean(W) * threshold
    count += (W > max_shred).sum().item()
    total_params += W.numel()

    # plot_outlier(W.cpu().numpy(), max_shred, f"outlier_distribution_{name}.png")

    outlier_ratio = float(count) / total_params * 100

    return outlier_ratio


def prepare_calibration_input(model, dataloader, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # dev = model.hf_device_map["model.embed_tokens"]
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (128, model.seqlen, model.config.hidden_size), dtype=dtype, device=device
    )
    inps.requires_grad = False
    cache = {"i": 0, "attention_mask": None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids


def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1, 1))
    thres = torch.gather(
        sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True) - 1
    )
    W_mask = W_metric <= thres
    cur_sparsity = (W_mask == True).sum() / W_mask.numel()
    return W_mask, cur_sparsity


def prune_magnitude(
    args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0
):

    print("model", model)

    layers = model.model.layers

    print(layers)
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data
            W_metric = torch.abs(W)
            if prune_n != 0:
                W_mask = torch.zeros_like(W) == 1
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii : (ii + prune_m)].float()
                        W_mask.scatter_(
                            1,
                            ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1],
                            True,
                        )
            else:
                thresh = torch.sort(W_metric.flatten().cuda())[0][
                    int(W.numel() * args.sparsity_ratio)
                ].cpu()
                W_mask = W_metric <= thresh

            W[W_mask] = 0


def prune_wanda_outlier_structure_special(
    args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0
):
    ##### calucalte outlier ratio
    all_layer_ratio = []  # step1: record the layer outlier ratio
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibdation data")
    dataloader, _ = get_loaders(
        "c4", nsamples=args.nsamples, seed=args.seed, seqlen=2048, tokenizer=tokenizer
    )
    print("dataset loading complete")
    with torch.no_grad():
        if "OPT" in model.__class__.__name__:
            print("Experiments with OPT models")
            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(
                model, dataloader, device
            )
        else:
            inps, outs, attention_mask, position_ids = prepare_calibration_input(
                model, dataloader, device
            )

    if "opt" in args.model:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if (
            f"model.layers.{i}" in model.hf_device_map
        ):  ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = (
                inps.to(dev),
                outs.to(dev),
                attention_mask.to(dev),
                position_ids.to(dev),
            )

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(
                        inps[j].unsqueeze(0), attention_mask=attention_mask
                    )[0]
                else:
                    outs[j] = layer(
                        inps[j].unsqueeze(0),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                    )[0]
        for h in handles:
            h.remove()
        layer_wmetric = []  # step2: record the layer weight metric

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                wrapped_layers[name].scaler_row.reshape((1, -1))
            )
            # activation_data = torch.sqrt(
            #     wrapped_layers[name].scaler_row.reshape((1, -1))
            # )
            # layer_wmetric.append(activation_data)

            layer_wmetric.append(W_metric)  # step2

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(
                        inps[j].unsqueeze(0), attention_mask=attention_mask
                    )[0]
                else:
                    outs[j] = layer(
                        inps[j].unsqueeze(0),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                    )[0]
        inps, outs = outs, inps

        # step2
        layer_wmetric = torch.cat([torch.flatten(x.cpu()) for x in layer_wmetric])

        # step3: calculate the outlier ratio
        for out_ratio in [args.Hyper_m]:
            out_ratio_layer = check_outlier_mean(layer_wmetric, out_ratio)  # step3
            print("layer outlier ratio", out_ratio, out_ratio_layer)

        all_layer_ratio.append(out_ratio_layer)

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

    # step 4: adjust the outlier ratio by z-scaling
    print("before adjustment", all_layer_ratio)

    all_layer_ratio = np.array(all_layer_ratio)
    all_layer_ratio = (all_layer_ratio - all_layer_ratio.min()) * (
        1
        / (all_layer_ratio.max() - all_layer_ratio.min())
        * args.Lamda  # 0.08 by default
    )
    all_layer_ratio = all_layer_ratio - np.mean(all_layer_ratio)

    all_layer_ratio = np.round(all_layer_ratio)

    # Here is the key step to adjust the outlier ratio with N:M
    for i in range(len(all_layer_ratio)):
        if all_layer_ratio[i] == 1.0:
            all_layer_ratio[i] = 2.0

    all_layer_ratio = prune_n - all_layer_ratio

    print("after adjustment", all_layer_ratio)
    ############## prune
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibdation data")
    dataloader, _ = get_loaders(
        "c4", nsamples=args.nsamples, seed=args.seed, seqlen=2048, tokenizer=tokenizer
    )
    print("dataset loading complete")
    with torch.no_grad():
        if "OPT" in model.__class__.__name__:
            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(
                model, dataloader, device
            )
        else:
            inps, outs, attention_mask, position_ids = prepare_calibration_input(
                model, dataloader, device
            )

    # print ("inps",inps)
    if "opt" in args.model:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        if (
            f"model.layers.{i}" in model.hf_device_map
        ):  ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = (
                inps.to(dev),
                outs.to(dev),
                attention_mask.to(dev),
                position_ids.to(dev),
            )

        prune_n = int(all_layer_ratio[i])
        print("Layer {} prune_n {} prune_m {}".format(i, prune_n, prune_m))

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(
                        inps[j].unsqueeze(0), attention_mask=attention_mask
                    )[0]
                else:
                    outs[j] = layer(
                        inps[j].unsqueeze(0),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                    )[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                wrapped_layers[name].scaler_row.reshape((1, -1))
            )
            activation_data = torch.sqrt(
                wrapped_layers[name].scaler_row.reshape((1, -1))
            )
            layer_sparsity_ratio = 1 - all_layer_ratio[i]
            if layer_sparsity_ratio <= 0:
                layer_sparsity_ratio = 0.01

            W_mask = (
                torch.zeros_like(W_metric) == 1
            )  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii : (ii + prune_m)].float()
                        W_mask.scatter_(
                            1,
                            ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1],
                            True,
                        )

            # print ("W_mask",W_mask)
            subset[name].weight.data[W_mask] = 0  ## set weights to zero

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(
                        inps[j].unsqueeze(0), attention_mask=attention_mask
                    )[0]
                else:
                    outs[j] = layer(
                        inps[j].unsqueeze(0),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                    )[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


def prune_wanda(
    args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0
):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibdation data")
    dataloader, _ = get_loaders(
        "c4", nsamples=args.nsamples, seed=args.seed, seqlen=2048, tokenizer=tokenizer
    )
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(
            model, dataloader, device
        )

    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if (
            f"model.layers.{i}" in model.hf_device_map
        ):  ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = (
                inps.to(dev),
                outs.to(dev),
                attention_mask.to(dev),
                position_ids.to(dev),
            )

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                wrapped_layers[name].scaler_row.reshape((1, -1))
            )

            W_mask = (
                torch.zeros_like(W_metric) == 1
            )  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii : (ii + prune_m)].float()
                        W_mask.scatter_(
                            1,
                            ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1],
                            True,
                        )
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:
                    # wanda variant
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0.0, 0.8]
                    W_mask, cur_sparsity = return_given_alpha(
                        alpha, sort_res, W_metric, tmp_metric, sum_before
                    )
                    while (torch.abs(cur_sparsity - args.sparsity_ratio) > 0.001) and (
                        alpha_hist[1] - alpha_hist[0] >= 0.001
                    ):
                        if cur_sparsity > args.sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new
                        W_mask, cur_sparsity = return_given_alpha(
                            alpha, sort_res, W_metric, tmp_metric, sum_before
                        )
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured pruning
                    indices = sort_res[1][
                        :, : int(W_metric.shape[1] * args.sparsity_ratio)
                    ]
                    W_mask.scatter_(1, indices, True)
            #             print ("W_mask",W_mask)
            subset[name].weight.data[W_mask] = 0  ## set weights to zero

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


def prune_mag_outlier(
    args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0
):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    ##### calucalte outlier ratio

    all_layer_ratio = []
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibdation data")
    dataloader, _ = get_loaders(
        "c4", nsamples=args.nsamples, seed=args.seed, seqlen=2048, tokenizer=tokenizer
    )
    print("dataset loading complete")
    with torch.no_grad():

        if "OPT" in model.__class__.__name__:

            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(
                model, dataloader, device
            )
        else:

            inps, outs, attention_mask, position_ids = prepare_calibration_input(
                model, dataloader, device
            )

    if "opt" in args.model:
        layers = model.model.decoder.layers

    else:
        layers = model.model.layers

    for i in range(len(layers)):
        layer = layers[i]

        subset = find_layers(layer)

        if (
            f"model.layers.{i}" in model.hf_device_map
        ):  ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = (
                inps.to(dev),
                outs.to(dev),
                attention_mask.to(dev),
                position_ids.to(dev),
            )

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(
                        inps[j].unsqueeze(0), attention_mask=attention_mask
                    )[0]
                else:
                    outs[j] = layer(
                        inps[j].unsqueeze(0),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                    )[0]
        for h in handles:
            h.remove()

        layer_wmetric = []

        for name in subset:

            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                wrapped_layers[name].scaler_row.reshape((1, -1))
            )

            activation_data = torch.sqrt(
                wrapped_layers[name].scaler_row.reshape((1, -1))
            )

            layer_wmetric.append(W_metric)

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(
                        inps[j].unsqueeze(0), attention_mask=attention_mask
                    )[0]
                else:
                    outs[j] = layer(
                        inps[j].unsqueeze(0),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                    )[0]
        inps, outs = outs, inps

        layer_wmetric = torch.cat([torch.flatten(x.cpu()) for x in layer_wmetric])

        for out_ratio in [args.Hyper_m]:

            out_ratio_layer = check_outlier_mean(layer_wmetric, out_ratio)
            print("layer outlier ratio", out_ratio, out_ratio_layer)

        all_layer_ratio.append(out_ratio_layer)

    print("before adjustment", all_layer_ratio)

    all_layer_ratio = np.array(all_layer_ratio)

    all_layer_ratio = (all_layer_ratio - all_layer_ratio.min()) * (
        1 / (all_layer_ratio.max() - all_layer_ratio.min()) * args.Lamda * 2
    )

    all_layer_ratio = (
        all_layer_ratio - np.mean(all_layer_ratio) + (1 - args.sparsity_ratio)
    )

    print(
        all_layer_ratio,
        np.mean(all_layer_ratio),
        np.max(all_layer_ratio),
        np.min(all_layer_ratio),
    )

    print("after adjustment", all_layer_ratio)

    ############## prune

    if "opt" in args.model:
        layers = model.model.decoder.layers

    else:
        layers = model.model.layers

    print(layers)
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:

            layer_sparsity_ratio = 1 - all_layer_ratio[i]
            W = subset[name].weight.data
            W_metric = torch.abs(W)
            if prune_n != 0:
                W_mask = torch.zeros_like(W) == 1
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii : (ii + prune_m)].float()
                        W_mask.scatter_(
                            1,
                            ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1],
                            True,
                        )
            else:
                thresh = torch.sort(W_metric.flatten().cuda())[0][
                    int(W.numel() * layer_sparsity_ratio)
                ].cpu()
                W_mask = W_metric <= thresh

            W[W_mask] = 0


def prune_wanda_outlier_structure(
    args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0
):
    ##### calucalte outlier ratio
    all_layer_ratio = []
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibdation data")
    dataloader, _ = get_loaders(
        "c4", nsamples=args.nsamples, seed=args.seed, seqlen=2048, tokenizer=tokenizer
    )
    print("dataset loading complete")
    with torch.no_grad():
        if "OPT" in model.__class__.__name__:
            print("Experiments with OPT models")
            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(
                model, dataloader, device
            )
        else:
            inps, outs, attention_mask, position_ids = prepare_calibration_input(
                model, dataloader, device
            )

    if "opt" in args.model:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers

    for i in range(len(layers)):
        layer = layers[i]

        subset = find_layers(layer)
        if (
            f"model.layers.{i}" in model.hf_device_map
        ):  ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = (
                inps.to(dev),
                outs.to(dev),
                attention_mask.to(dev),
                position_ids.to(dev),
            )

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(
                        inps[j].unsqueeze(0), attention_mask=attention_mask
                    )[0]
                else:
                    outs[j] = layer(
                        inps[j].unsqueeze(0),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                    )[0]
        for h in handles:
            h.remove()
        layer_wmetric = []

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                wrapped_layers[name].scaler_row.reshape((1, -1))
            )
            activation_data = torch.sqrt(
                wrapped_layers[name].scaler_row.reshape((1, -1))
            )
            # layer_wmetric.append(activation_data)

            layer_wmetric.append(W_metric)

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(
                        inps[j].unsqueeze(0), attention_mask=attention_mask
                    )[0]
                else:
                    outs[j] = layer(
                        inps[j].unsqueeze(0),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                    )[0]
        inps, outs = outs, inps

        layer_wmetric = torch.cat([torch.flatten(x.cpu()) for x in layer_wmetric])

        for out_ratio in [args.Hyper_m]:
            out_ratio_layer = check_outlier_mean(layer_wmetric, out_ratio)
            print("layer outlier ratio", out_ratio, out_ratio_layer)

        all_layer_ratio.append(out_ratio_layer)

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

    print("before adjustment", all_layer_ratio)

    all_layer_ratio = np.array(all_layer_ratio)
    all_layer_ratio = (all_layer_ratio - all_layer_ratio.min()) * (
        1 / (all_layer_ratio.max() - all_layer_ratio.min()) * args.Lamda
    )
    all_layer_ratio = all_layer_ratio - np.mean(all_layer_ratio)

    all_layer_ratio = np.round(all_layer_ratio)

    all_layer_ratio = prune_n - all_layer_ratio

    print("after adjustment", all_layer_ratio)
    ############## prune
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibdation data")
    dataloader, _ = get_loaders(
        "c4", nsamples=args.nsamples, seed=args.seed, seqlen=2048, tokenizer=tokenizer
    )
    print("dataset loading complete")
    with torch.no_grad():
        if "OPT" in model.__class__.__name__:
            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(
                model, dataloader, device
            )
        else:
            inps, outs, attention_mask, position_ids = prepare_calibration_input(
                model, dataloader, device
            )

    # print ("inps",inps)
    if "opt" in args.model:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        if (
            f"model.layers.{i}" in model.hf_device_map
        ):  ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = (
                inps.to(dev),
                outs.to(dev),
                attention_mask.to(dev),
                position_ids.to(dev),
            )

        prune_n = int(all_layer_ratio[i])
        print("Layer {} prune_n {} prune_m {}".format(i, prune_n, prune_m))

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(
                        inps[j].unsqueeze(0), attention_mask=attention_mask
                    )[0]
                else:
                    outs[j] = layer(
                        inps[j].unsqueeze(0),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                    )[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                wrapped_layers[name].scaler_row.reshape((1, -1))
            )
            activation_data = torch.sqrt(
                wrapped_layers[name].scaler_row.reshape((1, -1))
            )
            layer_sparsity_ratio = 1 - all_layer_ratio[i]
            if layer_sparsity_ratio <= 0:
                layer_sparsity_ratio = 0.01

            W_mask = (
                torch.zeros_like(W_metric) == 1
            )  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii : (ii + prune_m)].float()
                        W_mask.scatter_(
                            1,
                            ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1],
                            True,
                        )

            # print ("W_mask",W_mask)
            subset[name].weight.data[W_mask] = 0  ## set weights to zero

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(
                        inps[j].unsqueeze(0), attention_mask=attention_mask
                    )[0]
                else:
                    outs[j] = layer(
                        inps[j].unsqueeze(0),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                    )[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


def prune_wanda_outlier(
    args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0
):
    ##### calucalte outlier ratio

    all_layer_ratio = []
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibdation data")
    dataloader, _ = get_loaders(
        "c4", nsamples=args.nsamples, seed=args.seed, seqlen=2048, tokenizer=tokenizer
    )
    print("dataset loading complete")
    with torch.no_grad():

        if "OPT" in model.__class__.__name__:

            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(
                model, dataloader, device
            )
        else:

            inps, outs, attention_mask, position_ids = prepare_calibration_input(
                model, dataloader, device
            )

    if "opt" in args.model:
        layers = model.model.decoder.layers

    else:
        layers = model.model.layers

    for i in range(len(layers)):
        layer = layers[i]

        subset = find_layers(layer)

        if (
            f"model.layers.{i}" in model.hf_device_map
        ):  ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = (
                inps.to(dev),
                outs.to(dev),
                attention_mask.to(dev),
                position_ids.to(dev),
            )

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(
                        inps[j].unsqueeze(0), attention_mask=attention_mask
                    )[0]
                else:
                    outs[j] = layer(
                        inps[j].unsqueeze(0),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                    )[0]
        for h in handles:
            h.remove()

        layer_wmetric = []

        for name in subset:

            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                wrapped_layers[name].scaler_row.reshape((1, -1))
            )
            # W = subset[name].weight.data
            # part1 = torch.abs(W) / torch.sum(torch.abs(W), dim=0) + torch.abs(
            #     W
            # ) / torch.sum(torch.abs(W), dim=1).reshape(-1, 1)
            # part2 = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
            # part2 = part2**0.5
            # W_metric = part1 * part2

            activation_data = torch.sqrt(
                wrapped_layers[name].scaler_row.reshape((1, -1))
            )
            layer_wmetric.append(W_metric)

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(
                        inps[j].unsqueeze(0), attention_mask=attention_mask
                    )[0]
                else:
                    outs[j] = layer(
                        inps[j].unsqueeze(0),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                    )[0]
        inps, outs = outs, inps

        layer_wmetric = torch.cat([torch.flatten(x.cpu()) for x in layer_wmetric])

        for out_ratio in [args.Hyper_m]:
            out_ratio_layer = check_outlier_mean(
                layer_wmetric, out_ratio, name=f"layer_{i}"
            )
            print("layer outlier ratio", out_ratio, out_ratio_layer)

        all_layer_ratio.append(out_ratio_layer)

    print("before adjustment", all_layer_ratio)

    all_layer_ratio = np.array(all_layer_ratio)

    all_layer_ratio = (all_layer_ratio - all_layer_ratio.min()) * (
        1 / (all_layer_ratio.max() - all_layer_ratio.min()) * args.Lamda * 2
    )

    all_layer_ratio = (
        all_layer_ratio - np.mean(all_layer_ratio) + np.exp(1 - args.sparsity_ratio)
    )

    print(
        all_layer_ratio,
        np.mean(all_layer_ratio),
        np.max(all_layer_ratio),
        np.min(all_layer_ratio),
    )

    print("after adjustment", all_layer_ratio)

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    ############## prune

    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibdation data")
    dataloader, _ = get_loaders(
        "c4", nsamples=args.nsamples, seed=args.seed, seqlen=2048, tokenizer=tokenizer
    )
    print("dataset loading complete")
    with torch.no_grad():

        if "OPT" in model.__class__.__name__:

            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(
                model, dataloader, device
            )
        else:

            inps, outs, attention_mask, position_ids = prepare_calibration_input(
                model, dataloader, device
            )

    if "opt" in args.model:
        layers = model.model.decoder.layers

    else:
        layers = model.model.layers

    for i in range(len(layers)):
        layer = layers[i]

        subset = find_layers(layer)

        if (
            f"model.layers.{i}" in model.hf_device_map
        ):  ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = (
                inps.to(dev),
                outs.to(dev),
                attention_mask.to(dev),
                position_ids.to(dev),
            )

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(
                        inps[j].unsqueeze(0), attention_mask=attention_mask
                    )[0]
                else:
                    outs[j] = layer(
                        inps[j].unsqueeze(0),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                    )[0]
        for h in handles:
            h.remove()

        for name in subset:

            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                wrapped_layers[name].scaler_row.reshape((1, -1))
            )
            # W = subset[name].weight.data
            # part1 = torch.abs(W) / torch.sum(torch.abs(W), dim=0) + torch.abs(
            #     W
            # ) / torch.sum(torch.abs(W), dim=1).reshape(-1, 1)
            # part2 = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
            # part2 = part2**0.5
            # W_metric = part1 * part2

            activation_data = torch.sqrt(
                wrapped_layers[name].scaler_row.reshape((1, -1))
            )

            layer_sparsity_ratio = 1 - all_layer_ratio[i]

            if layer_sparsity_ratio <= 0:
                layer_sparsity_ratio = 0.01

            W_mask = (
                torch.zeros_like(W_metric) == 1
            )  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii : (ii + prune_m)].float()
                        W_mask.scatter_(
                            1,
                            ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1],
                            True,
                        )
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:
                    # wanda variant
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0.0, 0.8]
                    W_mask, cur_sparsity = return_given_alpha(
                        alpha, sort_res, W_metric, tmp_metric, sum_before
                    )
                    while (torch.abs(cur_sparsity - layer_sparsity_ratio) > 0.001) and (
                        alpha_hist[1] - alpha_hist[0] >= 0.001
                    ):
                        if cur_sparsity > layer_sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new
                        W_mask, cur_sparsity = return_given_alpha(
                            alpha, sort_res, W_metric, tmp_metric, sum_before
                        )
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured pruning
                    indices = sort_res[1][
                        :, : int(W_metric.shape[1] * layer_sparsity_ratio)
                    ]
                    W_mask.scatter_(1, indices, True)
            #             print ("W_mask",W_mask)
            subset[name].weight.data[W_mask] = 0  ## set weights to zero

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(
                        inps[j].unsqueeze(0), attention_mask=attention_mask
                    )[0]
                else:
                    outs[j] = layer(
                        inps[j].unsqueeze(0),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                    )[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


@torch.no_grad()
def prune_sparsegpt(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print("Starting ...")
    dataloader, _ = get_loaders(
        "c4", nsamples=args.nsamples, seed=args.seed, seqlen=2048, tokenizer=tokenizer
    )

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]

    print("Ready.")

    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = (
                inps.to(dev),
                outs.to(dev),
                attention_mask.to(dev),
                position_ids.to(dev),
            )

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(
                inps[j].unsqueeze(0),
                attention_mask=attention_mask,
                position_ids=position_ids,
            )[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print("Pruning ...")

            gpts[name].fasterprune(
                args.sparsity_ratio,
                prune_n=prune_n,
                prune_m=prune_m,
                percdamp=0.01,
                blocksize=128,
            )
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(
                inps[j].unsqueeze(0),
                attention_mask=attention_mask,
                position_ids=position_ids,
            )[0]

        layers[i] = layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


@torch.no_grad()
def prune_sparsegpt_outlier(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    ##### calucalte outlier ratio

    device = dev

    all_layer_ratio = []
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibdation data")
    dataloader, _ = get_loaders(
        "c4", nsamples=args.nsamples, seed=args.seed, seqlen=2048, tokenizer=tokenizer
    )
    print("dataset loading complete")
    with torch.no_grad():

        if "OPT" in model.__class__.__name__:

            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(
                model, dataloader, device
            )
        else:

            inps, outs, attention_mask, position_ids = prepare_calibration_input(
                model, dataloader, device
            )

    if "opt" in args.model:
        layers = model.model.decoder.layers

    else:
        layers = model.model.layers

    for i in range(len(layers)):
        layer = layers[i]

        subset = find_layers(layer)

        if (
            f"model.layers.{i}" in model.hf_device_map
        ):  ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = (
                inps.to(dev),
                outs.to(dev),
                attention_mask.to(dev),
                position_ids.to(dev),
            )

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(
                        inps[j].unsqueeze(0), attention_mask=attention_mask
                    )[0]
                else:
                    outs[j] = layer(
                        inps[j].unsqueeze(0),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                    )[0]
        for h in handles:
            h.remove()

        layer_wmetric = []

        for name in subset:

            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                wrapped_layers[name].scaler_row.reshape((1, -1))
            )

            activation_data = torch.sqrt(
                wrapped_layers[name].scaler_row.reshape((1, -1))
            )

            layer_wmetric.append(W_metric)

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(
                        inps[j].unsqueeze(0), attention_mask=attention_mask
                    )[0]
                else:
                    outs[j] = layer(
                        inps[j].unsqueeze(0),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                    )[0]
        inps, outs = outs, inps

        layer_wmetric = torch.cat([torch.flatten(x.cpu()) for x in layer_wmetric])

        for out_ratio in [args.Hyper_m]:

            out_ratio_layer = check_outlier_mean(layer_wmetric, out_ratio)
            print("layer outlier ratio", out_ratio, out_ratio_layer)

        all_layer_ratio.append(out_ratio_layer)

    print("before adjustment", all_layer_ratio)

    all_layer_ratio = np.array(all_layer_ratio)

    all_layer_ratio = (all_layer_ratio - all_layer_ratio.min()) * (
        1 / (all_layer_ratio.max() - all_layer_ratio.min()) * args.Lamda * 2
    )

    all_layer_ratio = (
        all_layer_ratio - np.mean(all_layer_ratio) + (1 - args.sparsity_ratio)
    )

    print(
        all_layer_ratio,
        np.mean(all_layer_ratio),
        np.max(all_layer_ratio),
        np.min(all_layer_ratio),
    )

    print("after adjustment", all_layer_ratio)

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    ############## prune
    print("Starting ...")

    dataloader, _ = get_loaders(
        "c4", nsamples=args.nsamples, seed=args.seed, seqlen=2048, tokenizer=tokenizer
    )

    use_cache = model.config.use_cache
    model.config.use_cache = False
    if "opt" in args.model:
        layers = model.model.decoder.layers

    else:
        layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]

    print("Ready.")

    for i in range(len(layers)):

        layer_sparsity_ratio = 1 - all_layer_ratio[i]

        if layer_sparsity_ratio <= 0:
            layer_sparsity_ratio = 0.01

        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = (
                inps.to(dev),
                outs.to(dev),
                attention_mask.to(dev),
                position_ids.to(dev),
            )
            # inps, outs, attention_mask = inps.to(dev), outs.to(dev), attention_mask.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(
                inps[j].unsqueeze(0),
                attention_mask=attention_mask,
                position_ids=position_ids,
            )[0]
            # outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print("Pruning ...")

            gpts[name].fasterprune(
                layer_sparsity_ratio,
                prune_n=prune_n,
                prune_m=prune_m,
                percdamp=0.01,
                blocksize=128,
            )
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(
                inps[j].unsqueeze(0),
                attention_mask=attention_mask,
                position_ids=position_ids,
            )[0]
            # outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


# 这个方法是为了用来测试和探索一些人工设置的规则是否有效果：
# 1. 设置不同layer ratio的范围
def test_prune_wanda_outlier(
    args,
    model,
    tokenizer,
    device=torch.device("cuda:0"),
    prune_n=0,
    prune_m=0,
    layer_func=None,
    shift=0,
    lamda=0.08,
    step=0.05,
):
    ##### calucalte outlier ratio
    assert layer_func is not None

    # all_layer_ratio = []
    if "OPT" in model.__class__.__name__:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers

    # Here is the naive baseline that uniformly set the layer ratio to be the same
    if layer_func == "uniform":
        all_layer_ratio = [args.sparsity_ratio for _ in range(len(layers))]
    elif layer_func == "cos":
        all_layer_ratio = layer_func_cos(
            len(layers), target_sparsity=args.sparsity_ratio, shift=shift, lamda=lamda
        )
    elif layer_func == "sym":
        all_layer_ratio = layer_func_symmetry(
            len(layers), target_sparsity=args.sparsity_ratio, step=step
        )
    else:
        raise NotImplementedError

    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("after adjustment", all_layer_ratio)

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    ############## prune

    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibdation data")
    dataloader, _ = get_loaders(
        "c4", nsamples=args.nsamples, seed=args.seed, seqlen=2048, tokenizer=tokenizer
    )
    print("dataset loading complete")
    with torch.no_grad():

        if "OPT" in model.__class__.__name__:

            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(
                model, dataloader, device
            )
        else:

            inps, outs, attention_mask, position_ids = prepare_calibration_input(
                model, dataloader, device
            )

    if "opt" in args.model:
        layers = model.model.decoder.layers

    else:
        layers = model.model.layers

    for i in range(len(layers)):
        layer = layers[i]

        subset = find_layers(layer)

        if (
            f"model.layers.{i}" in model.hf_device_map
        ):  ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = (
                inps.to(dev),
                outs.to(dev),
                attention_mask.to(dev),
                position_ids.to(dev),
            )

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(
                        inps[j].unsqueeze(0), attention_mask=attention_mask
                    )[0]
                else:
                    outs[j] = layer(
                        inps[j].unsqueeze(0),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                    )[0]
        for h in handles:
            h.remove()

        for name in subset:

            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                wrapped_layers[name].scaler_row.reshape((1, -1))
            )
            # W = subset[name].weight.data
            # part1 = torch.abs(W) / torch.sum(torch.abs(W), dim=0) + torch.abs(
            #     W
            # ) / torch.sum(torch.abs(W), dim=1).reshape(-1, 1)
            # part2 = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
            # part2 = part2**0.5
            # W_metric = part1 * part2

            activation_data = torch.sqrt(
                wrapped_layers[name].scaler_row.reshape((1, -1))
            )

            layer_sparsity_ratio = 1 - all_layer_ratio[i]

            if layer_sparsity_ratio <= 0:
                layer_sparsity_ratio = 0.01

            W_mask = (
                torch.zeros_like(W_metric) == 1
            )  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii : (ii + prune_m)].float()
                        W_mask.scatter_(
                            1,
                            ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1],
                            True,
                        )
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:
                    # wanda variant
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0.0, 0.8]
                    W_mask, cur_sparsity = return_given_alpha(
                        alpha, sort_res, W_metric, tmp_metric, sum_before
                    )
                    while (torch.abs(cur_sparsity - layer_sparsity_ratio) > 0.001) and (
                        alpha_hist[1] - alpha_hist[0] >= 0.001
                    ):
                        if cur_sparsity > layer_sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new
                        W_mask, cur_sparsity = return_given_alpha(
                            alpha, sort_res, W_metric, tmp_metric, sum_before
                        )
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured pruning
                    indices = sort_res[1][
                        :, : int(W_metric.shape[1] * layer_sparsity_ratio)
                    ]
                    W_mask.scatter_(1, indices, True)
            #             print ("W_mask",W_mask)
            subset[name].weight.data[W_mask] = 0  ## set weights to zero

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(
                        inps[j].unsqueeze(0), attention_mask=attention_mask
                    )[0]
                else:
                    outs[j] = layer(
                        inps[j].unsqueeze(0),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                    )[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


#wanda_owl_var
# wanda outlier but don't use outlier metric use var 
def test_prune_wanda_outlier_use_var(
    args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0, 
    lengine=None, 
):
    ##### calucalte outlier ratio

    all_layer_ratio = []
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibdation data")
    dataloader, _ = get_loaders(
        "c4", nsamples=args.nsamples, seed=args.seed, seqlen=2048, tokenizer=tokenizer
    )
    print("dataset loading complete")
    with torch.no_grad():

        if "OPT" in model.__class__.__name__:

            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(
                model, dataloader, device
            )
        else:

            inps, outs, attention_mask, position_ids = prepare_calibration_input(
                model, dataloader, device
            )

    if "opt" in args.model:
        layers = model.model.decoder.layers

    else:
        layers = model.model.layers

    for i in range(len(layers)):
        layer = layers[i]

        subset = find_layers(layer)

        if (
            f"model.layers.{i}" in model.hf_device_map
        ):  ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = (
                inps.to(dev),
                outs.to(dev),
                attention_mask.to(dev),
                position_ids.to(dev),
            )

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(
                        inps[j].unsqueeze(0), attention_mask=attention_mask
                    )[0]
                else:
                    outs[j] = layer(
                        inps[j].unsqueeze(0),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                    )[0]
        for h in handles:
            h.remove()

        layer_wmetric = []

        for name in subset:

            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                wrapped_layers[name].scaler_row.reshape((1, -1))
            )
            # W = subset[name].weight.data
            # part1 = torch.abs(W) / torch.sum(torch.abs(W), dim=0) + torch.abs(
            #     W
            # ) / torch.sum(torch.abs(W), dim=1).reshape(-1, 1)
            # part2 = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
            # part2 = part2**0.5
            # W_metric = part1 * part2

            # activation_data = torch.sqrt(
            #     wrapped_layers[name].scaler_row.reshape((1, -1))
            # )
            layer_wmetric.append(W_metric)

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(
                        inps[j].unsqueeze(0), attention_mask=attention_mask
                    )[0]
                else:
                    outs[j] = layer(
                        inps[j].unsqueeze(0),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                    )[0]
        inps, outs = outs, inps

        layer_wmetric = torch.cat([torch.flatten(x.cpu()) for x in layer_wmetric])

        # This part is the key of computing the outlier 
        # for out_ratio in [args.Hyper_m]:
        #     out_ratio_layer = check_outlier_mean(
        #         layer_wmetric, out_ratio, name=f"layer_{i}"
        #     )
        #     print("layer outlier ratio", out_ratio, out_ratio_layer)

        out_ratio_layer = lengine.compute_importance(layer_wmetric)

        all_layer_ratio.append(out_ratio_layer)

    # 0-1 scale 
    all_layer_ratio = np.array(all_layer_ratio)
    all_layer_ratio = (all_layer_ratio - all_layer_ratio.min()) / (all_layer_ratio.max() - all_layer_ratio.min())

    all_layer_ratio = [1-item for item in all_layer_ratio]
    all_layer_ratio = np.array(all_layer_ratio)

    print("before adjustment", all_layer_ratio)

    all_layer_ratio = (all_layer_ratio - all_layer_ratio.min()) * (
        1 / (all_layer_ratio.max() - all_layer_ratio.min()) * args.Lamda * 2
    )

    all_layer_ratio = (
        all_layer_ratio - np.mean(all_layer_ratio) + (1 - args.sparsity_ratio)
    )

    print(
        all_layer_ratio,
        np.mean(all_layer_ratio),
        np.max(all_layer_ratio),
        np.min(all_layer_ratio),
    )

    print("after adjustment", all_layer_ratio)

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    ############## prune

    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibdation data")
    dataloader, _ = get_loaders(
        "c4", nsamples=args.nsamples, seed=args.seed, seqlen=2048, tokenizer=tokenizer
    )
    print("dataset loading complete")
    with torch.no_grad():

        if "OPT" in model.__class__.__name__:

            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(
                model, dataloader, device
            )
        else:

            inps, outs, attention_mask, position_ids = prepare_calibration_input(
                model, dataloader, device
            )

    if "opt" in args.model:
        layers = model.model.decoder.layers

    else:
        layers = model.model.layers

    for i in range(len(layers)):
        layer = layers[i]

        subset = find_layers(layer)

        if (
            f"model.layers.{i}" in model.hf_device_map
        ):  ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = (
                inps.to(dev),
                outs.to(dev),
                attention_mask.to(dev),
                position_ids.to(dev),
            )

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(
                        inps[j].unsqueeze(0), attention_mask=attention_mask
                    )[0]
                else:
                    outs[j] = layer(
                        inps[j].unsqueeze(0),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                    )[0]
        for h in handles:
            h.remove()

        for name in subset:

            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                wrapped_layers[name].scaler_row.reshape((1, -1))
            )
            # W = subset[name].weight.data
            # part1 = torch.abs(W) / torch.sum(torch.abs(W), dim=0) + torch.abs(
            #     W
            # ) / torch.sum(torch.abs(W), dim=1).reshape(-1, 1)
            # part2 = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
            # part2 = part2**0.5
            # W_metric = part1 * part2

            activation_data = torch.sqrt(
                wrapped_layers[name].scaler_row.reshape((1, -1))
            )

            layer_sparsity_ratio = 1 - all_layer_ratio[i]

            if layer_sparsity_ratio <= 0:
                layer_sparsity_ratio = 0.01

            W_mask = (
                torch.zeros_like(W_metric) == 1
            )  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii : (ii + prune_m)].float()
                        W_mask.scatter_(
                            1,
                            ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1],
                            True,
                        )
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:
                    # wanda variant
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0.0, 0.8]
                    W_mask, cur_sparsity = return_given_alpha(
                        alpha, sort_res, W_metric, tmp_metric, sum_before
                    )
                    while (torch.abs(cur_sparsity - layer_sparsity_ratio) > 0.001) and (
                        alpha_hist[1] - alpha_hist[0] >= 0.001
                    ):
                        if cur_sparsity > layer_sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new
                        W_mask, cur_sparsity = return_given_alpha(
                            alpha, sort_res, W_metric, tmp_metric, sum_before
                        )
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured pruning
                    indices = sort_res[1][
                        :, : int(W_metric.shape[1] * layer_sparsity_ratio)
                    ]
                    W_mask.scatter_(1, indices, True)
            #             print ("W_mask",W_mask)
            subset[name].weight.data[W_mask] = 0  ## set weights to zero

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(
                        inps[j].unsqueeze(0), attention_mask=attention_mask
                    )[0]
                else:
                    outs[j] = layer(
                        inps[j].unsqueeze(0),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                    )[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()



# wanda_owl_evo
def test_prune_wanda_outlier_use_evo(
    args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0, 
    lengine=None, 
):
    ##### calucalte outlier ratio

    all_layer_ratio = []
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibdation data")
    dataloader, _ = get_loaders(
        "c4", nsamples=args.nsamples, seed=args.seed, seqlen=2048, tokenizer=tokenizer
    )
    print("dataset loading complete")
    with torch.no_grad():

        if "OPT" in model.__class__.__name__:

            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(
                model, dataloader, device
            )
        else:

            inps, outs, attention_mask, position_ids = prepare_calibration_input(
                model, dataloader, device
            )

    if "opt" in args.model:
        layers = model.model.decoder.layers

    else:
        layers = model.model.layers

    for i in range(len(layers)):
        layer = layers[i]

        subset = find_layers(layer)

        if (
            f"model.layers.{i}" in model.hf_device_map
        ):  ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = (
                inps.to(dev),
                outs.to(dev),
                attention_mask.to(dev),
                position_ids.to(dev),
            )

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(
                        inps[j].unsqueeze(0), attention_mask=attention_mask
                    )[0]
                else:
                    outs[j] = layer(
                        inps[j].unsqueeze(0),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                    )[0]
        for h in handles:
            h.remove()

        layer_wmetric = []

        for name in subset:

            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                wrapped_layers[name].scaler_row.reshape((1, -1))
            )
            # W = subset[name].weight.data
            # part1 = torch.abs(W) / torch.sum(torch.abs(W), dim=0) + torch.abs(
            #     W
            # ) / torch.sum(torch.abs(W), dim=1).reshape(-1, 1)
            # part2 = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
            # part2 = part2**0.5
            # W_metric = part1 * part2

            # activation_data = torch.sqrt(
            #     wrapped_layers[name].scaler_row.reshape((1, -1))
            # )
            layer_wmetric.append(W_metric)

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(
                        inps[j].unsqueeze(0), attention_mask=attention_mask
                    )[0]
                else:
                    outs[j] = layer(
                        inps[j].unsqueeze(0),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                    )[0]
        inps, outs = outs, inps

        layer_wmetric = torch.cat([torch.flatten(x.cpu()) for x in layer_wmetric])

        # This part is the key of computing the outlier 
        for out_ratio in [args.Hyper_m]:
            out_ratio_layer = check_outlier_mean(
                layer_wmetric, out_ratio, name=f"layer_{i}"
            )
            print("layer outlier ratio", out_ratio, out_ratio_layer)

        # out_ratio_layer = lengine.compute_importance(layer_wmetric)

        all_layer_ratio.append(out_ratio_layer)

    # 0-1 scale 
    # all_layer_ratio = np.array(all_layer_ratio)
    # all_layer_ratio = (all_layer_ratio - all_layer_ratio.min()) / (all_layer_ratio.max() - all_layer_ratio.min())

    # all_layer_ratio = [1-item for item in all_layer_ratio]
    # all_layer_ratio = np.array(all_layer_ratio)

    # print("before adjustment", all_layer_ratio)

    # all_layer_ratio = (all_layer_ratio - all_layer_ratio.min()) * (
    #     1 / (all_layer_ratio.max() - all_layer_ratio.min()) * args.Lamda * 2
    # )

    # all_layer_ratio = (
    #     all_layer_ratio - np.mean(all_layer_ratio) + (1 - args.sparsity_ratio)
    # )

    all_layer_ratio = evolution_for_gene(all_layer_ratio, args)


    all_layer_ratio,_,_ = post_adjust_list_mean(all_layer_ratio, args.sparsity_ratio)


    print(
        all_layer_ratio,
        np.mean(all_layer_ratio),
        np.max(all_layer_ratio),
        np.min(all_layer_ratio),
    )

    print("after adjustment", all_layer_ratio)

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    ############## prune

    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibdation data")
    dataloader, _ = get_loaders(
        "c4", nsamples=args.nsamples, seed=args.seed, seqlen=2048, tokenizer=tokenizer
    )
    print("dataset loading complete")
    with torch.no_grad():

        if "OPT" in model.__class__.__name__:

            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(
                model, dataloader, device
            )
        else:

            inps, outs, attention_mask, position_ids = prepare_calibration_input(
                model, dataloader, device
            )

    if "opt" in args.model:
        layers = model.model.decoder.layers

    else:
        layers = model.model.layers

    for i in range(len(layers)):
        layer = layers[i]

        subset = find_layers(layer)

        if (
            f"model.layers.{i}" in model.hf_device_map
        ):  ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = (
                inps.to(dev),
                outs.to(dev),
                attention_mask.to(dev),
                position_ids.to(dev),
            )

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(
                        inps[j].unsqueeze(0), attention_mask=attention_mask
                    )[0]
                else:
                    outs[j] = layer(
                        inps[j].unsqueeze(0),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                    )[0]
        for h in handles:
            h.remove()

        for name in subset:

            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                wrapped_layers[name].scaler_row.reshape((1, -1))
            )
            # W = subset[name].weight.data
            # part1 = torch.abs(W) / torch.sum(torch.abs(W), dim=0) + torch.abs(
            #     W
            # ) / torch.sum(torch.abs(W), dim=1).reshape(-1, 1)
            # part2 = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
            # part2 = part2**0.5
            # W_metric = part1 * part2

            activation_data = torch.sqrt(
                wrapped_layers[name].scaler_row.reshape((1, -1))
            )

            layer_sparsity_ratio = 1 - all_layer_ratio[i]

            if layer_sparsity_ratio <= 0:
                layer_sparsity_ratio = 0.01

            W_mask = (
                torch.zeros_like(W_metric) == 1
            )  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii : (ii + prune_m)].float()
                        W_mask.scatter_(
                            1,
                            ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1],
                            True,
                        )
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:
                    # wanda variant
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0.0, 0.8]
                    W_mask, cur_sparsity = return_given_alpha(
                        alpha, sort_res, W_metric, tmp_metric, sum_before
                    )
                    while (torch.abs(cur_sparsity - layer_sparsity_ratio) > 0.001) and (
                        alpha_hist[1] - alpha_hist[0] >= 0.001
                    ):
                        if cur_sparsity > layer_sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new
                        W_mask, cur_sparsity = return_given_alpha(
                            alpha, sort_res, W_metric, tmp_metric, sum_before
                        )
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured pruning
                    indices = sort_res[1][
                        :, : int(W_metric.shape[1] * layer_sparsity_ratio)
                    ]
                    W_mask.scatter_(1, indices, True)
            #             print ("W_mask",W_mask)
            subset[name].weight.data[W_mask] = 0  ## set weights to zero

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(
                        inps[j].unsqueeze(0), attention_mask=attention_mask
                    )[0]
                else:
                    outs[j] = layer(
                        inps[j].unsqueeze(0),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                    )[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()




# wanda_owl_ompq
def test_prune_wanda_outlier_use_ompq(
    args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0, 
    lengine=None, 
):
    ##### calucalte outlier ratio

    all_layer_ratio = []
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibdation data")
    dataloader, _ = get_loaders(
        "c4", nsamples=args.nsamples, seed=args.seed, seqlen=2048, tokenizer=tokenizer
    )
    print("dataset loading complete")
    with torch.no_grad():

        if "OPT" in model.__class__.__name__:

            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(
                model, dataloader, device
            )
        else:

            inps, outs, attention_mask, position_ids = prepare_calibration_input(
                model, dataloader, device
            )

    if "opt" in args.model:
        layers = model.model.decoder.layers

    else:
        layers = model.model.layers

    for i in range(len(layers)):
        layer = layers[i]

        subset = find_layers(layer)

        if (
            f"model.layers.{i}" in model.hf_device_map
        ):  ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = (
                inps.to(dev),
                outs.to(dev),
                attention_mask.to(dev),
                position_ids.to(dev),
            )

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(
                        inps[j].unsqueeze(0), attention_mask=attention_mask
                    )[0]
                else:
                    outs[j] = layer(
                        inps[j].unsqueeze(0),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                    )[0]
        for h in handles:
            h.remove()

        layer_wmetric = []

        for name in subset:

            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                wrapped_layers[name].scaler_row.reshape((1, -1))
            )
            # W = subset[name].weight.data
            # part1 = torch.abs(W) / torch.sum(torch.abs(W), dim=0) + torch.abs(
            #     W
            # ) / torch.sum(torch.abs(W), dim=1).reshape(-1, 1)
            # part2 = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
            # part2 = part2**0.5
            # W_metric = part1 * part2

            # activation_data = torch.sqrt(
            #     wrapped_layers[name].scaler_row.reshape((1, -1))
            # )
            layer_wmetric.append(W_metric)

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(
                        inps[j].unsqueeze(0), attention_mask=attention_mask
                    )[0]
                else:
                    outs[j] = layer(
                        inps[j].unsqueeze(0),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                    )[0]
        inps, outs = outs, inps

        layer_wmetric = torch.cat([torch.flatten(x.cpu()) for x in layer_wmetric])

        # This part is the key of computing the outlier 
        for out_ratio in [args.Hyper_m]:
            out_ratio_layer = check_outlier_mean(
                layer_wmetric, out_ratio, name=f"layer_{i}"
            )
            print("layer outlier ratio", out_ratio, out_ratio_layer)

        # out_ratio_layer = lengine.compute_importance(layer_wmetric)

        all_layer_ratio.append(out_ratio_layer)

    # 0-1 scale 
    # all_layer_ratio = np.array(all_layer_ratio)
    # all_layer_ratio = (all_layer_ratio - all_layer_ratio.min()) / (all_layer_ratio.max() - all_layer_ratio.min())

    # all_layer_ratio = [1-item for item in all_layer_ratio]
    # all_layer_ratio = np.array(all_layer_ratio)

    # print("before adjustment", all_layer_ratio)

    # all_layer_ratio = (all_layer_ratio - all_layer_ratio.min()) * (
    #     1 / (all_layer_ratio.max() - all_layer_ratio.min()) * args.Lamda * 2
    # )

    # all_layer_ratio = (
    #     all_layer_ratio - np.mean(all_layer_ratio) + (1 - args.sparsity_ratio)
    # )

    # all_layer_ratio = evolution_for_gene(all_layer_ratio, args)


    # all_layer_ratio,_,_ = post_adjust_list_mean(all_layer_ratio, args.sparsity_ratio)

    all_layer_ratio = ompq_process(all_layer_ratio, args)

    print(
        all_layer_ratio,
        np.mean(all_layer_ratio),
        np.max(all_layer_ratio),
        np.min(all_layer_ratio),
    )

    print("after adjustment", all_layer_ratio)

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    ############## prune

    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibdation data")
    dataloader, _ = get_loaders(
        "c4", nsamples=args.nsamples, seed=args.seed, seqlen=2048, tokenizer=tokenizer
    )
    print("dataset loading complete")
    with torch.no_grad():

        if "OPT" in model.__class__.__name__:

            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(
                model, dataloader, device
            )
        else:

            inps, outs, attention_mask, position_ids = prepare_calibration_input(
                model, dataloader, device
            )

    if "opt" in args.model:
        layers = model.model.decoder.layers

    else:
        layers = model.model.layers

    for i in range(len(layers)):
        layer = layers[i]

        subset = find_layers(layer)

        if (
            f"model.layers.{i}" in model.hf_device_map
        ):  ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = (
                inps.to(dev),
                outs.to(dev),
                attention_mask.to(dev),
                position_ids.to(dev),
            )

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(
                        inps[j].unsqueeze(0), attention_mask=attention_mask
                    )[0]
                else:
                    outs[j] = layer(
                        inps[j].unsqueeze(0),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                    )[0]
        for h in handles:
            h.remove()

        for name in subset:

            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                wrapped_layers[name].scaler_row.reshape((1, -1))
            )
            # W = subset[name].weight.data
            # part1 = torch.abs(W) / torch.sum(torch.abs(W), dim=0) + torch.abs(
            #     W
            # ) / torch.sum(torch.abs(W), dim=1).reshape(-1, 1)
            # part2 = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
            # part2 = part2**0.5
            # W_metric = part1 * part2

            activation_data = torch.sqrt(
                wrapped_layers[name].scaler_row.reshape((1, -1))
            )

            layer_sparsity_ratio = 1 - all_layer_ratio[i]

            if layer_sparsity_ratio <= 0:
                layer_sparsity_ratio = 0.01

            W_mask = (
                torch.zeros_like(W_metric) == 1
            )  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii : (ii + prune_m)].float()
                        W_mask.scatter_(
                            1,
                            ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1],
                            True,
                        )
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:
                    # wanda variant
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0.0, 0.8]
                    W_mask, cur_sparsity = return_given_alpha(
                        alpha, sort_res, W_metric, tmp_metric, sum_before
                    )
                    while (torch.abs(cur_sparsity - layer_sparsity_ratio) > 0.001) and (
                        alpha_hist[1] - alpha_hist[0] >= 0.001
                    ):
                        if cur_sparsity > layer_sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new
                        W_mask, cur_sparsity = return_given_alpha(
                            alpha, sort_res, W_metric, tmp_metric, sum_before
                        )
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured pruning
                    indices = sort_res[1][
                        :, : int(W_metric.shape[1] * layer_sparsity_ratio)
                    ]
                    W_mask.scatter_(1, indices, True)
            #             print ("W_mask",W_mask)
            subset[name].weight.data[W_mask] = 0  ## set weights to zero

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(
                        inps[j].unsqueeze(0), attention_mask=attention_mask
                    )[0]
                else:
                    outs[j] = layer(
                        inps[j].unsqueeze(0),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                    )[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

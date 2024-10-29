import torch
import random
import json
from torch import Tensor
from typing import TypeVar, Union
import numpy as np
from torch.distributions import Categorical
import math
import torch.nn.functional as F


Scalar = TypeVar("Scalar")
Vector = TypeVar("Vector")
Matrix = TypeVar("Matrix")

ALLTYPE = Union[Union[Scalar, Vector], Matrix]


def no_op(x: ALLTYPE) -> ALLTYPE:
    """No operation."""
    return x


def element_wise_log_op(x: ALLTYPE) -> ALLTYPE:
    if isinstance(x, torch.Tensor):
        x[x <= 0] = 1
        return torch.log(x + 1e-6)
    elif isinstance(x, np.ndarray):
        x[x <= 0] = 1
        return np.log(x + 1e-6)
    else:
        raise ValueError(f"x should be a tensor or ndarray, but got {type(x)}")


def element_wise_abslog_op(x: ALLTYPE) -> ALLTYPE:
    if isinstance(x, torch.Tensor):
        x[x == 0] = 1
        x = torch.abs(x)
        return torch.log(x)
    elif isinstance(x, np.ndarray):
        x[x == 0] = 1
        x = np.abs(x)
        return np.log(x)
    else:
        raise ValueError(f"x should be a tensor or ndarray, but got {type(x)}")


def element_wise_abs_op(x: ALLTYPE) -> ALLTYPE:
    if isinstance(x, torch.Tensor):
        return torch.abs(x)
    elif isinstance(x, np.ndarray):
        return np.abs(x)
    else:
        raise ValueError(f"x should be a tensor or ndarray, but got {type(x)}")


def element_wise_pow_op(x: ALLTYPE) -> ALLTYPE:
    if isinstance(x, torch.Tensor):
        return torch.pow(x, 2)
    elif isinstance(x, np.ndarray):
        return np.power(x, 2)
    else:
        raise ValueError(f"x should be a tensor or ndarray, but got {type(x)}")


def element_wise_exp_op(x: ALLTYPE) -> ALLTYPE:
    if isinstance(x, torch.Tensor):
        return torch.exp(x)
    elif isinstance(x, np.ndarray):
        return np.exp(x)
    else:
        raise ValueError(f"x should be a tensor or ndarray, but got {type(x)}")


def normalize_op(x: ALLTYPE) -> ALLTYPE:
    if isinstance(x, torch.Tensor):
        m = torch.mean(x)
        s = torch.std(x)
        C = (x - m) / s
        C[C != C] = 0
        return C
    elif isinstance(x, np.ndarray):
        m = np.mean(x)
        s = np.std(x)
        C = (x - m) / (s + 1e-6)
        C[np.isnan(C)] = 0
        return C
    else:
        raise ValueError(f"x should be a tensor or ndarray, but got {type(x)}")


def frobenius_norm_op(x: Matrix) -> Scalar:
    if isinstance(x, torch.Tensor):
        return torch.norm(x, p="fro")
    elif isinstance(x, np.ndarray):
        return np.linalg.norm(x, ord="fro")
    else:
        raise ValueError(f"x should be a matrix, but got {type(x)}")


def element_wise_normalized_sum_op(x: ALLTYPE) -> Scalar:
    if isinstance(x, torch.Tensor):
        return torch.sum(x) / x.numel()
    elif isinstance(x, np.ndarray):
        return np.sum(x) / x.size
    else:
        raise ValueError(f"x should be a tensor or ndarray, but got {type(x)}")


def l1_norm_op(x: ALLTYPE) -> Scalar:
    if isinstance(x, torch.Tensor):
        return torch.sum(torch.abs(x)) / x.numel()
    elif isinstance(x, np.ndarray):
        return np.sum(np.abs(x)) / x.size
    else:
        raise ValueError(f"x should be a tensor or ndarray, but got {type(x)}")


def softmax_op(x: Vector) -> Vector:
    if isinstance(x, torch.Tensor):
        return F.softmax(x, dim=0)
    elif isinstance(x, np.ndarray):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
    else:
        raise ValueError(f"x should be a vector, but got {type(x)}")


def logsoftmax_op(x: Vector) -> Vector:
    if isinstance(x, torch.Tensor):
        return F.log_softmax(x, dim=0)
    elif isinstance(x, np.ndarray):
        return np.log(softmax_op(x))
    else:
        raise ValueError(f"x should be a vector, but got {type(x)}")


def slogdet_op(x: Matrix) -> Scalar:
    if isinstance(x, torch.Tensor):
        sign, value = torch.linalg.slogdet(x)
        return value
    elif isinstance(x, np.ndarray):
        sign, value = np.linalg.slogdet(x)
        return value
    else:
        raise ValueError(f"x should be a matrix, but got {type(x)}")


def eig_op(x: Matrix) -> Vector:
    if isinstance(x, torch.Tensor):
        assert len(x.shape) == 2 and x.shape[0] == x.shape[1]
        return torch.linalg.eig(x)[0]
    elif isinstance(x, np.ndarray):
        assert len(x.shape) == 2 and x.shape[0] == x.shape[1], f"x.shape: {x.shape}"
        return np.linalg.eigvals(x)[0]
    else:
        raise ValueError(f"x should be a matrix, but got {type(x)}")


def hamming_op(x: ALLTYPE) -> Scalar:
    if isinstance(x, (list, tuple)):
        K_list = []
        for x_item in x:
            if isinstance(x_item, torch.Tensor):
                K_list.append(x_item @ x_item.t())
                K_list.append((1.0 - x_item) @ (1.0 - x_item.t()))
            elif isinstance(x_item, np.ndarray):
                K_list.append(x_item @ x_item.T)
                K_list.append((1.0 - x_item) @ (1.0 - x_item.T))
        return sum(K_list)
    else:
        if isinstance(x, torch.Tensor):
            return x @ x.t() + (1.0 - x) @ (1.0 - x.t())
        elif isinstance(x, np.ndarray):
            return x @ x.T + (1.0 - x) @ (1.0 - x.T)
        else:
            raise ValueError(f"x should be a tensor or ndarray, but got {type(x)}")


def element_wise_sigmoid_op(x: ALLTYPE) -> ALLTYPE:
    if isinstance(x, torch.Tensor):
        return torch.sigmoid(x)
    elif isinstance(x, np.ndarray):
        return 1 / (1 + np.exp(-x))
    else:
        raise ValueError(f"x should be a tensor or ndarray, but got {type(x)}")


def element_wise_tanh_op(x: ALLTYPE) -> ALLTYPE:
    if isinstance(x, torch.Tensor):
        return torch.tanh(x)
    elif isinstance(x, np.ndarray):
        return np.tanh(x)
    else:
        raise ValueError(f"x should be a tensor or ndarray, but got {type(x)}")


def gram_matrix_op(x: Tensor):
    assert (
        len(x.shape) == 2
    ), f"Gram matrix operation only support 2D tensor, but got {len(x)}D tensor."
    return x @ x.T


def corref_op(x: Tensor):
    assert (
        len(x.shape) == 2
    ), f"Correlation operation only support 2D tensor, but got {len(x)}D tensor."
    return torch.corrcoef(x)


def determinant_op(x: Matrix) -> Scalar:
    if isinstance(x, torch.Tensor):
        return torch.det(x)
    elif isinstance(x, np.ndarray):
        return np.linalg.det(x)
    else:
        raise ValueError(f"x should be a matrix, but got {type(x)}")


def diagonal_op(x: Matrix) -> Vector:
    if isinstance(x, torch.Tensor):
        return torch.diagonal(x)
    elif isinstance(x, np.ndarray):
        return np.diagonal(x)
    else:
        raise ValueError(f"x should be a matrix, but got {type(x)}")


def rank_op(x: Matrix) -> Scalar:
    if isinstance(x, torch.Tensor):
        return torch.linalg.matrix_rank(x)
    elif isinstance(x, np.ndarray):
        return np.linalg.matrix_rank(x)
    else:
        raise ValueError(f"x should be a matrix, but got {type(x)}")


def geometric_mean_op(x: ALLTYPE) -> Scalar:
    if isinstance(x, torch.Tensor):
        return torch.prod(x).pow(1.0 / x.numel())
    elif isinstance(x, np.ndarray):
        return np.prod(x) ** (1.0 / x.size)
    else:
        raise ValueError(f"x should be a tensor or ndarray, but got {type(x)}")


def entropy_op(x: torch.Tensor) -> torch.Tensor:
    if len(x.shape) == 1:
        # If the input is a 1D tensor, assume it represents probabilities
        probs = x
    else:
        # If the input is a 2D tensor, assume it represents logits and convert to probabilities
        probs = torch.softmax(x, dim=-1)

    # Check for invalid probabilities (NaNs or negative values)
    valid_probs = probs[~torch.isnan(probs) & (probs >= 0)]

    if valid_probs.numel() == 0:
        # If all probabilities are invalid, return NaN
        return torch.tensor(float("nan"))

    # Normalize the valid probabilities to ensure they sum up to 1
    valid_probs = valid_probs / valid_probs.sum(dim=-1, keepdim=True)

    # Create a Categorical distribution from the valid probabilities
    dist = Categorical(probs=valid_probs)

    # Calculate the entropy of the distribution
    entropy = dist.entropy()

    return entropy


class LayerEngine:
    """
    A class for computing the importance of layers in a neural network using a configurable computational graph.

    The LayerEngine class allows you to define a computational graph using a string representation, which consists
    of a series of stick operations (element-wise operations) and deform operations (matrix operations) applied to
    the weight matrix (W) of a layer. The class provides methods to parse the graph string, compute the layer
    importance based on the defined operations, and generate random graphs.

    The supported stick operations are:
    - ABS: Element-wise absolute value
    - LOG: Element-wise logarithm
    - ABSLOG: Element-wise absolute value followed by logarithm
    - POW: Element-wise power
    - EXP: Element-wise exponential
    - NORMALIZE: Normalization
    - SIGMOID: Element-wise sigmoid
    - TANH: Element-wise hyperbolic tangent

    The supported deform operations are:
    - GRAM: Gram matrix
    - CORREF: Correlation coefficient
    - DIAGONAL: Diagonal of the matrix
    - FROBENIUS_NORM: Frobenius norm
    - L1_NORM: L1 norm
    - DETERMINANT: Determinant of the matrix
    - RANK: Rank of the matrix
    - GEOMETRIC_MEAN: Geometric mean
    - MEAN: Mean of the matrix elements
    - VAR: Variance of the matrix elements
    - STD: Standard deviation of the matrix elements
    - SLOGDET: Sign and log of the determinant
    - ENTROPY: Entropy of the matrix

    Args:
        graph_string (str, optional): A string representation of the computational graph. If not provided, a random
            graph will be generated. The string format should be: 'W:(STICK_OPS)-(DEFORM_OPS)', where STICK_OPS and
            DEFORM_OPS are comma-separated lists of stick and deform operations, respectively.

    Attributes:
        _STICK_OPS (dict): A dictionary mapping stick operation names to their corresponding functions.
        _DEFORM_OPS (dict): A dictionary mapping deform operation names to their corresponding functions.
        _graph_structure (dict): A dictionary representing the parsed computational graph structure.
        _graph_string (str): The string representation of the computational graph.

    Methods:
        compute_importance(W): Applies the graph of operations to the weight matrix W to compute the layer importance.
        from_string(graph_string): Creates a LayerEngine instance from a graph string.
        generate_random_graph(): Generates a random computational graph string.
        save_json(json_path): Saves the graph structure to a JSON file.
        load_json(json_path): Loads the graph structure from a JSON file.

    Example:
        pengine = LayerEngine("W:(ABS,LOG)-(FROBENIUS_NORM)")
        importance = pengine.compute_importance(weight_matrix)
    """

    def __init__(self, graph_string=None):
        self._STICK_OPS = {
            "NO_OP": no_op,
            "ABS": element_wise_abs_op,
            "LOG": element_wise_log_op,
            "ABSLOG": element_wise_abslog_op,
            "POW": element_wise_pow_op,
            "EXP": element_wise_exp_op,
            "NORMALIZE": normalize_op,
            "SIGMOID": element_wise_sigmoid_op,
            "TANH": element_wise_tanh_op,
            "MMS": lambda x: (x - torch.min(x)) / (torch.max(x) - torch.min(x) + 1e-9),
        }

        self._DEFORM_OPS = {
            "NO_OP": no_op,
            "GRAM": gram_matrix_op,
            "CORREF": corref_op,
            "DIAGONAL": diagonal_op,
            "FROBENIUS_NORM": frobenius_norm_op,
            "L1_NORM": l1_norm_op,
            "DETERMINANT": determinant_op,
            "RANK": rank_op,
            "GEOMETRIC_MEAN": geometric_mean_op,
            "MEAN": torch.mean,
            "VAR": torch.var,
            "STD": torch.std,
            "SLOGDET": slogdet_op,
            "ENTROPY": entropy_op,
        }

        self._POST_OPS = {
            "NO_OP": no_op,
            "SIN": torch.sin,
            "COS": torch.cos,
            "TAN": torch.tan,
            "ASIN": torch.asin,
            "ACOS": torch.acos,
            "ATAN": torch.atan,
            "EXP": torch.exp,
            "LOG": torch.log,
            "ABS": torch.abs,
            "SIGMOID": torch.sigmoid,
            "TANH": torch.tanh,
            "SOFTPLUS": F.softplus,
        }

        self._graph_structure = {
            "W_STICK_OP": None,
            "W_DEFORM_OP": None,
            "W_POST_OP": None,
            "LAMBDA": None,
        }
        if graph_string is not None:
            self._graph_string = self._parse_graph(graph_string)
        else:
            self._graph_string = self.generate_random_graph()

        self.LAMBDA = random.randint(7, 10)

    def _parse_graph(self, graph_string):
        """Parses the graph string into a structured computational graph"""
        # Example of string: 'W:(GRAM,ABS)-(FROBENIUS_NORM)-(SIN,TANH)-(8)'
        if "-" in graph_string:
            if len(graph_string.split("-")) == 3:
                stick_ops_string, deform_ops_string, post_ops_string = (
                    graph_string.split("-")
                )
            elif len(graph_string.split("-")) == 4:
                (
                    stick_ops_string,
                    deform_ops_string,
                    post_ops_string,
                    lambda_op_string,
                ) = graph_string.split("-")

                self.LAMBDA = int(lambda_op_string.strip("()"))
                self._graph_structure["LAMBDA"] = self.LAMBDA
            else:
                raise ValueError(f"Invalid graph format: {graph_string}")

            stick_ops_string = stick_ops_string.strip("W:(").strip(")")
            deform_ops_string = deform_ops_string.strip("(").strip(")")
            post_ops_string = post_ops_string.strip("(").strip(")")

            stick_ops = stick_ops_string.split(",") if stick_ops_string else []
            deform_ops = deform_ops_string.split(",") if deform_ops_string else []
            post_ops = post_ops_string.split(",") if post_ops_string else []

            self._graph_structure["W_STICK_OP"] = [
                self._STICK_OPS[op] for op in stick_ops
            ]
            self._graph_structure["W_DEFORM_OP"] = [
                self._DEFORM_OPS[op] for op in deform_ops
            ]
            self._graph_structure["W_POST_OP"] = [self._POST_OPS[op] for op in post_ops]

        else:
            raise ValueError(f"Invalid graph format: {graph_string}")

        return graph_string

    def compute_importance(self, W):
        """Applies the graph of operations to W to compute the layer importance"""
        if self._graph_structure["W_STICK_OP"] is not None:
            for operation in self._graph_structure["W_STICK_OP"]:
                W = operation(W)

        if self._graph_structure["W_DEFORM_OP"] is not None:
            for operation in self._graph_structure["W_DEFORM_OP"]:
                W = operation(W)

        if self._graph_structure["W_POST_OP"] is not None:
            for operation in self._graph_structure["W_POST_OP"]:
                W = operation(W)

        # if W is scalar return it
        if isinstance(W, torch.Tensor) and W.numel() == 1:
            value = W.item()
        else:
            value = torch.mean(W).item()

        # Handle special values
        if math.isnan(value):
            return 0.0  # or any other appropriate default value
        elif math.isinf(value):
            return -1  # denotes invalid
        elif value == 0:
            return 0.0
        elif value == 1:
            return 1.0
        else:
            return value

    @staticmethod
    def from_string(graph_string):
        return LayerEngine(graph_string)

    def __str__(self):
        return self._graph_string

    def __repr__(self) -> str:
        return f"LayerEngine({self._graph_string})"

    def generate_random_graph(self):
        """Generates a random graph string"""
        stick_ops = ",".join(
            random.choices(list(self._STICK_OPS.keys()), k=random.randint(0, 2))
        )
        deform_ops = ",".join(random.choices(list(self._DEFORM_OPS.keys()), k=1))
        post_ops = ",".join(
            random.choices(list(self._POST_OPS.keys()), k=random.randint(1, 2))
        )
        lambda_ops = random.randint(7, 10)

        if stick_ops == "":
            stick_ops = "NO_OP"

        graph_string = f"W:({stick_ops})-({deform_ops})-({post_ops})-({lambda_ops})"
        # print("DEBUG:", graph_string)
        self._parse_graph(graph_string)
        return graph_string

    def save_json(self, json_path=None):
        assert json_path is not None, "Path must not be None"
        assert json_path.endswith(".json"), "Path must end with .json"
        # save self._graph_structure to a json file
        with open(json_path, "w") as f:
            json.dump(self._graph_structure, f)

    def load_json(self, json_path=None):
        assert json_path is not None, "Path must not be None"
        assert json_path.endswith(".json"), "Path must end with .json"
        # load self._graph_structure from a json file
        with open(json_path, "r") as f:
            self._graph_structure = json.load(f)

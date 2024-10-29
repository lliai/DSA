from scipy import optimize

def ompq_process(all_layer_ratio, args):
    P = all_layer_ratio
    L = len(all_layer_ratio)
    S = args.sparsity_ratio
    LAMBDA = 8

    # Objective function
    def func(x):
        """objective function"""
        sum_func = []
        assert len(P) == len(
            x
        ), f"Length of P and x should be the same. P: {len(P)}, x: {len(x)}"
        for i in range(L):
            sum_func.append(P[i] * x[i])
        return -sum(sum_func) + LAMBDA * sum((x - S) ** 2)

    # Constrain function
    # Average of layer-wise sparsity should less or equal to S
    def constrain_func(x):
        """constrain function"""
        return S - sum(x) / L

    # Derivate of the objective function
    def func_deriv(x):
        return -P + 2 * LAMBDA * (x - S)

    # Bounds for x
    bounds = [(0, 1) for i in range(L)]

    result = optimize.minimize(
        func,
        x0=[S for i in range(L)],
        jac=func_deriv,
        method="SLSQP",
        constraints=[{"type": "ineq", "fun": constrain_func}],
        bounds=bounds,
    )
    all_layer_ratio = result.x
    return all_layer_ratio 
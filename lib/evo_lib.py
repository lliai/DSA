import random 


def initialize_population(population_size, n_layers, args):
    """
    Initialize the population with a base sparsity ratio and small random perturbations.
    """
    population = []
    for _ in range(population_size):
        gene = [args.sparsity_ratio] * n_layers
        perturbations = [random.uniform(0, 0.005) for _ in range(n_layers)]
        individual = []
        for base, pert in zip(gene, perturbations):
            if random.random() < 0.5:
                individual.append(base + pert)
            else:
                individual.append(base - pert)
        population.append(individual)
    return population

def mutate_gene(gene, delta_sparsity, mutation_prob):
    """
    Mutate the gene (sparsity ratio) by delta_sparsity with the given mutation probability.
    """
    mutated_gene = []
    for sparsity_ratio in gene:
        if random.random() < mutation_prob:
            # Mutate the sparsity ratio
            if random.random() < 0.5:
                new_sparsity_ratio = max(0.0, min(1.0, sparsity_ratio + delta_sparsity))
            else:
                new_sparsity_ratio = max(0.0, min(1.0, sparsity_ratio - delta_sparsity))
            mutated_gene.append(new_sparsity_ratio)
        else:
            mutated_gene.append(sparsity_ratio)
    return mutated_gene


def fitness(layer_importance: list, gene: list):
    assert len(layer_importance) == len(gene), f"Length of layer_importance and gene should be the same. layer_importance: {len(layer_importance)}, gene: {len(gene)}"
    
    return sum([imp * gn for imp, gn in zip(layer_importance, gene)])

    # return sum([imp * (1 - gn) for imp, gn in zip(layer_importance, gene)])


def evolution_for_gene(layer_importance, args):
    import matplotlib.pyplot as plt 

    population = initialize_population(args.popu_size, len(layer_importance), args)

    # for plot the search process;
    iter_index = []
    score_index = []

    # best individual
    best_indiv = None 
    best_score = -1

    # run the cycle 
    for i in range(args.iterations):
        print(f"iteration {i}")
        # evaluate the population 
        score_list = []
        for indiv in population:
            score = fitness(layer_importance, indiv)
            score_list.append(score)

            if score > best_score:
                best_score = score 
                best_indiv = indiv 

        # print the best individual 
        print(f"best individual {best_indiv}")
        print(f"best score {best_score}")

        # selection
        # sort the population based on the score
        population = [x for _, x in sorted(zip(score_list, population), key=lambda pair: pair[0])]
        score_list.sort()

        # select the individual from top 5% of the population
        parent_candidate = population[:int(0.05 * args.popu_size)]

        # select the parent for mutation 
        m_parent_gene = random.choice(parent_candidate)

        # mutation
        if i < args.iterations // 3:
            _delta = args.delta_sparsity 
        elif i < args.iterations * 2 // 3:
            _delta = args.delta_sparsity / 2
        else: 
            _delta = args.delta_sparsity / 4
        mutated_gene = mutate_gene(m_parent_gene, _delta, args.mutation_prob)

        # evaluatte the mutated gene 
        m_score = fitness(layer_importance, mutated_gene)

        # replace the worst individual with the mutated gene
        population[-1] = mutated_gene


        # store the best individual
        iter_index.append(i)
        score_index.append(best_score)

    print(f"after {args.iterations} iterations, the best individual is {best_indiv}")

    plt.plot(iter_index, score_index)
    plt.xlabel("Iterations")
    plt.ylabel("Perplexity")
    plt.title("Evolution Search")
    plt.savefig("./evolution_search.png")

    return best_indiv

def post_adjust_list_mean(original_list, target_mean, min_value=0, max_value=1):
    original_list = [max(min(value, max_value), min_value) for value in original_list]

    current_mean = sum(original_list) / len(original_list)
    total_adjustment_needed = target_mean * len(original_list) - sum(original_list)

    modified_list = [max(min(value + (value * total_adjustment_needed / sum(original_list)), max_value), min_value) for value in original_list]

    return modified_list, current_mean, total_adjustment_needed



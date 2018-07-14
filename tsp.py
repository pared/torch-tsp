import torch
import random
import utils


class TSP:

    def __init__(self, cost_matrix, forbidden, use_gpu=False):
        """
        :param cost_matrix: 2d matrix representing costs of travelling between points af appropriate indexes
        :param forbidden: matrix of same shape as above filled wiht 1's and 0's, cannot use connections marked as 1.
        """

        assert cost_matrix.shape == forbidden.shape
        self.use_gpu=use_gpu
        self.cost_matrix = cost_matrix
        self.forbidden = forbidden
        if self.use_gpu:
            self.cost_matrix = self.cost_matrix.cuda()
            self.forbidden = self.forbidden.cuda()

    def individual(self, ids):
        # ids shall match cost_matrix rows and cols
        assert(len(ids) == self.cost_matrix.shape[0])

        # ids shall contai all possible row/col indexes
        assert(set(ids) == set([n for n in range(len(ids))]))

        individual = torch.zeros(len(ids), len(ids), dtype=torch.float)
        for i, id in enumerate(ids):
            individual[ids[i - 1]][id] = 1

        assert self.is_elgible(individual)

        if self.use_gpu:
            individual = individual.cuda()

        return individual

    def evaluate(self, individual):
        """
        :param individual: individual is a tensor od shape equal to cost_matrix, each column of individual has single 1 and zeros
        :return:
        """
        assert(len(individual) == len(set(individual)))
        return torch.sum(self.cost_matrix * self.individual(individual))

    def evaluate_multiple(self, individuals):

        
        individuals_as_matrices = [self.individual(i) for i in individuals]
        inds_tensor = torch.stack(individuals_as_matrices)
        if self.use_gpu:
            inds_tensor = inds_tensor.cuda()
        assert len(inds_tensor.shape) == 3
        return list(torch.sum(inds_tensor * self.cost_matrix, -1).sum(-1))

    def is_elgible(self, individual):
        does_not_contain_forbidden_routes = (self.forbidden * individual).sum() == 0
        # assert does_not_contain_forbidden_routes
        has_only_one_route_in_each_row = individual.sum(dim=1).eq(torch.ones(individual.shape[0])).all()
        # assert has_only_one_route_in_each_row
        has_only_one_route_in_each_col = individual.sum(dim=0).eq(torch.ones(individual.shape[0])).all()
        # assert has_only_one_route_in_each_col

        return does_not_contain_forbidden_routes and has_only_one_route_in_each_col and has_only_one_route_in_each_row

    def create_random_population(self, num_individuals):
        return [self.random_individual() for _ in range(num_individuals)]

    def random_individual(self):

        indices = [n for n in range(self.cost_matrix.shape[0])]
        random.shuffle(indices)
        return indices

    def tournament(self, pop_fitness, tournament_size=5):
        index_w_fitness = [(i, f) for i, f in enumerate(pop_fitness)]
        subpop = random.sample(index_w_fitness, tournament_size)
        return [e[0] for e in sorted(subpop, key=lambda pair: pair[1])[:2]]

    def crossover(self, parent1, parent2, split_index=None):
        if split_index is None:
            split_index = random.randint(0, len(parent1))

        child = parent1[:split_index]
        for elem in parent2:
            if elem not in child:
                child.append(elem)

        assert len(child) == len(parent1) == len(parent2)
        return child

    def mutate(self, individual, prob = 0.05, replace_indexes=None):
        ind1 = None
        ind2 = None

        if replace_indexes:
            assert len(replace_indexes) == 2
            ind1 = replace_indexes[0]
            ind2 = replace_indexes[1]

        if not ind1:
            if random.random() < prob:
                ind1 = random.randint(0, len(individual)-1)
                ind2 = random.randint(0, len(individual)-1)

        if ind1 is not None:
            individual[ind1], individual[ind2] = individual[ind2], individual[ind1]

        return individual

    def single_generation(self,
                          population,
                          preserve_best=0.25,
                          tournament_size=5):
        # evaluate population fitness
        pop_fitness = self.evaluate_multiple(population)

        # select indexes of best indiviudals
        most_fit_indexes = utils.get_sorted_n_args(
            pop_fitness, int(preserve_best * len(population)))

        # preserve best individuals
        new_population = [population[i] for i in most_fit_indexes]

        # choose parents indexes for crossover
        parents_to_crossover = [self.tournament(pop_fitness, tournament_size)
                                for i in range(len(population) - len(new_population))]

        for parent_1_index, parent_2_index in parents_to_crossover:

            new_population.append(self.crossover(
                population[parent_1_index],
                population[parent_2_index]))

        assert len(new_population) == len(population)

        for individual in new_population:
            self.mutate(individual)

        return new_population


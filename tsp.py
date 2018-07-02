import torch
import random
import utils


class TSP:

    def __init__(self, cost_matrix, forbidden):
        """
        :param cost_matrix: 2d matrix representing costs of travelling between points af appropriate indexes
        :param forbidden: matrix of same shape as above filled wiht 1's and 0's, cannot use connections marked as 1.
        """

        assert cost_matrix.shape == forbidden.shape
        self.cost_matrix = cost_matrix
        self.forbidden = forbidden

    def individual(self, ids):
        # ids shall match cost_matrix rows and cols
        assert(len(ids) == self.cost_matrix.shape[0])

        # ids shall contai all possible row/col indexes
        assert(set(ids) == set([n for n in range(len(ids))]))

        individual = torch.zeros(len(ids), len(ids), dtype=torch.float)
        for i, id in enumerate(ids):
            individual[ids[i - 1]][id] = 1

        assert self.is_elgible(individual)
        return individual

    def evaluate(self, individual):
        """
        :param individual: individual is a tensor od shape equal to cost_matrix, each column of individual has single 1 and zeros
        :return:
        """
        assert(self.cost_matrix.shape == individual.shape)
        return torch.sum(self.cost_matrix * individual)

    def evaluate_multiple(self, individuals):
        """
        :param individuals: List of tensors of shape equalt to cost matrix, each column has to have single 1 and zeros
        :return:
        """
        inds_tensor = torch.stack(individuals)
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
        while(True):
            individual = self.individual(indices)
            if self.is_elgible(individual):
                return individual
            else:
                print("not returning")
                random.shuffle(indices)

    def _tournament(self, pop_fitness, tournament_size=5):
        index_w_fitness = [(i, f) for i, f in enumerate(pop_fitness)]
        subpop = random.sample(index_w_fitness, tournament_size)
        return [e[0] for e in sorted(subpop, key=lambda pair: pair[1])[:2]]

    def crossover(self, parent1, parent2, split_index=None):

        while True:

            if not split_index:
                split_index = random.randint(1, parent1.shape[0])

        # first part comes from parent1
            child = parent1[:, :split_index]

        # not which rows already have ones (individual has to have exactly one 1 in each row and col)
            _, occupied_rows = child.max(dim=0)

        # get indexes where given row of p2 has 1
            _, p2_max_col_indexes = parent2.max(dim=1)

        # take all cols indexes that are not already occupied in child
            p2_cols_to_preserve = [e for row, e in enumerate(p2_max_col_indexes) if row not in occupied_rows]

        # copy each col:
            child = [child]
            [child.append(parent2[:, c].reshape([parent1.shape[0], 1])) for c in sorted(p2_cols_to_preserve)]

            result = torch.cat(child, 1)

            if self.is_elgible(result):
                return result


    def single_generation(self,
                          population,
                          preserve_best=0.25,
                          tournament_size=5):
        # evaluate population fitness
        pop_fitness = self.evaluate_multiple(population)

        # select indexes of best indiviudals
        most_fit_indexes = utils.get_max_indexes(
            pop_fitness, int(preserve_best * len(population)))

        # preserve best individuals
        new_population = [population[i] for i in most_fit_indexes]

        # choose parents indexes for crossover
        parents_to_crossover = [self._tournament(pop_fitness, tournament_size)
                                for i in range(len(population) - len(new_population))]

        for parent_1_index, parent_2_index in parents_to_crossover:
            new_population.append(self.crossover(
                population[parent_1_index],
                population[parent_2_index]))

        print(parents_to_crossover)

        return new_population

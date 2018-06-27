
import torch

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
        return torch.sum(inds_tensor * self.cost_matrix, -1).sum(-1)

    def is_elgible(self, individual):
        return (self.forbidden * individual).sum() == 0

    def create_random_population(self, num_individuals):
        pass





import numpy as np
from numpy.random import uniform
from random import choice
import math


class Population:
    def __init__(self, bitstring='', vector=None, cost=0, affinity=0):
        if vector is None:
            vector = [0, 0]
        self.bitstring = bitstring
        self.vector = vector
        self.cost = cost
        self.affinity = affinity

    def set_bitstring(self, bitstring):
        self.bitstring = bitstring

    def set_vector(self, vector):
        self.vector = vector

    def set_cost(self, cost):
        self.cost = cost

    def set_affinity(self, affinity):
        self.affinity = affinity


def objective_function(vector):
    return np.sum(np.power(vector, 2))


def decode(bitstring, search_space, bits_per_param):
    vector = []

    for i, bounds in enumerate(search_space):
        off = i * bits_per_param
        sum = 0.0
        param_rev = bitstring[off:(off + bits_per_param)]
        param = param_rev[::-1]

        for x in range(len(param)):
            sum += (1.0 if (param[x] == '1') else 0.0) * (np.power(2.0, float(x)))

        min = bounds[0]
        max = bounds[1]

        vector.append(min + ((max - min) / (np.power(2.0, float(bits_per_param)) - 1.0)) * sum)
    return vector


def evaluate(pop, search_space, bits_per_param):
    for p in pop:
        p.set_vector(decode(p.bitstring, search_space, bits_per_param))
        p.set_cost(objective_function(p.vector))


def random_bitstring(num_bits):
    return ''.join(choice('01') for _ in range(num_bits))


def point_mutation(bitstring, rate):
    child = ''

    for x in range(len(bitstring)):
        bit = bitstring[x]
        child += (('0' if (bit == '1') else '1') if (uniform(0, 1) < rate) else bit)

    return child


def calculate_mutation_rate(antibody, mutate_factor=-2.5):
    return math.exp(mutate_factor * antibody.affinity)


def calculate_num_clones(pop_size, clone_factor):
    return math.floor(pop_size * clone_factor)


def calculate_affinity(pop):
    pop = sorted(pop, key=lambda _p: _p.cost)
    cost_range = pop[-1].cost - pop[0].cost

    if cost_range == 0.0:
        for p in pop:
            p.set_affinity(1.0)
    else:
        for p in pop:
            p.set_affinity(1.0 - (p.cost / cost_range))


def clone_and_hypermutate(pop, clone_factor):
    clones = []
    num_clones = calculate_num_clones(len(pop), clone_factor)
    calculate_affinity(pop)
    for p in pop:
        m_rate = calculate_mutation_rate(p)
        for _ in range(num_clones):
            clone = Population(bitstring=point_mutation(p.bitstring, m_rate))
            clones.append(clone)

    return clones


def random_insertion(search_space, pop, num_rand, bits_per_param):
    if num_rand == 0:
        return pop

    rands = []
    for _ in range(num_rand):
        rands.append(Population(bitstring=random_bitstring(len(search_space) * bits_per_param)))

    evaluate(rands, search_space, bits_per_param)

    sorted_pop = sorted(pop + rands, key=lambda p: p.cost)
    return sorted_pop[:len(pop)]


def search(search_space, max_gens, pop_size, clone_factor, num_rand, bits_per_param=16):
    pop = []
    for _ in range(pop_size):
        pop.append(Population(bitstring=random_bitstring(len(search_space) * bits_per_param)))

    evaluate(pop, search_space, bits_per_param)
    best = min(pop, key=lambda p: p.cost)
    for x in range(max_gens):
        clones = clone_and_hypermutate(pop, clone_factor)
        evaluate(clones, search_space, bits_per_param)
        sorted_pop = sorted(pop + clones, key=lambda p: p.cost)
        pop = sorted_pop[:pop_size]
        pop = random_insertion(search_space, pop, num_rand, bits_per_param)
        best = min((pop + [best]), key=lambda p: p.cost)

    return best


if __name__ == '__main__':
    problem_size = 2
    search_space = []

    for x in range(problem_size):
        search_space.append([-5, +5])

    max_gens = 100
    pop_size = 100
    clone_factor = 0.1
    num_rand = 2
    best = search(search_space, max_gens, pop_size, clone_factor, num_rand)
    print('SOLUTION: f=' + str(best.cost) + ', s=' + str(best.vector))

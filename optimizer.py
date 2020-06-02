from network import Network
from functools import reduce
from operator import add
import random
import math


class Optimizer():

    def __init__(self, params, retain=0.4,
                 random_select=0.1, mutate_chance=0.2):
        self.params = params
        self.random_select = random_select
        self.retain = retain
        self.mutate_chance = mutate_chance

    def population(self, count):
        population = []
        for _ in range(0, count):
            network = Network(self.params)
            print('Randomize network parameters')
            network.random()
            network.print_network()
            population.append(network)
        return population

    @staticmethod
    def fitness(network):
        return network.accuracy

    def grade(self, population):
        summed = reduce(add, (self.fitness(network) for network in population))
        return summed / float((len(population)))

    def mutate(self, network):
        mutation = random.choice(list(self.params.keys()))

        network.network[mutation] = random.choice(
            self.params[mutation])

        return network

    def breed(self, mother, father):
        children = []
        for _ in range(2):

            child = {}

            for param in self.params:
                child[param] = random.choice(
                    [mother.network[param], father.network[param]]
                )

            network = Network(self.params)
            network.create_set(child)

            if self.mutate_chance > random.random():
                network = self.mutate(network)

            children.append(network)

        return children

    def evolve(self, population):
        graded = [(self.fitness(network), network) for network in population]
        graded = [x[1] for x in sorted(graded, key=lambda x: x[0],
                                       reverse=True)]

        retain_length = math.ceil(len(graded)*self.retain)
        # retain_length = int(len(graded)*self.retain)
        parents = graded[:retain_length]

        for individual in graded[retain_length:]:
            if self.random_select > random.random():
                parents.append(individual)

        parents_length = len(parents)
        desired_length = len(population) - parents_length
        children = []

        while len(children) < desired_length:
            male = random.randint(0, parents_length - 1)
            female = random.randint(0, parents_length - 1)

            if male != female:
                male = parents[male]
                female = parents[female]

                babies = self.breed(male, female)

                for baby in babies:
                    if len(children) < desired_length:
                        children.append(baby)

        parents.extend(children)

        return parents

    def print(self):
        print(self.params)

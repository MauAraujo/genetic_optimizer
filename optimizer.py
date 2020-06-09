from network import Network
from functools import reduce
from operator import add
import random
import math


class Optimizer():
    # Clase que representa el algoritmo genético

    def __init__(self, params, retain=0.4,
                 random_select=0.1, mutate_chance=0.2):

        # Constructor, inicializa los valores
        self.params = params
        self.random_select = random_select # Probabilidad de retener un individuo rechazado
        self.retain = retain # Porcentaje de individuos que se retendrán en la próxima generación
        self.mutate_chance = mutate_chance # Probabilidad de mutación

    def population(self, count):
        # Función para crear una población

        population = []
        for _ in range(0, count):
            network = Network(self.params) # Instanciar red
            print('Randomize network parameters')
            network.random() # Iniciar con valores aleatorios
            network.print_network()
            population.append(network)
        return population

    @staticmethod
    def fitness(network):
        # La precisión de la red se utiliza como función fitness

        return network.accuracy

    def grade(self, population):
        # Función para evaluar la población,
        # encuentra el promedio de fitness

        summed = reduce(add, (self.fitness(network) for network in population))
        return summed / float((len(population)))

    def mutate(self, network):
        # Función para mutar un parámetro aleatoriamente

        mutation = random.choice(list(self.params.keys()))

        network.network[mutation] = random.choice(
            self.params[mutation])

        return network

    def breed(self, mother, father):
        # Función para crear un nuevo individuo a partir de padre y madre

        children = []
        for _ in range(2):

            child = {}

            # Escoger parametros del hijo utilizando a los padres
            for param in self.params:
                child[param] = random.choice(
                    [mother.network[param], father.network[param]]
                )

            # Instanciar red y asignar valores
            network = Network(self.params)
            network.create_set(child)

            # Aplicar mutación aleatoriamente al nuevo individuo
            if self.mutate_chance > random.random():
                network = self.mutate(network)

            children.append(network)

        return children

    def evolve(self, population):
        # Función para evolucionar la población

        # Obtener fitness promedio de cada individuo
        graded = [(self.fitness(network), network) for network in population]

        # Sortear las redes según su fitness
        graded = [x[1] for x in sorted(graded, key=lambda x: x[0],
                                       reverse=True)]

        # Obtener cuenta de individuos que se mantendrán
        # en la próxima generación
        retain_length = math.ceil(len(graded)*self.retain)
        # retain_length = int(len(graded)*self.retain)

        # Individuos seguros que se mantendrán (los mejores)
        # y utilizarán para crear nuevos individuos
        parents = graded[:retain_length]

        # Individuos aleatorios que se mantendrán (para agregar variedad a la población y no sesgarla)
        for individual in graded[retain_length:]:
            if self.random_select > random.random():
                parents.append(individual)

        # Calcular cúantas redes faltan para crear la
        # nueva población y llenarlas con los hijos
        parents_length = len(parents)
        desired_length = len(population) - parents_length
        children = []

        # Agregar hijos
        while len(children) < desired_length:
            print(len(children) < desired_length)
            # Obtener padres de manera aleatoria de los disponibles
            male = random.randint(0, parents_length - 1)
            female = random.randint(0, parents_length - 1)

            # Reproducirlos solo si no son la misma red
            if male != female:
                print('Inside')
                male = parents[male]
                female = parents[female]

                # Creación de nuevos individuos (hijos)
                babies = self.breed(male, female)

                # Agregar a los hijos uno por uno,
                # respetando el tamaño de población
                for baby in babies:
                    if len(children) < desired_length:
                        children.append(baby)
            else:
                break

        # Combinar padres e hijos para crear la nueva población
        parents.extend(children)

        return parents

    def print(self):
        # Función para imprimir parametros seleccionados
        print(self.params)

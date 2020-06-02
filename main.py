from network import Network
from optimizer import Optimizer
import numpy as np
from tqdm import tqdm


def train_networks(networks, dataset):
    pbar = tqdm(total=len(networks))
    for network in networks:
        network.train(dataset)
        pbar.update(1)
    pbar.close()


def get_average_accuracy(networks):
    total_accuracy = 0
    for network in networks:
        total_accuracy += network.accuracy

    return total_accuracy / len(networks)


def generate(generations, population, params, dataset):
    f= open("results.txt","a+")
    optimizer = Optimizer(params, 0.1, 0.2)
    networks = optimizer.population(population)

    for i in range(generations):
        print('generation', i)
        train_networks(networks, dataset)
        average_accuracy = get_average_accuracy(networks)
        print("Generation average: %.2f%%" % (average_accuracy * 100))

        if i != generations - 1:
            networks = optimizer.evolve(networks)

    networks = sorted(networks, key=lambda x: x.accuracy, reverse=True)
    write_results(networks[:5], f)
    f.close()


def write_results(networks, f):
    for network in networks:
        f.write(network.get_result())

def main():
    print('Tamaño de población: ')
    population = int(input())

    if population < 4:
        print('Población mínima de 4')
        return
        
    print('Número de generaciones: ')
    generations = int(input())
    dataset = 'fire.csv'

    params = {
        'neurons': list(range(3, 24)),
        'layers': list(range(1, 5)),
        'epochs': list(range(100, 151, 50)),
        'lr': list(np.arange(0.2, 0.41, 0.01)),
        'momentum': list(np.arange(0.2, 0.41, 0.01))
    }

    print('Iniciando con parámetros: ', params)
    generate(generations, population, params, dataset)


if __name__ == '__main__':
    main()

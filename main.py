from network import Network
from optimizer import Optimizer
import numpy as np
from tqdm import tqdm


def train_networks(networks, dataset):
    # Función para entrenar a cada una de las redes de la población

    pbar = tqdm(total=len(networks)) # Para mostrar la barra de progreso en la terminal
    for network in networks:
        network.train(dataset) # Entrenar a la red
        pbar.update(1)
    pbar.close()


def get_average_accuracy(networks):
    # Función para calcular el promedio de precisión

    total_accuracy = 0
    for network in networks:
        total_accuracy += network.accuracy

    return total_accuracy / len(networks)


def generate(generations, population, params, dataset):
    # Función para crear las generaciones con sus respectivas poblaciones

    f= open("results.txt","a+") # Abrir archivo donde se guardarán los resultados
    optimizer = Optimizer(params, 0.1, 0.2) # Instanciar el algoritmo genético
    networks = optimizer.population(population) # Crear la población con el número deseado

    # ciclo de generaciones
    for i in range(generations):
        print('generation', i)

        # Entrenar a todas las redes de la población
        train_networks(networks, dataset)

        # Obtener la precisión promedio de la generación
        average_accuracy = get_average_accuracy(networks)
        print("Generation average: %.2f%%" % (average_accuracy * 100))

        # Si no es la última generación, evolucionar a la población
        if i != generations - 1:
            networks = optimizer.evolve(networks)

    # Una vez terminado el ciclo de generaciones, se sortean los resultados según la precisión
    networks = sorted(networks, key=lambda x: x.accuracy, reverse=True)

    #Escribir los resultados (5 mejores redes) en un archivo
    write_results(networks[:5], f)
    f.close()


def write_results(networks, f):
    # Función para escribir los resultados a un archivo
    for network in networks:
        f.write(network.get_result())

def main():
    # Función principal, lee del input el tamaño de la población y el número de generaciones
    print('Tamaño de población: ')
    population = int(input()) # Leer el tamaño de población

    # Población mínima de 4 para poder cruzar las redes
    if population < 4:
        print('Población mínima de 4')
        return

    print('Número de generaciones: ')
    generations = int(input()) # Leer el número de generaciones
    dataset = 'fire.csv' # Selección de base de datos

    # Parámetros iniciales para los experimentos, definidos en rangos
    params = {
        'neurons': list(range(3, 24)),
        'layers': list(range(1, 5)),
        'epochs': list(range(100, 151, 50)),
        'lr': list(np.arange(0.2, 0.41, 0.01)),
        'momentum': list(np.arange(0.2, 0.41, 0.01))
    }

    print('Iniciando con parámetros: ', params)
    # Empieza el proceso generando las poblaciones requeridas
    generate(generations, population, params, dataset)


if __name__ == '__main__':
    main()

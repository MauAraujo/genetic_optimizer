from train import train
import random
import json


class Network():
# Clase que representa a una red neuronal

    def __init__(self, params):
        # Constructor, inicializa los parámetros
        # momentum, lr, neuronas, capas, épocas
        
        self.accuracy = 0
        self.params = params
        self.network = {}

    def train(self, dataset):
        # Función para entrenar la red

        if self.accuracy == 0:
            self.accuracy = train(self.network, dataset)

    def random(self):
        # Función para crear una red con parámetros aleatorios (según los rangos definidos)

        for key in self.params:
            self.network[key] = random.choice(self.params[key])

    def create_set(self, network):
        # Asignar valores a esta instancia

        self.network = network

    def print_network(self):
        # Función para imprimir los valores de la clase

        print(self.network)
        print("Network accuracy: %.2f%%" % (self.accuracy * 100))

    def get_result(self):
        # Función para obtener la precisión en string formateado

        self.network['accuracy'] = self.accuracy * 100
        return json.dumps(self.network) + '\n'

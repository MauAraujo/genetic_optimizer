from train import train
import random
import json


class Network():

    def __init__(self, params):
        self.accuracy = 0
        self.params = params
        self.network = {}

    def train(self, dataset):
        if self.accuracy == 0:
            self.accuracy = train(self.network, dataset)

    def random(self):
        for key in self.params:
            self.network[key] = random.choice(self.params[key])

    def create_set(self, network):
        self.network = network

    def print_network(self):
        print(self.network)
        print("Network accuracy: %.2f%%" % (self.accuracy * 100))

    def get_result(self):
        self.network['accuracy'] = self.accuracy * 100
        return json.dumps(self.network) + '\n'
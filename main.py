from __future__ import annotations
from enum import Enum, auto
import random
from typing import List, Optional


class InitMethods(Enum):
    ZERO = auto()
    RANDOM = auto()
    HE = auto()


class Node:
    value: Optional[float]
    weights_with_previous: List[float]
    connected_previous_nodes: List[Node]

    def __init__(self):
        self.value = None
        self.weights_with_previous = []
        self.connected_previous_nodes = []

    def connect(self, other_node: Node):
        self.connected_previous_nodes.append(other_node)

    def initialize(self, method: InitMethods):
        for index in range(len(self.connected_previous_nodes)):
            if method == InitMethods.RANDOM:
                self.weights_with_previous.append(random.random())

    def calculate_node(self) -> float:
        previous_sums = [
            weight * (node.value if node.value is not None else node.calculate_node())
            for weight, node in zip(self.weights_with_previous, self.connected_previous_nodes)
        ]
        return sum(previous_sums)

    def init_value(self, value: float):
        self.value = value


class Layer:
    _layer_number = 1

    def __init__(self, *, amount_of_nodes: int, layer_name: str = None):
        if not isinstance(amount_of_nodes, int) or amount_of_nodes < 1:
            raise ValueError("Amount of nodes can have only int type and bigger than 0!")
        self.amount_of_nodes = amount_of_nodes
        self.nodes = [Node() for _ in range(amount_of_nodes)]
        self.layer_name = layer_name if layer_name is not None else f"Layer_{Layer._layer_number}"

        Layer._layer_number += 1

    def connect_to_previous(self, other_layer: Layer):
        for self_node in self.nodes:
            for other_node in other_layer.nodes:
                self_node.connect(other_node)

    def calculate_nodes(self) -> List[float]:
        return [node.calculate_node() for node in self.nodes]

    def initialize(self, method: InitMethods):
        for node in self.nodes:
            node.initialize(method)

    def init_values(self, values: List[float]):
        for node, value in zip(self.nodes, values):
            node.init_value(value)


class NeuralNetwork:
    input_layer: Optional[Layer]
    output_layer: Optional[Layer]
    layers: List[Layer]

    def __init__(self, *, amount_of_input_nodes: int = None):
        if amount_of_input_nodes is None:
            self.input_layer = None
        else:
            self.add_input_layer(amount_of_input_nodes)

        self.output_layer = None
        self.layers = []

    def add_input_layer(self, amount_of_input_nodes: int):
        self.input_layer = Layer(amount_of_nodes=amount_of_input_nodes, layer_name="input")
        self.layers.append(self.input_layer)

    def add_output_layer(self, amount_of_output_nodes: int):
        self.output_layer = Layer(amount_of_nodes=amount_of_output_nodes, layer_name="output")
        self.output_layer.connect_to_previous(self.layers[-1])
        self.layers.append(self.output_layer)

    def add_layer(self, amount_of_layer_nodes: int, layer_name: str = None):
        if self.input_layer is None:
            raise ValueError("Add the input layer before you can")
        if self.output_layer is not None:
            raise ValueError("Neural network already has an output layer. In order to add new layer you should "
                             "consider changing the architecture.")
        new_layer = Layer(amount_of_nodes=amount_of_layer_nodes, layer_name=layer_name)
        new_layer.connect_to_previous(self.layers[-1])
        self.layers.append(new_layer)

    def initialize(self, method: InitMethods):
        for layer in self.layers[1:]:
            layer.initialize(method)

    def predict(self, data: List[float]):
        if len(data) != len(self.input_layer.nodes):
            raise ValueError("Input data should have the same size as the input layer!")
        self.input_layer.init_values(data)
        return self.output_layer.calculate_nodes()

    def print_statistics(self):
        amount_of_nodes = sum([len(layer.nodes) for layer in self.layers])
        amount_of_weights = sum([sum([len(node.weights_with_previous) for node in layer.nodes])
                                 for layer in self.layers])
        print(f"Amount of nodes: {amount_of_nodes}\n"
              f"Amount of layers: {len(self.layers)}\n"
              f"Amount of weights: {amount_of_weights}")


nn = NeuralNetwork()
nn.add_input_layer(4)
nn.add_layer(3)
nn.add_layer(10)
nn.add_output_layer(2)
nn.initialize(InitMethods.RANDOM)
nn.print_statistics()
print(nn.predict([1, 2, 3, 4]))

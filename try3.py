import random
import numpy as np

# Constants
POPULATION_SIZE = 30
NUM_GENERATIONS = 10
MUTATION_RATE = 0.2
CROSSOVER_RATE = 1
INPUT_SIZE = 16
OUTPUT_SIZE = 1
ELITE_SIZE = 5


# Load data
# Parse the data from nn0.txt
data = []
with open('nn0.txt', 'r') as file:
    for line in file:
        binary_string, label = line.strip().split()
        data.append((binary_string, int(label)))

# Split data into training and test sets
random.shuffle(data)
train_size = int(0.8 * len(data))
train_data = data[:train_size]
test_data = data[train_size:]


# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


# Neural network class
class Network:
    def __init__(self, structure, weights=None):
        self.structure = structure
        if weights is None:
            self.weights = [np.random.randn(structure[i + 1], structure[i]) for i in range(len(structure) - 1)]
        else:
            self.weights = weights

    def predict(self, input_data):
        hidden = input_data
        for layer_weights in self.weights[:-1]:
            hidden = sigmoid(np.dot(layer_weights,hidden))
        output = sigmoid(np.dot(self.weights[-1], hidden))
        return output


# Genetic algorithm
def initialize_population(population_size, input_size, output_size):
    population = []
    layer_size = np.random.randint(2, 11)
    for _ in range(population_size):
        structure = [input_size, layer_size, layer_size, layer_size, layer_size, output_size]  # Randomly initialize network structure
        population.append(Network(structure))
    return population


def calculate_fitness(network, data):
    correct_predictions = 0
    for example in data:
        input_data = np.array(list(map(int, example[0])))
        target = example[1]
        output = network.predict(input_data)
        predicted_class = 1 if output > 0.5 else 0
        if predicted_class == target:
            correct_predictions += 1
    fitness = correct_predictions / len(data)
    return fitness


def selection(population, fitness):
    parent_indices = np.random.choice(range(len(population)), size=2, replace=True, p=fitness / np.sum(fitness))
    parents = [population[idx] for idx in parent_indices]
    return parents[0], parents[1]


def crossover(parent1, parent2):
    offspring_weights = []
    for i in range(len(parent1.weights)):
        # Get the shape of the matrices
        rows, cols = parent1.weights[i].shape

        # Generate a random crossover point
        crossover_point = np.random.randint(1, cols)

        # Perform one-point crossover
        offspring = np.concatenate((parent1.weights[i][:, :crossover_point], parent2.weights[i][:, crossover_point:]),
                                   axis=1)
        offspring_weights.append(offspring)

    child = Network(parent1.structure, offspring_weights)
    return child


def mutation(network):
    mutated_weights = []
    for weight_matrix in network.weights:
        mutated_matrix = weight_matrix.copy()
        mask = np.random.rand(*mutated_matrix.shape) < MUTATION_RATE
        random_values = np.random.randn(*mutated_matrix.shape)
        mutated_matrix[mask] += random_values[mask]

        # Clip the mutated values to the range of -1 to 1
        mutated_matrix = np.clip(mutated_matrix, -1, 1)

        mutated_weights.append(mutated_matrix)

    mutated_network = Network(network.structure, mutated_weights)
    return mutated_network


best_network = None


def evolve(population, train_data):
    global best_network
    fitness_array = [(network, calculate_fitness(network, train_data)) for network in population]
    fitness_array.sort(key=lambda x: x[1], reverse=True)
    fitness = [f for network, f in fitness_array]

    # Elitism: Keep the best individual in the population
    best_network = fitness_array[0][0]
    print("best is: ", fitness_array[0][1])
    new_population = [network for network, f in fitness_array[:ELITE_SIZE]]

    while len(new_population) < len(population):
        parent1, parent2 = selection(population, fitness)
        # if np.random.uniform() < CROSSOVER_RATE:
        offspring = crossover(parent1, parent2)
        # else:
        #     offspring = parent1
        offspring = mutation(offspring)
        new_population.append(offspring)

    return new_population


# Main loop
population = initialize_population(POPULATION_SIZE, INPUT_SIZE, OUTPUT_SIZE)

for generation in range(NUM_GENERATIONS):
    print("generation number: ", generation + 1)
    population = evolve(population, train_data)


# Evaluate the best network on the test data
test_accuracy = calculate_fitness(best_network, test_data)
print("Test Accuracy:", test_accuracy)

# Write network structure and weights to file
with open("wnet0.txt", "w") as file:
    # Write layer sizes
    layer_sizes = [str(layer) for layer in best_network.structure]
    file.write(" ".join(layer_sizes) + "\n")

    # Write weights between layers
    for weights in best_network.weights:
        flattened_weights = weights.flatten()  # Flatten the weight matrix
        weight_str = " ".join([str(weight) for weight in flattened_weights])
        file.write(weight_str + "\n")
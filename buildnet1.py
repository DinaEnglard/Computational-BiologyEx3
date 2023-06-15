import random
import numpy as np
from sklearn.metrics import roc_auc_score

# Constants
POPULATION_SIZE = 100
NUM_GENERATIONS = 100
MUTATION_RATE = 0.1
CROSSOVER_RATE = 1
INPUT_SIZE = 16
OUTPUT_SIZE = 1
ELITE_SIZE = 5
BEST = (POPULATION_SIZE * 10) // 100


# Load data
# Parse the data from nn0.txt
data = []
with open('nn1.txt', 'r') as file:
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


def xavier_init(shape):
    """
    Xavier initialization for weight matrices.
    """
    n_inputs, n_outputs = shape[1], shape[0]
    limit = np.sqrt(6 / (n_inputs + n_outputs))
    return np.random.uniform(-limit, limit, shape)


# Neural network class
class Network:
    def __init__(self, structure, weights=None):
        self.structure = structure
        if weights is None:
            self.weights = [xavier_init((structure[i + 1], structure[i])) for i in range(len(structure) - 1)]
        else:
            self.weights = weights

    def predict(self, input_data):
        hidden = input_data
        for layer_weights in self.weights[:-1]:
            hidden = sigmoid(np.dot(layer_weights, hidden))
        output = sigmoid(np.dot(self.weights[-1], hidden))
        return output


# Genetic algorithm
def initialize_population(population_size, input_size, output_size):
    population = []
    for _ in range(population_size):
        structure = [input_size, 2, output_size]  # Randomly initialize network structure
        population.append(Network(structure))
    return population


def calculate_fitness(network, data):
    correct_predictions = 0
    for example in data:
        input_data = np.array(list(map(int, example[0])))
        target = int(example[1])
        output = network.predict(input_data)
        predicted_class = 1 if output > 0.5 else 0

        if predicted_class == target:
            correct_predictions += 1
    fitness = correct_predictions / len(data)
    print(fitness)
    return fitness


# def calculate_fitness(network, data):
#     predicted_probs = []
#     true_labels = []
#     i=0
#     for example in data:
#         input_data = np.array(list(map(int, example[0])))
#         target = int(example[1])
#         output = network.predict(input_data)
#         p = 1 if output > 0.5 else 0
#         if target == p:
#             i += 1
#         predicted_probs.append(output)
#         true_labels.append(target)
#     print(i/len(data))
#     fitness = roc_auc_score(true_labels, predicted_probs)
#     return fitness


# def calculate_fitness(network, data, roc_weight=0.7, fp_penalty=0.1):
#     predicted_probs = []
#     true_labels = []
#     correct_predictions = 0
#     false_positives = 0
#
#     for example in data:
#         input_data = np.array(list(map(int, example[0])))
#         target = int(example[1])
#         output = network.predict(input_data)
#         predicted_probs.append(output)
#         true_labels.append(target)
#
#         predicted_class = 1 if output > 0.5 else 0
#         if predicted_class == target:
#             correct_predictions += 1
#         elif predicted_class == 1 and target == 0:
#             false_positives += 1
#
#     roc_auc = roc_auc_score(true_labels, predicted_probs)
#     accuracy = correct_predictions / len(data)
#
#     fitness = roc_weight * roc_auc + (1 - roc_weight) * accuracy - fp_penalty * false_positives
#     fitness = max(fitness, 0)  # Ensure fitness is non-negative
#     return fitness

# def calculate_fitness(network, data):
#     predicted_probs = []
#     true_labels = []
#     correct_predictions = 0
#     false_positives = 0
#
#     for example in data:
#         input_data = np.array(list(map(int, example[0])))
#         target = int(example[1])
#         output = network.predict(input_data)
#         predicted_probs.append(output)
#         true_labels.append(target)
#
#         predicted_class = 1 if output > 0.5 else 0
#         if predicted_class == target:
#             correct_predictions += 1
#         elif predicted_class == 1 and target == 0:
#             false_positives += 1
#
#     roc_auc = roc_auc_score(true_labels, predicted_probs)
#     accuracy = correct_predictions / len(data)
#     print(accuracy)
#
#     # Define weights for each metric
#     weight_roc = 0.7
#     weight_accuracy = 0.3
#     fp_penalty = 0.1
#
#     fitness = weight_roc * roc_auc + weight_accuracy * accuracy - fp_penalty * false_positives
#     fitness = max(fitness, 0)  # Ensure fitness is non-negative
#
#     return fitness

def selection(population, fitness):
    parent_indices = np.random.choice(range(len(population)), size=2, replace=False, p=fitness / np.sum(fitness))
    parents = [population[idx] for idx in parent_indices]
    return parents[0], parents[1]


def crossover(parent1, parent2):
    offspring_weights_one = []
    offspring_weights_two = []
    for i in range(len(parent1.weights)):
        # Get the shape of the matrices
        rows, cols = parent1.weights[i].shape

        # Generate a random crossover point
        crossover_point = np.random.randint(1, cols)

        # Perform one-point crossover
        offspring_one = np.concatenate((parent1.weights[i][:, :crossover_point], parent2.weights[i][:, crossover_point:]),
                                   axis=1)
        offspring_weights_one.append(offspring_one)

        offspring_two = np.concatenate(
            (parent2.weights[i][:, :crossover_point], parent1.weights[i][:, crossover_point:]),
            axis=1)
        offspring_weights_two.append(offspring_two)

    child_one = Network(parent1.structure, offspring_weights_one)
    child_two = Network(parent1.structure, offspring_weights_two)

    return child_one, child_two

# def crossover(parent1, parent2, eta=1):
#     offspring_weights_one = []
#     offspring_weights_two = []
#     for i in range(len(parent1.weights)):
#         rows, cols = parent1.weights[i].shape
#         offspring_one = np.zeros((rows, cols))
#         offspring_two = np.zeros((rows, cols))
#
#         for j in range(rows):
#             for k in range(cols):
#                 if random.random() <= 0.5:
#                     beta = (2 * random.random()) ** (1.0 / (eta + 1))
#                 else:
#                     beta = (0.5 / (1 - random.random())) ** (1.0 / (eta + 1))
#
#                 offspring_one[j, k] = 0.5 * ((1 + beta) * parent1.weights[i][j, k] + (1 - beta) * parent2.weights[i][j, k])
#                 offspring_two[j, k] = 0.5 * ((1 - beta) * parent1.weights[i][j, k] + (1 + beta) * parent2.weights[i][j, k])
#
#         offspring_weights_one.append(offspring_one)
#         offspring_weights_two.append(offspring_two)
#
#     child_one = Network(parent1.structure, offspring_weights_one)
#     child_two = Network(parent1.structure, offspring_weights_two)
#
#     return child_one, child_two


# def mutation(network):
#     mutated_weights = []
#     for weight_matrix in network.weights:
#         mutated_matrix = weight_matrix.copy()
#         mask = np.random.rand(*mutated_matrix.shape) < MUTATION_RATE
#         random_values = np.random.randn(*mutated_matrix.shape)
#         mutated_matrix[mask] += random_values[mask]
#
#         # Clip the mutated values to the range of -1 to 1
#         mutated_matrix = np.clip(mutated_matrix, -1, 1)
#
#         mutated_weights.append(mutated_matrix)
#
#     mutated_network = Network(network.structure, mutated_weights)
#     # if calculate_fitness(network, train_data) > calculate_fitness(mutated_network,train_data):
#     #     return network
#     return mutated_network
#
#
# best_network = None

def mutation(network, mutation_rate=0.1, distribution_index=20):
    mutated_weights = []
    for weight_matrix in network.weights:
        mutated_matrix = weight_matrix.copy()
        rows, cols = mutated_matrix.shape
        for i in range(rows):
            for j in range(cols):
                if random.random() <= MUTATION_RATE:
                    u = random.random()
                    if u <= 0.5:
                        delta = (2 * u) ** (1.0 / (distribution_index + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1.0 / (distribution_index + 1))

                    mutated_matrix[i, j] += delta

                    # Clip the mutated value to the range of -1 to 1
                    mutated_matrix[i, j] = np.clip(mutated_matrix[i, j], -1, 1)

        mutated_weights.append(mutated_matrix)

    mutated_network = Network(network.structure, mutated_weights)
    return mutated_network


def evolve(population, train_data):
    global best_network
    fitness_array = [(network, calculate_fitness(network, train_data)) for network in population]
    fitness_array.sort(key=lambda x: x[1], reverse=True)
    fitness_array = fitness_array[:-BEST] + [fitness_array[0]]*BEST
    fitness_array.sort(key=lambda x: x[1], reverse=True)
    population = [network for network, f in fitness_array]
    fitness = [f for network, f in fitness_array]

    # Elitism: Keep the best individual in the population
    best_network = fitness_array[0][0]
    print("best is: ", fitness_array[0][1])
    new_population = [network for network, f in fitness_array[:ELITE_SIZE]]

    while len(new_population) < len(population):
        parent1, parent2 = selection(population, fitness)
        if np.random.uniform() < CROSSOVER_RATE:
            offspring_one, offspring_two = crossover(parent1, parent2)
            offspring_one = mutation(offspring_one)
            offspring_two = mutation(offspring_two)
            new_population.append(offspring_one)
            new_population.append(offspring_two)
        else:
            offspring = parent1
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
with open("wnet1.txt", "w") as file:
    # Write layer sizes
    layer_sizes = [str(layer) for layer in best_network.structure]
    file.write(" ".join(layer_sizes) + "\n")

    # Write weights between layers
    for weights in best_network.weights:
        flattened_weights = weights.flatten()  # Flatten the weight matrix
        weight_str = " ".join([str(weight) for weight in flattened_weights])
        file.write(weight_str + "\n")

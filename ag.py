import random
import string

# Frase objetivo
TARGET = "hola mundo"
POPULATION_SIZE = 100
MUTATION_RATE = 0.1
GENERATIONS = 1000

# Función para generar una cadena aleatoria del mismo tamaño que la meta
def random_phrase(length):
    chars = string.ascii_lowercase + " "
    return ''.join(random.choice(chars) for _ in range(length))

# Función de aptitud: entre más caracteres correctos en la posición correcta, mayor el puntaje
def fitness(individual):
    return sum(1 for a, b in zip(individual, TARGET) if a == b)

# Cruza entre dos padres
def crossover(parent1, parent2):
    child = ''
    for i in range(len(TARGET)):
        child += random.choice([parent1[i], parent2[i]])
    return child

# Mutación aleatoria
def mutate(individual):
    chars = string.ascii_lowercase + " "
    return ''.join(
        c if random.random() > MUTATION_RATE else random.choice(chars)
        for c in individual
    )

# Selección por ruleta
def roulette_selection(population, fitnesses):
    total_fitness = sum(fitnesses)
    if total_fitness == 0:
        return random.choice(population)
    pick = random.uniform(0, total_fitness)
    current = 0
    for individual, fit in zip(population, fitnesses):
        current += fit
        if current > pick:
            return individual

# Selección por torneo binario
def tournament_selection(population, fitnesses):
    i1, i2 = random.sample(range(len(population)), 2)
    return population[i1] if fitnesses[i1] > fitnesses[i2] else population[i2]

# Inicializar población
population = [random_phrase(len(TARGET)) for _ in range(POPULATION_SIZE)]

# Algoritmo genético
for generation in range(GENERATIONS):
    fitnesses = [fitness(ind) for ind in population]
    population = sorted(population, key=lambda x: fitness(x), reverse=True)
    if population[0] == TARGET:
        print(f"Generación {generation}: {population[0]} (¡Encontrado!)")
        break
    
    next_generation = population[:50]  # Elite
    while len(next_generation) < POPULATION_SIZE:
        #parent1, parent2 = random.choices(population[:50], k=2)
        parent1 = roulette_selection(population, fitnesses)
        parent2 = roulette_selection(population, fitnesses)
        #parent1 = tournament_selection(population, fitnesses)
        #parent2 = tournament_selection(population, fitnesses)
        child = mutate(crossover(parent1, parent2))
        next_generation.append(child)
    population = next_generation
    print(f"Gen {generation} Mejor: {population[0]}")


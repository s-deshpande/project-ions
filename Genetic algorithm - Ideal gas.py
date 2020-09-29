import random
# In this code, we attempt to use genetic algorithms to find variables in an equation. Please note that this code isnt in its
# final form and steps such as optimizing parameters or introducing reinforcement learning can help achieve perfect results.

# Here given a value P i.e. pressure, we try to find V, n and T that lie in a boundary. This code using the ideal gas equation
# but can be changed for any equation. To get better values of the variables, run the code again and again.

def generate_pop(size,V_boundary,n_boundary,T_boundary):
    V_lower_boundary, V_upper_boundary = V_boundary
    n_lower_boundary, n_upper_boundary = n_boundary
    T_lower_boundary, T_upper_boundary = T_boundary

    population = []
    for i in range(size):
        individual = {
            'V' : random.uniform(V_lower_boundary,V_upper_boundary),
            'n': random.uniform(n_lower_boundary,n_upper_boundary),
            'T': random.uniform(T_lower_boundary, T_upper_boundary)
        }
        population.append(individual)
    return population

def function(individual,P):
    """Calculates the function we want to find the max of/ also counts as fitness"""
    V = individual['V']
    n = individual['n']
    T = individual['T']

    P_calc = (n*T*8.8314)/V
    fitness = P- P_calc
    return fitness

def sort_pop_by_fitness(population):
    sorted_pop = sorted(population , key= function)
    return sorted_pop

def crossover(sorted_pop):
    individual_a = sorted_pop[-1]
    individual_b = sorted_pop[-2]
    Va = individual_a['V']
    na = individual_a['n']
    Ta = individual_a['T']

    Vb = individual_b['x']
    nb = individual_b['y']
    Tb = individual_b['T']

    return {'V': (Va + Vb) / 3, 'n': (na + nb) / 3, 'T': (Ta + Tb) / 2}

def mutation(individual,V_boundary,n_boundary,T_boundary):
    next_V = individual['V'] + random.uniform(-0.5, 0.5)
    next_n = individual['n'] + random.uniform(-0.3, 0.3)
    next_T = individual['T'] + random.uniform(-10, 10)

    V_lower_boundary, V_upper_boundary = V_boundary
    n_lower_boundary, n_upper_boundary = n_boundary
    T_lower_boundary, T_upper_boundary = T_boundary
    # Guarantee its within the boundaries
    next_V = min(max(next_V,V_lower_boundary), V_upper_boundary)
    next_n = min(max(next_n, n_lower_boundary), n_upper_boundary)
    next_T = min(max(next_T, T_lower_boundary), T_upper_boundary)

    return {'V': next_V, 'n': next_n, 'T' : next_T}

def make_next_gen(previous_population, individual,population):
    next_gen = []
    pop_size = len(previous_population)

    for i in range(pop_size):
        individual = crossover(sort_pop_by_fitness(population))
        individual= mutation(individual)
        next_gen.append(individual)
    return next_gen

generations = 10000

population = generate_pop(size=30, V_boundary=(1, 10), n_boundary=(0.1, 10), T_boundary= (1, 100))
i = 1
P = 100
best_scores = []
while True:
    print('Generation:', i)
    for individual in population:
        print(individual)
        print(function(individual,P))
        if function(individual, P) == 0:
            print('The solution is:' + function(individual, P))
            break
        elif (function(individual, P) >= -10) and (function(individual, P) <= 20):
            print('The solution is close to :', function(individual, P), 'The variables are:',  individual)
            break
    i = i+1
    if i == generations:
        break


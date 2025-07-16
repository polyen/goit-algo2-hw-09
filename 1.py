import random
import math

import numpy as np


# Визначення функції Сфери
def sphere_function(x):
    return sum(xi ** 2 for xi in x)


def limit_by_bounds(point, bounds):
    return [max(min(coord, bounds[i][1]), bounds[i][0]) for i, coord in enumerate(point)]


# Hill Climbing
def hill_climbing(func, bounds, iterations=1000, epsilon=1e-6):
    step_size = 0.1

    def get_neighbors(current, step_size=0.1):
        x, y = current

        x = max(min(x, bounds[0][1]), bounds[0][0])
        y = max(min(y, bounds[1][1]), bounds[1][0])

        return [
            limit_by_bounds([x + step_size, y], bounds),
            limit_by_bounds([x - step_size, y], bounds),
            limit_by_bounds([x, y + step_size], bounds),
            limit_by_bounds([x, y - step_size], bounds),
        ]

    current_point = [random.uniform(bounds[0][0], bounds[0][1]),
                     random.uniform(bounds[1][0], bounds[1][1])]
    current_value = func(current_point)

    for iteration in range(iterations):
        neighbors = get_neighbors(current_point, step_size)

        # Пошук найкращого сусіда
        next_point = None
        next_value = np.inf

        for neighbor in neighbors:
            value = func(neighbor)
            if value < next_value:
                next_point = neighbor
                next_value = value

        # Якщо не вдається знайти кращого сусіда — зупиняємось
        if abs(next_value - current_value) < epsilon:
            break

        # Переходимо до кращого сусіда
        current_point, current_value = next_point, next_value

    return current_point, current_value


# Random Local Search
def random_local_search(func, bounds, iterations=1000, epsilon=1e-6):
    step_size = 0.1
    probability = 0.2

    def get_random_neighbor(current, step_size=0.1):
        x, y = current
        new_x = x + random.uniform(-step_size, step_size)
        new_y = y + random.uniform(-step_size, step_size)
        return limit_by_bounds([new_x, new_y], bounds)

    current_point = [random.uniform(bounds[0][0], bounds[0][1]),
                     random.uniform(bounds[1][0], bounds[1][1])]
    current_value = func(current_point)

    for iteration in range(iterations):
        # Отримання випадкового сусіда
        new_point = get_random_neighbor(current_point, step_size)
        new_value = func(new_point)

        if abs(new_value - current_value) < epsilon:
            break

        # Перевірка умови переходу
        if new_value < current_value or random.random() < probability:
            current_point, current_value = new_point, new_value

    return current_point, current_value


# Simulated Annealing
def simulated_annealing(func, bounds, iterations=1000, temp=1000, cooling_rate=0.95, epsilon=1e-6):
    def generate_neighbor(solution):
        x, y = solution
        new_x = x + random.uniform(-1, 1)
        new_y = y + random.uniform(-1, 1)
        return limit_by_bounds([new_x, new_y], bounds)

    current_solution = [random.uniform(bounds[0][0], bounds[0][1]),
                        random.uniform(bounds[1][0], bounds[1][1])]
    current_energy = func(current_solution)

    while temp > epsilon and iterations > 0:
        new_solution = generate_neighbor(current_solution)
        new_energy = func(new_solution)
        delta_energy = new_energy - current_energy

        if delta_energy < 0 or random.random() < math.exp(-delta_energy / temp):
            current_solution = new_solution
            current_energy = new_energy

        temp *= cooling_rate
        iterations -= 1

    return current_solution, current_energy


if __name__ == "__main__":
    # Межі для функції
    bounds = [(-5, 5), (-5, 5)]

    # Виконання алгоритмів
    print("Hill Climbing:")
    hc_solution, hc_value = hill_climbing(sphere_function, bounds)
    print("Розв'язок:", hc_solution, "Значення:", hc_value)

    print("\nRandom Local Search:")
    rls_solution, rls_value = random_local_search(sphere_function, bounds)
    print("Розв'язок:", rls_solution, "Значення:", rls_value)

    print("\nSimulated Annealing:")
    sa_solution, sa_value = simulated_annealing(sphere_function, bounds)
    print("Розв'язок:", sa_solution, "Значення:", sa_value)

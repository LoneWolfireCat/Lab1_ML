import numpy as np
import matplotlib.pyplot as plt
import random
from typing import List, Tuple, Callable
import time


class LogisticsOptimizer:
    def __init__(self, n_prod: int, k_cities: int, budget: float):
        self.n_prod = n_prod
        self.k_cities = k_cities
        self.budget = budget
        self.gen_data()

    def gen_data(self):
        """Генерация реалистичных данных"""
        # Производственные мощности (100-500 единиц)
        self.supply = np.random.randint(100, 500, self.n_prod)
        # Потребности городов (50-300 единиц)
        self.demand = np.random.randint(50, 300, self.k_cities)

        # Матрица расстояний (50-400 км)
        self.distances = np.random.randint(50, 400, (self.n_prod, self.k_cities))

        # Стоимость перевозки (10-20 руб/км)
        transport_cost = np.random.uniform(10, 20, (self.n_prod, self.k_cities))
        self.cost_matrix = self.distances * transport_cost

        # Балансировка спроса и предложения
        total_demand = np.sum(self.demand)
        total_supply = np.sum(self.supply)

        if total_demand > total_supply:
            scale = total_demand / total_supply * 1.15  # Запас 15%
            self.supply = (self.supply * scale).astype(int)

    def create_individual(self) -> np.ndarray:
        """Создание случайной особи с учетом ограничений"""
        individual = np.zeros((self.n_prod, self.k_cities), dtype=int)
        remaining_supply = self.supply.copy()
        remaining_demand = self.demand.copy()

        # Случайное распределение с приоритетом на удовлетворение спроса
        for _ in range(self.n_prod * self.k_cities):
            i = random.randint(0, self.n_prod - 1)
            j = random.randint(0, self.k_cities - 1)

            if remaining_supply[i] > 0 and remaining_demand[j] > 0:
                max_possible = min(remaining_supply[i], remaining_demand[j])
                delivery = random.randint(0, max_possible)

                individual[i, j] += delivery
                remaining_supply[i] -= delivery
                remaining_demand[j] -= delivery

        return individual

    def calculate_fitness(self, individual: np.ndarray) -> float:
        """Расчет функции приспособленности"""
        total_cost = np.sum(individual * self.cost_matrix)

        # Расчет превышения поставок
        city_supply = np.sum(individual, axis=0)
        excess_supply = np.sum(np.maximum(0, city_supply - self.demand))

        # Расчет неудовлетворенного спроса
        unsatisfied_demand = np.sum(np.maximum(0, self.demand - city_supply))

        # Расчет перепроизводства
        production_used = np.sum(individual, axis=1)
        overproduction = np.sum(np.maximum(0, production_used - self.supply))

        # Штрафы
        cost_penalty = max(0, total_cost - self.budget) * 100
        demand_penalty = unsatisfied_demand * 200
        production_penalty = overproduction * 100

        # Функция приспособленности (минимизация штрафов)
        fitness = 1 / (1 + excess_supply + cost_penalty + demand_penalty + production_penalty)

        return fitness

    # Методы скрещивания
    def single_point_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        flat1 = parent1.flatten()
        flat2 = parent2.flatten()

        point = random.randint(1, len(flat1) - 1)

        child1 = np.concatenate([flat1[:point], flat2[point:]])
        child2 = np.concatenate([flat2[:point], flat1[point:]])

        return (child1.reshape(parent1.shape),
                child2.reshape(parent2.shape))

    def two_point_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        flat1 = parent1.flatten()
        flat2 = parent2.flatten()

        point1 = random.randint(1, len(flat1) - 2)
        point2 = random.randint(point1 + 1, len(flat1) - 1)

        child1 = np.concatenate([flat1[:point1], flat2[point1:point2], flat1[point2:]])
        child2 = np.concatenate([flat2[:point1], flat1[point1:point2], flat2[point2:]])

        return (child1.reshape(parent1.shape),
                child2.reshape(parent2.shape))

    def uniform_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mask = np.random.random(parent1.shape) < 0.5

        child1 = np.where(mask, parent1, parent2)
        child2 = np.where(mask, parent2, parent1)

        return child1, child2

    # Методы мутации
    def random_mutation(self, individual: np.ndarray, mutation_rate: float = 0.1) -> np.ndarray:
        mutated = individual.copy()

        for i in range(self.n_prod):
            for j in range(self.k_cities):
                if random.random() < mutation_rate:
                    max_val = min(self.supply[i], self.demand[j])
                    mutated[i, j] = random.randint(0, max_val)

        return mutated

    def swap_mutation(self, individual: np.ndarray, mutation_rate: float = 0.1) -> np.ndarray:
        mutated = individual.copy()

        if random.random() < mutation_rate:
            # Выбираем две случайные позиции для обмена
            i1, j1 = random.randint(0, self.n_prod - 1), random.randint(0, self.k_cities - 1)
            i2, j2 = random.randint(0, self.n_prod - 1), random.randint(0, self.k_cities - 1)

            mutated[i1, j1], mutated[i2, j2] = mutated[i2, j2], mutated[i1, j1]

        return mutated

    def adaptive_mutation(self, individual: np.ndarray, mutation_rate: float = 0.1) -> np.ndarray:
        mutated = individual.copy()

        # Анализ эффективности маршрутов
        efficiency = individual / (self.cost_matrix + 1e-10)
        avg_efficiency = np.mean(efficiency)

        for i in range(self.n_prod):
            for j in range(self.k_cities):
                current_efficiency = efficiency[i, j]

                # Адаптивная вероятность мутации
                adaptive_rate = mutation_rate * (1 - current_efficiency / (avg_efficiency + 1e-10))

                if random.random() < adaptive_rate:
                    max_val = min(self.supply[i], self.demand[j])
                    mutated[i, j] = random.randint(0, max_val)

        return mutated

    def tournament_selection(self, population: List[np.ndarray], fitnesses: List[float],
                             tournament_size: int = 3) -> np.ndarray:
        participants = random.sample(list(zip(population, fitnesses)), tournament_size)
        best_individual = max(participants, key=lambda x: x[1])[0]
        return best_individual.copy()

    def genetic_algorithm(self, pop_size: int = 50, generations: int = 200,
                          crossover_func: Callable = None, mutation_func: Callable = None,
                          crossover_rate: float = 0.8, mutation_rate: float = 0.1) -> dict:
        """Основной генетический алгоритм"""

        # Инициализация популяции
        population = [self.create_individual() for _ in range(pop_size)]
        best_fitness_history = []
        avg_fitness_history = []

        for generation in range(generations):
            # Расчет приспособленности
            fitnesses = [self.calculate_fitness(ind) for ind in population]

            # Сохранение статистики
            best_fitness = max(fitnesses)
            avg_fitness = np.mean(fitnesses)
            best_fitness_history.append(best_fitness)
            avg_fitness_history.append(avg_fitness)

            # Создание нового поколения
            new_population = []

            # Элитизм - сохранение лучшей особи
            best_idx = np.argmax(fitnesses)
            new_population.append(population[best_idx].copy())

            # Заполнение остальной части популяции
            while len(new_population) < pop_size:
                # Селекция
                parent1 = self.tournament_selection(population, fitnesses)
                parent2 = self.tournament_selection(population, fitnesses)

                # Скрещивание
                if random.random() < crossover_rate and crossover_func:
                    child1, child2 = crossover_func(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()

                # Мутация
                if mutation_func:
                    child1 = mutation_func(child1, mutation_rate)
                    child2 = mutation_func(child2, mutation_rate)

                new_population.extend([child1, child2])

            # Обновление популяции
            population = new_population[:pop_size]

            if generation % 50 == 0:
                print(f"Поколение {generation}: Лучшая приспособленность = {best_fitness:.6f}")

        # Поиск лучшего решения
        final_fitnesses = [self.calculate_fitness(ind) for ind in population]
        best_idx = np.argmax(final_fitnesses)
        best_solution = population[best_idx]

        return {
            'solution': best_solution,
            'best_fitness': best_fitness_history,
            'avg_fitness': avg_fitness_history,
            'total_cost': np.sum(best_solution * self.cost_matrix),
            'total_excess': np.sum(np.maximum(0, np.sum(best_solution, axis=0) - self.demand))
        }


def run_experiments():
    """Запуск экспериментов с разными методами скрещивания и мутации"""
    # Параметры задачи
    n_prod = 4
    k_cities = 6
    budget = 1000000

    optimizer = LogisticsOptimizer(n_prod, k_cities, budget)

    # Определение методов
    crossover_methods = {
        'Одноточечное': optimizer.single_point_crossover,
        'Двухточечное': optimizer.two_point_crossover,
        'Равномерное': optimizer.uniform_crossover
    }

    mutation_methods = {
        'Случайная': optimizer.random_mutation,
        'Обмен': optimizer.swap_mutation,
        'Адаптивная': optimizer.adaptive_mutation
    }

    results = {}

    # Проведение экспериментов
    for cross_name, cross_func in crossover_methods.items():
        for mut_name, mut_func in mutation_methods.items():
            key = f"{cross_name} + {mut_name}"
            print(f"\nЗапуск: {key}")

            start_time = time.time()
            results[key] = optimizer.genetic_algorithm(
                pop_size=50,
                generations=150,
                crossover_func=cross_func,
                mutation_func=mut_func,
                crossover_rate=0.8,
                mutation_rate=0.1
            )
            execution_time = time.time() - start_time
            print(f"Завершено за {execution_time:.2f} сек")

    # Визуализация результатов
    plot_convergence(results)


def plot_convergence(results: dict):
    """Построение графиков сходимости"""
    plt.figure(figsize=(15, 5))

    # График лучшей приспособленности
    plt.subplot(1, 2, 1)
    for method, result in results.items():
        plt.plot(result['best_fitness'], label=method, linewidth=2)

    plt.title('Сходимость генетического алгоритма\n(Лучшая приспособленность)', fontsize=12)
    plt.xlabel('Поколение')
    plt.ylabel('Лучшая приспособленность')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # График средней приспособленности
    plt.subplot(1, 2, 2)
    for method, result in results.items():
        plt.plot(result['avg_fitness'], label=method, linewidth=2)

    plt.title('Сходимость генетического алгоритма\n(Средняя приспособленность)', fontsize=12)
    plt.xlabel('Поколение')
    plt.ylabel('Средняя приспособленность')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Вывод итоговых результатов
    print("\n" + "=" * 70)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТОВ")
    print("=" * 70)

    for method, result in results.items():
        best_fitness = max(result['best_fitness'])
        final_cost = result['total_cost']
        final_excess = result['total_excess']

        print(f"\n{method}:")
        print(f"  Лучшая приспособленность: {best_fitness:.6f}")
        print(f"  Общая стоимость: {final_cost:.2f} руб")
        print(f"  Превышение поставок: {final_excess:.2f} ед.")


if __name__ == "__main__":
    run_experiments()

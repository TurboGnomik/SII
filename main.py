import numpy as np
import matplotlib.pyplot as plt
import random

# Параметры задачи: n пунктов производства, k городов
N_PRODUCTION_POINTS = 5
N_CITIES = 4

# Производственные мощности пунктов
production_capacity = np.array([100, 150, 120, 200, 180])
# Потребности городов
city_demand = np.array([80, 130, 90, 160])
# Стоимость перевозки от пункта i к городу j
transport_costs = np.array([
    [5, 8, 6, 10],
    [7, 5, 9, 6],
    [8, 7, 5, 8],
    [6, 9, 7, 5],
    [9, 6, 8, 7]
])

# Параметры генетического алгоритма
ALPHA, BETA = 0.7, 0.3  # Веса для превышения поставок и транспортных расходов
MUTATION_PROBABILITY = 0.4
CROSSOVER_PROBABILITY = 0.8
BUDGET_LIMIT = 2500  # Лимит транспортных расходов

class Individual:
    def __init__(self, genome):
        self.genome = genome  # Матрица n x k: genome[i][j] - количество продуктов от пункта i к городу j
        self.fitness, self.total_excess, self.total_cost, self.demand_satisfaction = self.calculate_fitness()
    
    def calculate_fitness(self):
        # Общая стоимость перевозок
        total_cost = np.sum(self.genome * transport_costs)
        
        # Поставки в каждый город
        deliveries_to_cities = np.sum(self.genome, axis=0)
        
        # Превышение поставок над спросом (только положительные значения)
        excess_delivery = np.sum(np.maximum(0, deliveries_to_cities - city_demand))
        
        # Удовлетворение спроса (отрицательные значения - недопоставка)
        demand_diff = deliveries_to_cities - city_demand
        demand_satisfaction = np.sum(np.where(demand_diff >= 0, city_demand, deliveries_to_cities))
        
        # Штраф за превышение бюджета
        budget_penalty = max(0, total_cost - BUDGET_LIMIT) * 10
        
        # Проверка ограничений производства
        production_used = np.sum(self.genome, axis=1)
        production_excess = np.sum(np.maximum(0, production_used - production_capacity))
        
        # Функция приспособленности: максимизируем удовлетворение спроса, минимизируем превышение и стоимость
        max_demand = np.sum(city_demand)
        max_cost = np.sum(production_capacity) * np.max(transport_costs)
        
        norm_satisfaction = demand_satisfaction / max_demand
        norm_excess = 1 - (excess_delivery / (max_demand * 2))  # Нормализация превышения
        norm_cost = 1 - (total_cost + budget_penalty) / (max_cost * 1.5)
        
        # Итоговая приспособленность
        fitness = (ALPHA * norm_satisfaction + 
                  BETA * norm_excess * 0.7 + 
                  BETA * norm_cost * 0.3)
        
        return fitness, excess_delivery, total_cost, demand_satisfaction / max_demand

def create_random_individual():
    # Создаем случайную матрицу поставок
    genome = np.zeros((N_PRODUCTION_POINTS, N_CITIES))
    
    for i in range(N_PRODUCTION_POINTS):
        # Распределяем продукцию пункта i между городами
        remaining_capacity = production_capacity[i]
        for j in range(N_CITIES):
            if remaining_capacity > 0:
                # Случайное количество, но не более оставшейся мощности и потребности города
                max_possible = min(remaining_capacity, city_demand[j] * 1.5)
                if max_possible > 0:
                    delivery = random.uniform(0, max_possible)
                    genome[i][j] = delivery
                    remaining_capacity -= delivery
    
    return Individual(genome)

def select_parents(population):
    candidates = random.sample(population, 5)
    candidates.sort(key=lambda x: x.fitness, reverse=True)
    return candidates[:2]

# Операторы скрещивания
def matrix_crossover(parent1, parent2):
    # Одноточечное скрещивание для матриц
    point_i = random.randint(1, N_PRODUCTION_POINTS - 1)
    point_j = random.randint(1, N_CITIES - 1)
    
    child1 = parent1.genome.copy()
    child2 = parent2.genome.copy()
    
    # Обмен квадратами
    child1[point_i:, point_j:] = parent2.genome[point_i:, point_j:]
    child2[point_i:, point_j:] = parent1.genome[point_i:, point_j:]
    
    return Individual(child1), Individual(child2)

def uniform_matrix_crossover(parent1, parent2):
    # Равномерное скрещивание
    mask = np.random.random((N_PRODUCTION_POINTS, N_CITIES)) > 0.5
    child1 = np.where(mask, parent1.genome, parent2.genome)
    child2 = np.where(mask, parent2.genome, parent1.genome)
    
    return Individual(child1), Individual(child2)

def arithmetic_crossover(parent1, parent2):
    # Арифметическое скрещивание (усреднение)
    alpha = random.random()
    child1 = alpha * parent1.genome + (1 - alpha) * parent2.genome
    child2 = alpha * parent2.genome + (1 - alpha) * parent1.genome
    
    return Individual(child1), Individual(child2)

# Операторы мутации
def random_adjust_mutation(ind):
    genome = ind.genome.copy()
    i, j = random.randint(0, N_PRODUCTION_POINTS - 1), random.randint(0, N_CITIES - 1)
    
    # Случайная корректировка значения
    adjustment = random.uniform(-20, 20)
    genome[i][j] = max(0, genome[i][j] + adjustment)
    
    # Проверка ограничений производства
    production_used = np.sum(genome, axis=1)
    for idx in range(N_PRODUCTION_POINTS):
        if production_used[idx] > production_capacity[idx]:
            scale = production_capacity[idx] / production_used[idx]
            genome[idx] *= scale
    
    return Individual(genome)

def swap_routes_mutation(ind):
    genome = ind.genome.copy()
    
    # Обмен маршрутами между двумя случайными пунктами
    i1, i2 = random.sample(range(N_PRODUCTION_POINTS), 2)
    genome[i1], genome[i2] = genome[i2].copy(), genome[i1].copy()
    
    return Individual(genome)

def redistribute_mutation(ind):
    genome = ind.genome.copy()
    
    # Выбираем случайный пункт производства
    point_idx = random.randint(0, N_PRODUCTION_POINTS - 1)
    
    # Полностью перераспределяем его продукцию
    total_production = np.sum(genome[point_idx])
    genome[point_idx] = np.zeros(N_CITIES)
    
    remaining = total_production
    cities = list(range(N_CITIES))
    random.shuffle(cities)
    
    for j in cities:
        if remaining > 0:
            delivery = random.uniform(0, min(remaining, city_demand[j] * 1.2))
            genome[point_idx][j] = delivery
            remaining -= delivery
    
    return Individual(genome)

def evolve_population(population, crossover, mutation):
    new_pop = []
    while len(new_pop) < len(population):
        p1, p2 = select_parents(population)
        
        if random.random() < CROSSOVER_PROBABILITY:
            c1, c2 = crossover(p1, p2)
        else:
            c1, c2 = p1, p2
            
        if random.random() < MUTATION_PROBABILITY:
            c1 = mutation(c1)
        if random.random() < MUTATION_PROBABILITY:
            c2 = mutation(c2)
            
        new_pop.extend([c1, c2])
    
    return new_pop[:len(population)]

# Функция для красивого вывода матрицы поставок
def print_genome_matrix(genome, title="Матрица поставок"):
    print(f"\n{title}:")
    print("        " + "".join([f"Город {j+1:>10}" for j in range(N_CITIES)]))
    print("        " + "----------" * N_CITIES)
    
    for i in range(N_PRODUCTION_POINTS):
        row = [f"{genome[i][j]:>9.1f}" for j in range(N_CITIES)]
        print(f"Пункт {i+1}: " + "".join(row))
    
    # Вычисляем итоги по строкам и столбцам
    row_totals = np.sum(genome, axis=1)
    col_totals = np.sum(genome, axis=0)
    
    print("        " + "----------" * N_CITIES)
    print("Итого:  " + "".join([f"{col_totals[j]:>9.1f}" for j in range(N_CITIES)]))
    
    print("\nПоставки от пунктов:")
    for i in range(N_PRODUCTION_POINTS):
        print(f"  Пункт {i+1}: {row_totals[i]:.1f} из {production_capacity[i]}")
    
    print("\nПоставки в города:")
    for j in range(N_CITIES):
        print(f"  Город {j+1}: {col_totals[j]:.1f} из {city_demand[j]}")

def run_experiment(crossover, crossover_name, mutation, mutation_name, color):
    population = [create_random_individual() for _ in range(30)]
    best_over_time = []
    best_individual = None
    
    for generation in range(100):
        population = evolve_population(population, crossover, mutation)
        best_in_generation = max(population, key=lambda ind: ind.fitness)
        best_fitness = best_in_generation.fitness
        best_over_time.append(best_fitness)
        
        if best_individual is None or best_fitness > best_individual.fitness:
            best_individual = best_in_generation
    
    plt.plot(best_over_time, label=f"{crossover_name} + {mutation_name}", color=color)
    
    return best_individual, best_over_time

# Основной эксперимент
def main():
    # Определяем все комбинации операторов
    crossovers = [
        (matrix_crossover, "One point"),
        (uniform_matrix_crossover, "Uniform"), 
        (arithmetic_crossover, "Arithmetic")
    ]
    
    mutations = [
        (random_adjust_mutation, "Adjust"),
        (swap_routes_mutation, "Swap"),
        (redistribute_mutation, "Redistribute")
    ]
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan', 'gray']
    
    print("="*70)
    print("ГЕНЕТИЧЕСКИЙ АЛГОРИТМ ДЛЯ ЗАДАЧИ ТРАНСПОРТИРОВКИ")
    print("="*70)
    
    plt.figure(figsize=(14, 8))
    
    best_solutions = []
    color_idx = 0
    
    # Перебираем все комбинации кроссоверов и мутаций
    for crossover_func, crossover_name in crossovers:
        for mutation_func, mutation_name in mutations:
            if color_idx < len(colors):
                color = colors[color_idx]
                color_idx += 1
            else:
                color = np.random.rand(3,)
                
            print(f"\n{'='*60}")
            print(f"Комбинация: {crossover_name} + {mutation_name}")
            print(f"{'='*60}")
            
            best_solution, fitness_history = run_experiment(
                crossover_func, crossover_name, 
                mutation_func, mutation_name, 
                color
            )
            
            best_solutions.append((best_solution, crossover_name, mutation_name))
            
            print(f"Фитнес: {best_solution.fitness:.4f}")
            print(f"Удовлетворение спроса: {best_solution.demand_satisfaction*100:.1f}%")
            print(f"Превышение поставок: {best_solution.total_excess:.2f}")
            print(f"Транспортные расходы: {best_solution.total_cost:.2f}")
            
            # Выводим лучший геном для этой комбинации операторов
            print_genome_matrix(best_solution.genome, f"Лучший геном для {crossover_name} + {mutation_name}")
    
    plt.title("Сравнение различных комбинаций операторов генетического алгоритма")
    plt.xlabel("Поколение")
    plt.ylabel("Функция приспособленности")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    
    # Сохраняем график в PNG
    plt.savefig('genetic_algorithm_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Находим и выводим лучшее решение
    best_overall = max(best_solutions, key=lambda x: x[0].fitness)
    best_solution, best_crossover, best_mutation = best_overall
    
    print("\n" + "="*70)
    print("ЛУЧШЕЕ РЕШЕНИЕ ОБЩЕЕ:")
    print(f"Комбинация: {best_crossover} + {best_mutation}")
    print("="*70)
    
    print(f"Фитнес: {best_solution.fitness:.4f}")
    print(f"Удовлетворение спроса: {best_solution.demand_satisfaction*100:.1f}%")
    print(f"Превышение поставок: {best_solution.total_excess:.2f}")
    print(f"Транспортные расходы: {best_solution.total_cost:.2f}")
    
    # Выводим геном лучшего решения
    print_genome_matrix(best_solution.genome, f"Лучший геном (общий) для {best_crossover} + {best_mutation}")
    
    # Визуализация матрицы поставок лучшего решения
    plt.figure(figsize=(10, 6))
    plt.imshow(best_solution.genome, cmap='YlOrBr', aspect='auto')
    plt.colorbar(label='Количество продуктов')
    plt.xticks(range(N_CITIES), [f'Город {i+1} ({city_demand[i]})' for i in range(N_CITIES)])
    plt.yticks(range(N_PRODUCTION_POINTS), [f'Пункт {i+1} ({production_capacity[i]})' for i in range(N_PRODUCTION_POINTS)])
    plt.title(f'Матрица оптимальных поставок ({best_crossover} + {best_mutation})')
    plt.xlabel('Города')
    plt.ylabel('Пункты производства')
    
    # Добавление значений в ячейки
    for i in range(N_PRODUCTION_POINTS):
        for j in range(N_CITIES):
            plt.text(j, i, f'{best_solution.genome[i, j]:.1f}', 
                    ha='center', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('best_solution_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
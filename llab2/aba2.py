import numpy as np
import skfuzzy as fuzz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_triplet(prompt, default_values):
    while True:
        user_input = input(f"{prompt} (введите три числа через пробел, Enter = {default_values}): ").strip()
        if not user_input:
            return default_values
        parts = user_input.split()
        if len(parts) != 3:
            print("Нужно ввести ровно три числа — например: 20 40 60")
            continue
        try:
            a, b, c = map(float, parts)
        except ValueError:
            print("Введите только числа, без лишних символов.")
            continue
        if not (0 <= a <= b <= c <= 100):
            print("Каждое число должно быть в диапазоне от 0 до 100, и порядок должен быть a ≤ b ≤ c.")
            continue
        return (a, b, c)


def read_numeric_value(prompt, default_value):
    while True:
        user_input = input(f"{prompt} (0–100, Enter = {default_value}): ").strip()
        if not user_input:
            return default_value
        try:
            value = float(user_input)
        except ValueError:
            print("Нужно ввести число, например 45.")
            continue
        if not (0 <= value <= 100):
            print("Число должно быть в диапазоне от 0 до 100.")
            continue
        return value


def build_membership_functions(x_values, default_params):
    print("\nСейчас зададим параметры треугольных функций принадлежности (a ≤ b ≤ c).")
    print("Они описывают, где начинается, где центр и где заканчивается каждая категория.\n")

    params_low = read_triplet("Параметры для низкого уровня", default_params[0])
    params_medium = read_triplet("Параметры для среднего уровня", default_params[1])
    params_high = read_triplet("Параметры для высокого уровня", default_params[2])

    membership_low = fuzz.trimf(x_values, params_low)
    membership_medium = fuzz.trimf(x_values, params_medium)
    membership_high = fuzz.trimf(x_values, params_high)

    return (membership_low, membership_medium, membership_high), (params_low, params_medium, params_high)


def fuzzy_label(value, x_values, membership_dict):
    memberships = {name: fuzz.interp_membership(x_values, mf, value) for name, mf in membership_dict.items()}
    return max(memberships, key=memberships.get), memberships


def calculate_implication(soil_value, growth_value, soil_memberships, growth_memberships, result_memberships, result_x):
    soil_degrees = {
        "dry": fuzz.interp_membership(soil_x, soil_memberships[0], soil_value),
        "normal": fuzz.interp_membership(soil_x, soil_memberships[1], soil_value),
        "wet": fuzz.interp_membership(soil_x, soil_memberships[2], soil_value),
    }

    growth_degrees = {
        "slow": fuzz.interp_membership(growth_x, growth_memberships[0], growth_value),
        "medium": fuzz.interp_membership(growth_x, growth_memberships[1], growth_value),
        "fast": fuzz.interp_membership(growth_x, growth_memberships[2], growth_value),
    }

    activations = []

    rules = {
        ('dry', 'slow'): 'low',
        ('dry', 'medium'): 'medium',
        ('dry', 'fast'): 'medium',
        ('normal', 'slow'): 'medium',
        ('normal', 'medium'): 'medium',
        ('normal', 'fast'): 'high',
        ('wet', 'slow'): 'medium',
        ('wet', 'medium'): 'high',
        ('wet', 'fast'): 'high',
    }

    for (soil_level, growth_level), result_level in rules.items():
        rule_strength = np.fmin(soil_degrees[soil_level], growth_degrees[growth_level])
        if result_level == 'low':
            activations.append(np.fmin(rule_strength, result_memberships[0]))
        elif result_level == 'medium':
            activations.append(np.fmin(rule_strength, result_memberships[1]))
        else:
            activations.append(np.fmin(rule_strength, result_memberships[2]))

    aggregated_result = np.fmax.reduce(activations)
    return aggregated_result


soil_x = np.arange(0, 101, 1)
growth_x = np.arange(0, 101, 1)
result_x = np.arange(0, 101, 1)

default_soil_params = [(0, 20, 40), (30, 50, 70), (60, 80, 100)]
default_growth_params = [(0, 30, 60), (40, 60, 80), (70, 85, 100)]
default_result_params = [(0, 25, 50), (40, 60, 80), (70, 85, 100)]

print("Добро пожаловать в систему нечеткой логики по теме 'Агрономия'!")
print("Мы оценим, как уровень влажности почвы и скорость роста растения влияют на общий показатель.\n")

print("=== Настройка функций принадлежности для показателя 'Влажность почвы' ===")
(soil_memberships, soil_params) = build_membership_functions(soil_x, default_soil_params)

print("\n=== Настройка функций принадлежности для показателя 'Скорость роста растения' ===")
(growth_memberships, growth_params) = build_membership_functions(growth_x, default_growth_params)

print("\n=== Настройка функций принадлежности для 'Результата импликации' (итоговой оценки) ===")
(result_memberships, result_params) = build_membership_functions(result_x, default_result_params)

print("\nТеперь введём конкретные значения (от 0 до 100). Если не хотите вводить — просто нажмите Enter, и возьмутся значения по умолчанию.")
soil_value = read_numeric_value("Введите уровень влажности почвы", 55)
growth_value = read_numeric_value("Введите скорость роста растения", 60)

aggregated_result = calculate_implication(soil_value, growth_value, soil_memberships, growth_memberships, result_memberships, result_x)
crisp_result = fuzz.defuzz(result_x, aggregated_result, 'centroid')

soil_label, soil_members = fuzzy_label(soil_value, soil_x, {
    "Сухо": soil_memberships[0],
    "Умеренно влажно": soil_memberships[1],
    "Влажно": soil_memberships[2]
})
growth_label, growth_members = fuzzy_label(growth_value, growth_x, {
    "Медленный рост": growth_memberships[0],
    "Средний рост": growth_memberships[1],
    "Быстрый рост": growth_memberships[2]
})
result_label, result_members = fuzzy_label(crisp_result, result_x, {
    "Низкий результат": result_memberships[0],
    "Средний результат": result_memberships[1],
    "Высокий результат": result_memberships[2]
})

print("\nРЕЗУЛЬТАТ ОЦЕНКИ")
print(f"Влажность почвы ({soil_value:.1f}) → {soil_label}")
print(f"Скорость роста ({growth_value:.1f}) → {growth_label}")
print(f"Итоговая оценка ({crisp_result:.2f}) → {result_label}")

plt.figure(figsize=(8, 4))
plt.plot(result_x, result_memberships[0], '--', label='Низкий результат')
plt.plot(result_x, result_memberships[1], '--', label='Средний результат')
plt.plot(result_x, result_memberships[2], '--', label='Высокий результат')
plt.fill_between(result_x, 0, aggregated_result, color='orange', alpha=0.6, label='Импликация')
plt.axvline(crisp_result, color='k', linestyle=':', label=f'crisp = {crisp_result:.2f}')
plt.title(f'Импликация: влажность={soil_value}, рост={growth_value}')
plt.xlabel('Результат (итоговая оценка)')
plt.ylabel('Степень принадлежности')
plt.legend()
plt.savefig(f"agro_result_{int(soil_value)}_{int(growth_value)}.png")

print(f"\nГотово! График сохранён в файл: agro_result_{int(soil_value)}_{int(growth_value)}.png\n")


import pandas as pd
import numpy as np
import time
import random
import multiprocessing
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

# Imports de tu librería
from customhys import benchmark_func as bf
from customhys import metaheuristic as mh
from ml_utils import mittag_leffler 


DATA = None
T_DATA = None
Y_DATA = None

def init_data(data_file):
    global DATA, T_DATA, Y_DATA
    DATA = pd.read_csv(data_file)
    T_DATA = DATA["time"] / DATA["time"].max()
    Y_DATA = DATA["median"] / DATA["median"].max()

def fractional_zener_model(t, params):
    E0, E1, tau, alpha = params
    z = -(t / tau)**alpha
    ml_values = mittag_leffler(alpha, z)
    return E0 + E1 * ml_values

class P1(bf.BasicProblem):
    def __init__(self, variable_num, min_search_range, max_search_range):
        super().__init__(variable_num)
        self.min_search_range = np.array(min_search_range)
        self.max_search_range = np.array(max_search_range)
        self.global_optimum_solution = 0.0
        self.func_name = 'P1'

    def get_func_val(self, variables, *args):
        y_pred = fractional_zener_model(T_DATA, variables)
        residuals = Y_DATA - y_pred
        mse = np.mean(residuals**2)
        mae = np.mean(np.abs(residuals))
        return mse + mae




def evaluate_sequence_performance_all(sequence, prob, num_agents, num_iterations, num_replicas):
    """
    Corre Metaheuristic(prob, sequence, ...) num_replicas veces en paralelo,
    y retorna:
       - fitness_vals (array con f_best de cada réplica)
       - perf_median_iqr (mediana + iqr de fitness_vals)
       - avg_time (tiempo promedio de ejecución)
    """
    def run_metaheuristic():
        start_t = time.time()
        met = mh.Metaheuristic(prob, sequence, num_agents=num_agents, num_iterations=num_iterations)
        met.run()
        _, f_best = met.get_solution()
        elapsed = time.time() - start_t
        return f_best, elapsed

    num_cores = multiprocessing.cpu_count()
    results_parallel = Parallel(n_jobs=num_cores)(
        delayed(run_metaheuristic)() for _ in range(num_replicas)
    )

    fitness_vals = np.array([r[0] for r in results_parallel], dtype=float)
    times_vals   = np.array([r[1] for r in results_parallel], dtype=float)

    fitness_median = np.median(fitness_vals)
    iqr = np.percentile(fitness_vals, 75) - np.percentile(fitness_vals, 25)
    perf_median_iqr = fitness_median + iqr
    avg_time = np.mean(times_vals)

    return fitness_vals, perf_median_iqr, avg_time


# ========================================================================
# 3) Función hashable para heurísticas
# ========================================================================

def get_heuristic_key(h):
    name, params, modo = h
    sorted_items = sorted(params.items())
    param_str = "|".join(f"{k}:{v}" for k, v in sorted_items)
    return f"{name}||{param_str}||{modo}"


# ========================================================================
# 4) Búsqueda local
# ========================================================================

def local_search(seq, heuristics_list, prob, num_agents, num_iterations, num_replicas):
    """
    Retorna:
      best_seq, best_fv, best_perf, best_time,
      worst_seq, worst_fv, worst_perf, worst_time
    """
    # 1) Evaluar la secuencia original
    fit_orig, perf_orig, time_orig = evaluate_sequence_performance_all(
        seq, prob, num_agents, num_iterations, num_replicas
    )
    all_solutions = [(seq[:], fit_orig, perf_orig, time_orig)]
    neighbors = []

    # 2) SWAP (si la secuencia tiene al menos 2 heurísticas)
    if len(seq) > 1:
        seq_swap = seq[:]
        i, j = random.sample(range(len(seq)), 2)
        seq_swap[i], seq_swap[j] = seq_swap[j], seq_swap[i]
        neighbors.append(seq_swap)

    # 3) REPLACE (sustituir un operador por otro aleatorio que no esté en la secuencia)
    seq_replace = seq[:]
    pos_replace = random.randint(0, len(seq) - 1)
    current_keys = set(get_heuristic_key(x) for x in seq_replace)
    candidates = [h for h in heuristics_list if get_heuristic_key(h) not in current_keys]
    if candidates:
        seq_replace[pos_replace] = random.choice(candidates)
        neighbors.append(seq_replace)

    # 4) Evaluar vecinos
    for nb in neighbors:
        fit_nb, perf_nb, time_nb = evaluate_sequence_performance_all(
            nb, prob, num_agents, num_iterations, num_replicas
        )
        all_solutions.append((nb, fit_nb, perf_nb, time_nb))

    # 5) Best y worst global entre la original y los vecinos
    best_tuple = min(all_solutions, key=lambda x: x[2])   # performance menor => mejor
    worst_tuple = max(all_solutions, key=lambda x: x[2]) # performance mayor => peor

    (best_seq, best_fv, best_perf, best_time) = best_tuple
    (worst_seq, worst_fv, worst_perf, worst_time) = worst_tuple

    return best_seq, best_fv, best_perf, best_time, worst_seq, worst_fv, worst_perf, worst_time




def build_solution_greedy_random(heuristics_list, max_len=3):
    return random.sample(heuristics_list, max_len)


def heuristic_selection(
    heuristics_list,
    prob,
    num_agents=48
    num_iterations=70
    num_replicas=30
    heuristic_selection=10,
    max_len=3
):
    """
    Retorna:
      best_sequence_global, best_perf_global,
      best_fitness_history, best_perf_history, best_time_history,
      improvement_history,
      worst_sequence_global, worst_perf_global
    """
    # Variables para llevar la mejor secuencia global:
    best_sequence_global = None
    best_perf_global = float('inf')

    # Variables para llevar la peor secuencia global:
    worst_sequence_global = None
    worst_perf_global = float('-inf')

    # Historiales
    best_fitness_history = []
    best_perf_history = []
    best_time_history = []
    improvement_history = []

    # Bucle 
    for it in range(heuristic_selection):
        # 1) Construcción (solución aleatoria)
        candidate_seq = build_solution_greedy_random(heuristics_list, max_len)
        fv_candidate, perf_candidate, time_candidate = evaluate_sequence_performance_all(
            candidate_seq, prob, num_agents, num_iterations, num_replicas
        )

        # 2) Búsqueda local
        (best_seq_ls, best_fv_ls, best_perf_ls, best_time_ls,
         worst_seq_ls, worst_fv_ls, worst_perf_ls, worst_time_ls) = local_search(
            candidate_seq, heuristics_list, prob,
            num_agents, num_iterations, num_replicas
        )

        # 3) Decidir si tomamos la secuencia del LS o la construida
        if best_perf_ls < perf_candidate:
            final_seq_it  = best_seq_ls
            final_fv_it   = best_fv_ls
            final_perf_it = best_perf_ls
            final_time_it = best_time_ls
        else:
            final_seq_it  = candidate_seq
            final_fv_it   = fv_candidate
            final_perf_it = perf_candidate
            final_time_it = time_candidate

        # 4) Actualizar el mejor global
        if final_perf_it < best_perf_global:
            best_perf_global = final_perf_it
            best_sequence_global = final_seq_it[:]
            # Registramos la mejora en la lista de mejoras
            improvement_history.append((it+1, final_fv_it, final_perf_it))

        # 5) Actualizar el peor global (usamos la "peor secuencia" que salió de local_search)
        if worst_perf_ls > worst_perf_global:
            worst_perf_global = worst_perf_ls
            worst_sequence_global = worst_seq_ls[:]

        # 6) Guardar históricos (del "final" de esta iteración)
        best_fitness_history.append(final_fv_it)
        best_perf_history.append(final_perf_it)
        best_time_history.append(final_time_it)

        # 7) Impresión en consola (opcional)
        def get_indices(hseq):
            return [heuristics_list.index(h) for h in hseq]

        best_ls_idx  = get_indices(best_seq_ls)
        worst_ls_idx = get_indices(worst_seq_ls)

        print(
            f"[Gen {it+1}/{heuristic_selection}] "
            f"Best: MH_*= {best_ls_idx}, Perf_*= {best_perf_ls:.4f}, Time= {best_time_ls:.4f}, "
            f"Worst: MH_w= {worst_ls_idx}, Perf_w= {worst_perf_ls:.4f}, Time= {worst_time_ls:.4f}"
        )

    return (best_sequence_global,
            best_perf_global,
            best_fitness_history,
            best_perf_history,
            best_time_history,
            improvement_history,
            worst_sequence_global,
            worst_perf_global)


def plot_improvement_boxplot(improvement_history):
    if not improvement_history:
        print("No hubo mejoras en performance.")
        return

    data = [ih[1] for ih in improvement_history]  # ih[1] es array de fitness
    labels = [f"Iter {ih[0]}" for ih in improvement_history]

    plt.figure(figsize=(7,4))
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.tight_layout()
    plt.show()


def plot_fitness_and_performance(best_fitness_history, best_perf_history):
    n_iter = len(best_fitness_history)
    x_vals = np.arange(1, n_iter + 1)

    plt.figure(figsize=(8, 5))
    plt.boxplot(best_fitness_history, positions=x_vals, widths=0.6, showmeans=True)
    plt.plot(x_vals, best_perf_history, 'ro--', label='Performance')
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def plot_time_history(best_time_history):
    n_iter = len(best_time_history)
    x_vals = np.arange(1, n_iter + 1)

    plt.figure(figsize=(7,4))
    plt.plot(x_vals, best_time_history, 'g-o', label="Tiempo promedio")
    plt.xlabel("Generation")
    plt.ylabel("Average Time (s)")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_performance_vs_time(best_perf_history, best_time_history):
    n_iter = len(best_perf_history)
    x_vals = best_time_history
    y_vals = best_perf_history

    plt.figure(figsize=(6, 5))
    plt.scatter(x_vals, y_vals, color='purple')
    plt.xlabel("Average Time (s)")
    plt.ylabel("Performance")

    for i in range(n_iter):
        plt.annotate(f"Iter {i+1}", (x_vals[i], y_vals[i]),
                     textcoords="offset points", xytext=(5,5),
                     ha='left', color='darkgreen')

    plt.title("Performance vs. Tiempo")
    plt.grid(True)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    data_file = "/home/admin01/MicroHH/MG63_2f1_scaled_data.csv"
    init_data(data_file)

    # Definir problema
    min_range = [0.00001, 0.00001, 0.00001, 0.1]
    max_range = [1.5, 1.5, 1.5, 1.0]
    fun = P1(variable_num=4, min_search_range=min_range, max_search_range=max_range)
    prob = fun.get_formatted_problem()


    heuristics = [
        ('central_force_dynamic', {'gravity': 0.001, 'alpha': 0.01, 'beta': 1.5, 'dt': 1.0}, 'all'),
        ('central_force_dynamic', {'gravity': 0.001, 'alpha': 0.01, 'beta': 1.5, 'dt': 1.0}, 'greedy'),
        ('differential_mutation', {'expression': 'current-to-best', 'num_rands': 1, 'factor': 1.0}, 'all'),
        ('differential_mutation', {'expression': 'current-to-best', 'num_rands': 1, 'factor': 1.0}, 'greedy'),
        ('differential_mutation', {'expression': 'rand-to-best-and-current', 'num_rands': 1, 'factor': 1.0}, 'all'),
        ('differential_mutation', {'expression': 'rand-to-best-and-current', 'num_rands': 1, 'factor': 1.0}, 'greedy'),
        ('firefly_dynamic', {'distribution': 'uniform', 'alpha': 1.0, 'beta': 1.0, 'gamma': 100.0}, 'all'),
        ('firefly_dynamic', {'distribution': 'uniform', 'alpha': 1.0, 'beta': 1.0, 'gamma': 100.0}, 'greedy'),
        ('genetic_crossover', {'pairing': 'cost', 'crossover': 'uniform', 'mating_pool_factor': 0.4}, 'all'),
        ('genetic_crossover', {'pairing': 'cost', 'crossover': 'uniform', 'mating_pool_factor': 0.4}, 'greedy'),
        (
        'genetic_crossover', {'pairing': 'tournament_2_100', 'crossover': 'uniform', 'mating_pool_factor': 0.4}, 'all'),
        ('genetic_crossover', {'pairing': 'tournament_2_100', 'crossover': 'uniform', 'mating_pool_factor': 0.4},
         'greedy'),
        (
        'genetic_mutation', {'scale': 1.0, 'elite_rate': 0.1, 'mutation_rate': 0.25, 'distribution': 'uniform'}, 'all'),
        ('genetic_mutation', {'scale': 1.0, 'elite_rate': 0.1, 'mutation_rate': 0.25, 'distribution': 'uniform'},
         'greedy'),
        ('gravitational_search', {'gravity': 1.0, 'alpha': 0.02}, 'all'),
        ('gravitational_search', {'gravity': 1.0, 'alpha': 0.02}, 'greedy'),
        ('local_random_walk', {'probability': 0.75, 'scale': 1.0, 'distribution': 'gaussian'}, 'all'),
        ('local_random_walk', {'probability': 0.75, 'scale': 1.0, 'distribution': 'gaussian'}, 'greedy'),
        ('random_sample', {}, 'all'),
        ('random_sample', {}, 'greedy'),
        ('random_search', {'scale': 1.0, 'distribution': 'uniform'}, 'greedy'),
        ('random_search', {'scale': 0.01, 'distribution': 'uniform'}, 'greedy'),
        ('spiral_dynamic', {'radius': 0.9, 'angle': 22.5, 'sigma': 0.1}, 'all'),
        ('spiral_dynamic', {'radius': 0.9, 'angle': 22.5, 'sigma': 0.1}, 'greedy'),
        ('swarm_dynamic',
         {'factor': 0.7, 'self_conf': 2.54, 'swarm_conf': 2.56, 'version': 'inertial', 'distribution': 'uniform'},
         'all'),
        ('swarm_dynamic',
         {'factor': 0.7, 'self_conf': 2.54, 'swarm_conf': 2.56, 'version': 'inertial', 'distribution': 'uniform'},
         'greedy'),
        ('swarm_dynamic',
         {'factor': 1.0, 'self_conf': 2.54, 'swarm_conf': 2.56, 'version': 'constriction', 'distribution': 'uniform'},
         'all'),
        ('swarm_dynamic',
         {'factor': 1.0, 'self_conf': 2.54, 'swarm_conf': 2.56, 'version': 'constriction', 'distribution': 'uniform'},
         'greedy')
    ]


    all_results = []
    NUM_REPLICAS_EXTERNAS = 10
    for replica_idx in range(NUM_REPLICAS_EXTERNAS):
        print(f"\n=== Replica externa #{replica_idx+1} de {NUM_REPLICAS_EXTERNAS} ===\n")

        (best_seq, best_perf,
         best_fitness_hist, best_perf_hist, best_time_hist,
         improvement_hist,
         worst_seq, worst_perf) = heuristic_selection(
            heuristics_list=heuristics,
            prob=prob,
            num_agents=40,
            num_iterations=80,
            num_replicas=30,
            heuristic_selection=10,
            max_len=3
        )

        results_dict = {
            'replica': replica_idx + 1,
            'best_sequence': best_seq,
            'best_performance': best_perf,
            'worst_sequence': worst_seq,
            'worst_performance': worst_perf,
            'best_fitness_history': best_fitness_hist,
            'best_perf_history': best_perf_hist,
            'best_time_history': best_time_hist,
            'improvement_history': improvement_hist,
        }
        all_results.append(results_dict)

    df_results = pd.DataFrame(all_results)
    df_results.to_csv("results_10_replicas_with_worst_3op.csv", index=False)
    print(df_results)

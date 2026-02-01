''' MNA-IoT-DI
It runs, from a .json input file, generating solutions (stored in .txt files), 
as well as a new .json output file containing the parameters of the best solution 
found by this algorithm.

Designed by Murilo Táparo - November 2023

Last modified: 06/27/2025'''
# -*- coding: utf-8 -*-
# MNA-IoT mem optimization

# Imports for command line arguments reading
import os
from sys import argv, path

if __name__ == "__main__":
    path.append(os.path.join(os.path.dirname(path[0]), "commandLine"))
    path.append(os.path.join(os.path.dirname(path[0]), "inputReader"))
    import commandLine as cl
    import parquetReader as pr
    source_dir, target_dir = cl.get_target_dirs(argv)

import email_sender

import math
import json
import time
import numpy as np
from itertools import combinations
import heapq
from tabulate import tabulate
import pandas as pd
import pyarrow.parquet as pq
import pandas as pd
import pyarrow as pa
import numba
from numba import int32, int64, threading_layer
from numba.types import Tuple

#---------------------------- Global variables ---------------------------------
INF = 10**12 # Infinite
t_c = 1 # Time connection (t_c=1 ms)
#-------------------------------------------------------------------------------

def vector_space_generator(filtered_nodes, numNodes, cut_sol):
    """
    Generates binary masks with up to 'cut_sol' active nodes, only at the indexes in filtered_nodes.

    Args:
    filtered_nodes (list[int]): Indexes of the nodes considered.
    numNodes (int): Total nodes in the network.
    cut_sol (int): Maximum active nodes per combination.

    Yields:
    list[int]: Binary mask with 1s in the active nodes of the combination.
    """
    for k in range(1, cut_sol + 1):  # combinações de 1 até cut_sol nós ativos
        for combo in combinations(filtered_nodes, k):
            mask = [0] * numNodes
            for i in combo:
                mask[i] = 1
            yield mask

# Based on Dijkstra's Algorithm
def get_latencies(source, adjList, numNodes):
    latencies= [INF] * numNodes
    latencies[source] = 0
    heap = [(0, source)]

    while heap:
        curr_lat, u = heapq.heappop(heap)
        if curr_lat > latencies[u]:
            continue
        for v, weight in adjList[u]:
            if latencies[v] > curr_lat + weight:
                latencies[v] = curr_lat + weight
                heapq.heappush(heap, (latencies[v], v))
    return latencies


# Prints to the Terminal the execution header of each input file and the current job
def print_header(file, numRunnings, r, job, jr, jb, jl, jo):
    #---------------- Clear terminal window before running each Job ---------------
    # os.system('cls' if os.name == 'nt' else 'clear')
    print("\n---------------------------------------------------------")
    print(f"\n Processing the file: {file}")
    print(f"\n Running: {r+1}/{numRunnings}")
    print(f"\n Job {job} [{jr[job]}, {jb[job]}, {jl[job]}, {jo[job]}]")
    print("\n---------------------------------------------------------\n")
    return


def preselect_nodes(available, N_R, N_B, N_L, jr_job, jb_job, l_job, cut_comb_nodes):
    
    candidate_nodes = set()
    num_nodes = len (available)
    combs_found = 0
    
    for i in range(num_nodes):
        
        if combs_found == cut_comb_nodes:
            break

        if not available[i]:
            continue

        single_R = N_R[i]
        single_B = N_B[i]
        single_L = N_L[i]

        if single_R >= jr_job and single_B >= jb_job and single_L <= l_job:
            candidate_nodes.add(i)
            combs_found += 1
                        
        for j in range(1, num_nodes):

            if combs_found == cut_comb_nodes:
                break

            if not available[j]:
                continue

            total_R = single_R + N_R[j]
            total_B = single_B + N_B[j]
            max_L = max(single_L, N_L[j])
            
            if total_R >= jr_job and total_B >= jb_job and max_L <= l_job:

                candidate_nodes.add(i)
                candidate_nodes.add(j)
                combs_found += 1
    
    return np.fromiter(candidate_nodes, dtype=int, count=len(candidate_nodes))

VALOR_INVALIDO = -1

assinatura = Tuple((int32[:], int64))( # Tipos de retorno
    int64[:],    # valid_nodes
    numba.types.Array(int64, 1, 'C', readonly=True),     # N_R
    numba.types.Array(int64, 1, 'C', readonly=True),     # N_B
    int64[:],     # N_L
    int64,        # jr_job
    int64,        # jb_job
    int64         # l_job
)


def core_find_best_comb(
    valid_nodes, N_R, N_B, N_L, jr_job, jb_job, l_job
):
    """
    Busca paralela no espaço de busca sem condições de corrida, 
    calculando o melhor local para cada thread.
    """
    n_nodes = valid_nodes.shape[0]
    comb_size = 2
    
    # Arrays para guardar o melhor OF e a melhor combinação de cada thread.
    num_threads = numba.get_num_threads()
    melhores_OFs_locais = np.full(num_threads, INF, dtype=np.int64)
    melhores_combs_locais = np.full((num_threads, comb_size), VALOR_INVALIDO, dtype=np.int32)
    
    # node_i = 156
    # sin_R = N_R[node_i]
    # sin_B = N_B[node_i]
    # sin_L = N_L[node_i]
    #
    # f0 = sin_R - jr_job
    # f1 = sin_B - jb_job
    # f2 = l_job - sin_L
    # OF = (f0 * f0) + (f1 * f1) - f2
    #
    # print(node_i, OF)
    #
    #
    # node_i = 158
    # node_j = 812
    # sin_R = N_R[node_i] + N_R[node_j]
    # sin_B = N_B[node_i] + N_B[node_j]
    # sin_L = N_L[node_i] + N_L[node_j]
    #
    # f0 = sin_R - jr_job
    # f1 = sin_B - jb_job
    # f2 = l_job - sin_L
    # OF = (f0 * f0) + (f1 * f1) - f2
    
    # print([node_i, node_j], OF)


    # Distribuição das iterações entre os threads
    numba.set_parallel_chunksize(8)
    for i in numba.prange(n_nodes):
        # Descobre qual thread está executando esta iteração
        thread_id = numba.get_thread_id()

        node_i = valid_nodes[i]

        sin_R = N_R[node_i]
        sin_B = N_B[node_i]
        sin_L = N_L[node_i]

        if (sin_R >= jr_job and sin_B >= jb_job and sin_L <= l_job):
            f0 = sin_R - jr_job
            f1 = sin_B - jb_job
            f2 = l_job - sin_L
            OF = (f0 * f0) + (f1 * f1) - f2
            
            # Atualiza o melhor local para o caso de nó individual
            
            if OF < melhores_OFs_locais[thread_id]:
                melhores_OFs_locais[thread_id] = OF
                melhores_combs_locais[thread_id][0] = node_i
                melhores_combs_locais[thread_id][1] = VALOR_INVALIDO

        for j in range(i+1, n_nodes):
            
            node_j = valid_nodes[j]
            sum_R = sin_R + N_R[node_j]
            sum_B = sin_B + N_B[node_j]
            sum_L = sin_L + N_L[node_j]

            if (sum_R >= jr_job and sum_B >= jb_job and sum_L <= l_job):
                f0 = sum_R - jr_job
                f1 = sum_B - jb_job
                f2 = l_job - sum_L
                OF = (f0 * f0) + (f1 * f1) - f2
            
            # Atualiza o melhor local
            
                if OF < melhores_OFs_locais[thread_id]:
                    melhores_OFs_locais[thread_id] = OF
                    melhores_combs_locais[thread_id][0] = node_i
                    melhores_combs_locais[thread_id][1] = node_j
                    # print([node_i, node_j], OF)


    # Redução Final para encontrar o melhor global 
    
    # Índice do thread que teve o menor OF
    indice_melhor_thread = np.argmin(melhores_OFs_locais)
    
    # Melhor OF e a melhor combinação global a partir desse índice
    melhor_OF_global = melhores_OFs_locais[indice_melhor_thread]
    melhor_comb_global = melhores_combs_locais[indice_melhor_thread]
            
    
    return melhor_comb_global, melhor_OF_global

# function declaration for thread using
find_best_comb_par = numba.njit(assinatura, parallel=True)(core_find_best_comb)
find_best_comb = numba.njit(assinatura, parallel=False)(core_find_best_comb)


#-----------------------------------------------------------------------------------------------
def run_mna_jobs(file, numRunnings, r, jr, jb, jl, jo, N_R, N_B, adjList, numNodes, nThreads):
    n_jobs = len(jr)
    l_job = 0
    v_all_OF = []
    v_all_nodes = []
    v_all_sol_feasible = []
    v_times = {"latencies":0, "filter":0, "numba":0, "total":0}
    v_num_combs = []
    available= [True] * numNodes  # Allocated nodes management. In the beginning, all nodes are available
    # print("Iniciar ciclo de jobs")
    for job in range(n_jobs):
        # Prints to the Terminal the execution header of each input file and the current job
        # print_header(file, numRunnings, r, job, jr, jb, jl, jo) 
        source = jo[job]  # Stores the position value of the source node
        # Discounts the initial node connection time in the calculation (only 1 time for each job)
        l_job = jl[job] - t_c
        start_real = time.perf_counter()

        N_L = np.array(get_latencies(source, adjList, numNodes))
        
        v_times["latencies"] += time.perf_counter() - start_real

        Min_FX = INF
        v_sol_feasible = 0 # Nº feasible solutions
        #---------------------------------------------------------------------------------------------------------------------
        # Hyperparameters for cutting combinations considered and maximum number of nodes in the generated solutions
        # cut_comb_nodes = 500 # Considers the cut_comb_nodes combinations, according to DI, for each job.
        # cut_sol = 2 # Considers a maximum of 2 nodes when allocating a job
        #---------------------------------------------------------------------------------------------------------------------
        
        start = time.perf_counter()

        filtered_nodes = preselect_nodes(available, N_R, N_B, N_L, jr[job], jb[job], l_job, cut_comb_nodes)
        
        v_times["filter"] += time.perf_counter() - start
        # Flatten combinations to get only node indices (Ordered according to DI criteria)

        # Reduces the search space to only the filtered nodes (vector_space_generator=filtered_nodes)

        n_nodes = len(filtered_nodes)
        total_combinations = 0
        for k in range(1, 2 + 1):
            total_combinations += math.comb(n_nodes, k)

        # start = time.perf_counter_ns()

        v_num_combs.append(total_combinations)

        start = time.perf_counter()

        #Limite inferior para aplicação de paralelismo:
        if total_combinations < min_comb_threads:

            melhor_combinacao, Min_FX = find_best_comb(
            filtered_nodes, N_R, N_B, N_L, jr[job], jb[job], l_job)
        else:
            melhor_combinacao, Min_FX = find_best_comb_par(
            filtered_nodes, N_R, N_B, N_L, jr[job], jb[job], l_job)
        
        # Cálculo do tempo da exploração do espaço de busca
        v_times["numba"] += time.perf_counter() - start
        
        v_all_sol_feasible.append(v_sol_feasible)

        if Min_FX < INF:
            allocated = [i for i in melhor_combinacao.tolist() if i >= 0]
            
            for i in allocated:
                available[i] = False
            v_all_OF.append(Min_FX)
            v_all_nodes.append(allocated)
        else:
            v_all_OF.append(0)
            v_all_nodes.append([])
    
        v_times["total"] += time.perf_counter() - start_real

    return v_all_OF, v_all_nodes, v_all_sol_feasible, v_num_combs, v_times

def run_mna_iot_batch(source_dir, target_dir, numRunnings):
    """
    Reads files generated by the save_nodes_graph_to_parquet function 
    and executes the MNA-IoT algorithm in batch mode for all datasets found.
    
    Parameters:
    source_dir (str): directory where the generated files are located.
    target_dir (str): directory where the results will be saved.
    numRunnings (int): number of executions.
    """

    os.makedirs(target_dir, exist_ok=True)

    for files in pr.get_file_paths(source_dir):
        for nt in [1, 2, 4, 6, 8, 12, 16]:
            
            numba.set_num_threads(nt)
            # print(files)

            start_r = time.perf_counter()
            jr, jb, jl, jo, V_R, V_B, V_Busy, V_Inactive, numNodes, edge_nodes, adjList = pr.read_input(files, source_dir)

            # Gives the jobs_file and the suffix of it and receives it's base name in return
            full_base = pr.full_base_name(files[3], files[-1])

            c0, c1, c2 = 60, 1, 39
            ordered_lists = sorted(zip(jr, jb, jl, jo),
                                    key=lambda x: ((c0*(0.253*x[0])+c1*(0.024*x[1])-c2*(0.723*x[2]))/(c0+c1+c2)),
                                    reverse=True)
            jr, jb, jl, jo = map(list, zip(*ordered_lists))
            
            # Executions
            times_execs = []
            numba_times = []
            # print("Leitura? ")
            
            for r in range(numRunnings):
                start = time.perf_counter()
                
                v_all_OF, v_all_nodes, v_all_sol_feasible, v_num_combs, v_times = run_mna_jobs(
                    full_base, numRunnings, r, jr, jb, jl, jo, V_R, V_B, adjList, numNodes, nt)
                runtime = time.perf_counter() - start
                # print(f"Tempo de mna_jobs: {runtime:.6f}")
                times_execs.append(runtime)
                numba_times.append(v_times["numba"])

                file_path = save_results(r, files[3], full_base, edge_nodes, adjList,
                            jr, jb, jl, jo, V_R, V_B,
                            V_Busy, V_Inactive, v_all_OF, v_all_nodes, v_all_sol_feasible, times_execs, nt,
                            v_num_combs, v_times, numba_times)
                
                try:
                    email_sender.send_result(file_path, cut_sol, cut_comb_nodes, min_comb_threads)
                except:
                    pass
            
                
# Salva resultados em .txt e estatísticas na mesma abertura do arquivo
def save_results(r, jobs_file, full_base, edge_nodes, adjList,
                jr, jb, jl, jo, V_R, V_B, V_Busy, V_Inactive, 
                v_all_OF, v_all_nodes, v_all_sol_feasible,
                times_execs, nt,
                v_num_combs, v_times, v_numba_times):
    
    runtime = times_execs[-1]

    def format_float(x): return f"{x:18,.1f}"
    output_path = os.path.join(target_dir, f"c{nt}_results_{r}_MNA_IoT_{full_base}.txt")
    with open(output_path, 'w', encoding="utf-8") as out:
        out.write(f"Input file: {jobs_file}\nNumber of jobs: {len(jr)}\n\n")
        rows = (
            [i, f"[{jr[i]}, {jb[i]}, {jl[i]}, {jo[i]}]", format_float(of), str(sorted(nodes)), 
             v_num_combs[i]]
            for i, (of, nodes) in enumerate(zip(v_all_OF, v_all_nodes))
        )
        table = tabulate(rows, headers=["Job", "[Jr,Jb,Jl,Jo]", "OF", "Allocated nodes", "num_combs", "time"],
                          tablefmt="plain")
        out.write(table)
        out.write(f"\n\nTotal OF: {format_float(np.sum(v_all_OF)).strip()}\nRuntime: {runtime:.5f} sec")

        t_latencies = v_times["latencies"]
        t_filter = v_times["filter"]
        t_numba = v_times["numba"]
        t_total = v_times["total"]

        out.write(f"\n\nTotal time in jobs: {t_total:.5f}\n")
        out.write(f"get_latencies: {t_latencies:.5f} - {format_float(t_latencies / t_total * 100)}%\n")
        out.write(f" filter_nodes: {t_filter:.5f} - {format_float(t_filter / t_total * 100)}%\n")
        out.write(f"        numba: {t_numba:.5f} - {format_float(t_numba / t_total * 100)}%\n")
        # Última execução: salva arquivos parquet e estatísticas dentro do mesmo with
        if r == numRunnings - 1:
            prefix = os.path.join(target_dir, f"MNA_IoT_{full_base}")

            df_info_out = pd.DataFrame({"edge_nodes": [edge_nodes]})
            pq.write_table(pa.Table.from_pandas(df_info_out), f"{prefix}_info.parquet")

            df_jobs_out = pd.DataFrame({"jr": jr, "jb": jb, "jl": jl, "jo": jo})
            pq.write_table(pa.Table.from_pandas(df_jobs_out), f"{prefix}_jobs.parquet")

            df_nodes_out = pd.DataFrame({"R": V_R, "B": V_B, "Busy": V_Busy, "Inactive": V_Inactive})
            pq.write_table(pa.Table.from_pandas(df_nodes_out), f"{prefix}_nodes.parquet")

            edge_data = []
            for src, neighbors in enumerate(adjList):
                for tgt, lat in neighbors:
                    edge_data.append((src, tgt, lat))
            df_edges_out = pd.DataFrame(edge_data, columns=["source", "target", "latency"])
            pq.write_table(pa.Table.from_pandas(df_edges_out), f"{prefix}_edges.parquet")

            df_results_out = pd.DataFrame({
                "v_all_OF": v_all_OF,
                "v_all_nodes": [json.dumps(n) for n in v_all_nodes],
                "v_all_sol_feasible": v_all_sol_feasible,
                "Min_FX": [sum(v_all_OF)] * len(v_all_OF)
            })
            pq.write_table(pa.Table.from_pandas(df_results_out), f"{prefix}_results.parquet")

            # Estatísticas de todas as execuções
            out.write("\n--------------------- Times Execs -------------------------------")
            for i, t in enumerate(times_execs):
                out.write(f"\n {i:4d}:  {t:,.5f}")
            mean, sd = np.mean(times_execs), np.std(times_execs, ddof=(0 if numRunnings==1 else 1))
            out.write(f"\n mean: {mean:,.5f}\n   sd: {sd:,.5f}\n")
            
            out.write("\nFirst ignored:\n")
            mean, sd = np.mean(times_execs[1:]), 0 if numRunnings-1 == 1 else np.std(times_execs[1:], ddof=(0 if numRunnings==1 else 1))
            out.write(f"\n mean: {mean:,.5f}\n   sd: {sd:,.5f}\n")
            out.write("-----------------------------------------------------------------\n")
            out.write("\nNumba Times:\n")
            for i, t in enumerate(v_numba_times):
                out.write(f"\n {i:4d}:  {t:,.5f}")
            mean, sd = np.mean(v_numba_times), np.std(v_numba_times, ddof=(0 if numRunnings==1 else 1))
            out.write(f"\n mean: {mean:,.5f}\n   sd: {sd:,.5f}\n")
            out.write("\nFirst ignored:\n")
            mean, sd = np.mean(v_numba_times[1:]), 0 if numRunnings-1 == 1 else np.std(v_numba_times[1:], ddof=(0 if numRunnings==1 else 1))
            out.write(f"\n mean: {mean:,.5f}\n   sd: {sd:,.5f}\n")


    return output_path
#--------------------------------------------------------------------------------------------------------------------------------------------
# Start of execution (main):

#---------------- Clear terminal window before program execution ---------------
# os.system('cls' if os.name == 'nt' else 'clear')
# On Windows (nt) systems, use the cls command
# On Unix/Linux and MacOS systems, use the clear command
#-------------------------------------------------------------------------------
if __name__ == "__main__":
    # Setting default values (global)
    cut_comb_nodes = 5000
    cut_sol = 2
    min_comb_threads = 250_000

    # Set the directory path where the .json files are located

    # Number of runnings of a given configuration
    numRunnings = 5

    # Create the output folder if it doesn't exist; keep if it already exists
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    run_mna_iot_batch(source_dir, target_dir, numRunnings)

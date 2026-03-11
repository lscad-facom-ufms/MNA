''' DI-MNA-Multiprocessing - Multiprocessing (conflict-free + no-empty allocation)
Correções (coordenação, Top-K e custo soft)

Designed by Murilo Táparo - June 2023

Last modified: June 2025

'''
# -*- coding: utf-8 -*-

import os
import json
import time
import numpy as np
from itertools import combinations
import heapq
import gc
from tabulate import tabulate
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from multiprocessing import Pool, cpu_count

#---------------------------- Globais ---------------------------------------------------
INF = 10e12      # Large penalty for non unfeasible candidates
t_c = 1          # Connection time (ms) discounted from total latency
TOP_K = 128      # maximum number of proposals per job (fine-tuning performance/quality)
#----------------------------------------------------------------------------------------

def vector_space_generator(filtered_nodes, numNodes, cut_sol):
    """Generates binary masks with up to 'cut_sol' active nodes, restricted to filtered_nodes"""
    if not isinstance(filtered_nodes, (list, tuple)):
        filtered_nodes = sorted(filtered_nodes)
    for k in range(1, cut_sol + 1):
        for combo in combinations(filtered_nodes, k):
            mask = [0] * numNodes
            for i in combo:
                mask[i] = 1
            yield mask

def get_latencies(source, adjList, numNodes):
    """Dijkstra's Path"""
    latencies = [INF] * numNodes
    latencies[source] = 0
    heap = [(0, source)]
    while heap:
        curr_lat, u = heapq.heappop(heap)
        if curr_lat > latencies[u]:
            continue
        for v, weight in adjList[u]:
            nv = curr_lat + weight
            if latencies[v] > nv:
                latencies[v] = nv
                heapq.heappush(heap, (nv, v))
    return latencies

def soft_cost(sum_R, sum_B, sum_L, jr_job, jb_job, l_job):
    """
    "Soft" cost:
        - If feasible: base = (surplus_R)^2 + (surplus_B)^2 - slack_L (same spirit as the original)
        - If unfeasible: penalizes quadratic deficits + INF (to always prefer feasible)
        Returns (cost, is_feasible)
    """
    deficit_R = max(0.0, jr_job - sum_R)
    deficit_B = max(0.0, jb_job - sum_B)
    over_L    = max(0.0, sum_L - l_job)

    if deficit_R == 0 and deficit_B == 0 and over_L == 0:
        f0 = sum_R - jr_job
        f1 = sum_B - jb_job
        f2 = l_job - sum_L
        return (f0**2 + f1**2 - f2, True)
    else:
        penalty = deficit_R**2 + deficit_B**2 + over_L**2
        return (INF + penalty, False)

def preselect_nodes(N_R, N_B, N_L, jr_job, jb_job, l_job, numNodes, cut_comb_nodes):
    """
    Pré-seleciona uma cabeça de nós promissores (sem checar disponibilidade global aqui).
    Heurística: baixa latência e maior (R+B).
    Retorna lista de índices de nós (head).
    """
    scored = sorted(
        range(numNodes),
        key=lambda i: (N_L[i], -(N_R[i] + N_B[i]))
    )
    return scored[:max(1, cut_comb_nodes)]

def run_job(args):
    """
    Calcula, em paralelo, as TOP_K melhores propostas (com 1..cut_sol nós) para um job.
    NÃO mexe na disponibilidade global.
    Retorna: (job_idx, [ (cost, tuple(nodes), feasible_bool) ... ] ordenada por custo)
    """
    (job_idx, jr_job, jb_job, jl_job, jo_job,
     N_R, N_B, adjList, numNodes, cut_comb_nodes, cut_sol) = args

    source = int(jo_job)
    l_job = jl_job - t_c
    N_L = get_latencies(source, adjList, numNodes)

    head_nodes = preselect_nodes(N_R, N_B, N_L, jr_job, jb_job, l_job, numNodes, cut_comb_nodes)

    # Monta um min-heap de candidatas (cost, nodes_tuple, feasible)
    # Usamos heap como max-heap limitado a TOP_K (armazenando -cost para cortar piores).
    max_heap = []  # guardará (-cost, nodes_tuple, feasible)
    seen = set()

    # Gera 1..cut_sol nós em cima da "cabeça" head_nodes
    for k in range(1, cut_sol + 1):
        for combo in combinations(head_nodes, k):
            if combo in seen:
                continue
            seen.add(combo)
            sum_R = sum(N_R[i] for i in combo)
            sum_B = sum(N_B[i] for i in combo)
            # métrica de latência: soma das distâncias da origem do job até cada nó
            sum_L = sum(N_L[i] for i in combo)

            cost, feasible = soft_cost(sum_R, sum_B, sum_L, jr_job, jb_job, l_job)
            item = (-cost, combo, feasible)  # para limitar TOP_K no heap

            if len(max_heap) < TOP_K:
                heapq.heappush(max_heap, item)
            else:
                # se melhor que o pior atual, substitui
                if item > max_heap[0]:
                    heapq.heapreplace(max_heap, item)

    # Converte heap em lista ordenada por custo crescente
    proposals = [(-neg_cost, nodes, feasible) for (neg_cost, nodes, feasible) in max_heap]
    proposals.sort(key=lambda x: (x[0], 0 if x[2] else 1))  # custo crescente; factível antes de inviável em empate

    return job_idx, proposals

def get_positive_integer(prompt, default):
    while True:
        try:
            user_input = input(f"{prompt} ({default}): ")
            value = int(user_input) if user_input else default
            if value > 0:
                return value
            print("⚠️  O número deve ser positivo. Tente novamente.")
        except ValueError:
            print("⚠️  Entrada inválida. Digite um inteiro.")

def get_config_type(prompt="Type L (Lightweight) or H (Heavyweight) to run the configuration type: "):
    while True:
        letter = input(prompt).strip().upper()
        if letter == 'L':
            while True:
                print(f"\nSelected configuration: {letter}")
                confirm = input("\nDo you want to run the lightweight configuration? (Y/n): ").strip()
                if confirm == '':
                    confirm = 'Y'
                if confirm == 'Y':
                    return "lightweight"
                elif confirm == 'n':
                    break
                else:
                    print("⚠️ Invalid option! Please enter 'Y' or 'n'.")
        elif letter == 'H':
            while True:
                print(f"\nSelected configuration: {letter}")
                confirm = input("\nDo you want to run the heavyweight configuration? (Y/n): ").strip()
                if confirm == '':
                    confirm = 'Y'
                if confirm == 'Y':
                    return "heavyweight"
                elif confirm == 'n':
                    break
                else:
                    print("⚠️ Invalid option! Please enter 'Y' or 'n'.")
        else:
            print("⚠️ Invalid key! Try again!")

def run_mna_iot_batch(source_dir, target_dir, numRunnings,
                      cut_comb_nodes, cut_sol):
    """
    Lê os arquivos (parquet) e executa o DI-MNA-Multiprocessing em batch,
    calculando propostas por job em paralelo (Top-K),
    e coordenando no processo pai para garantir:
      - exclusividade de nós
      - nenhuma solução vazia (sempre aloca algo)
    """

    os.makedirs(target_dir, exist_ok=True)
    all_files = os.listdir(source_dir)

    # Tipos de configuração (light / heavy)
    type_suffix = {"lightweight": "_light", "heavyweight": "_heavy"}

    for cfg, suffix in type_suffix.items():
        # encontra todos os jobs para essa config
        job_files = [f for f in all_files if f.endswith(f"{suffix}.parquet") and "jobs" in f]

        if not job_files:
            print(f"⚠️ Nenhum jobs file encontrado para {cfg}")
            continue

        for jobs_file in job_files:
            # base do nome (ex: input_5000nds_3500jobs)
            full_base = jobs_file.replace(f"{suffix}.parquet", "")
            print(f"\n📁 Dataset detectado: {full_base} ({cfg})")

            # info correspondente
            info_file = f"{full_base}{suffix}_info.parquet"
            if info_file not in all_files:
                print(f"⚠️ Missing info file: {info_file}")
                continue

            # localiza nós e manifest
            dataset_prefix = "_".join(full_base.split("_")[:2])
            nodes_file_candidates = [f for f in all_files if f.startswith(dataset_prefix) and f.endswith("_nodes.parquet")]
            manifest_file_candidates = [f for f in all_files if f.startswith(dataset_prefix) and f.endswith("_edges_manifest.parquet")]

            if not nodes_file_candidates or not manifest_file_candidates:
                print(f"⚠️ Missing nodes/manifest files for {dataset_prefix}")
                continue

            nodes_file = nodes_file_candidates[0]
            manifest_file = manifest_file_candidates[0]

            # paths absolutos
            path_jobs = os.path.join(source_dir, jobs_file)
            path_info = os.path.join(source_dir, info_file)
            path_nodes = os.path.join(source_dir, nodes_file)
            path_manifest = os.path.join(source_dir, manifest_file)

            # leitura dos dataframes
            df_jobs = pd.read_parquet(path_jobs)
            df_info = pd.read_parquet(path_info)
            df_nodes = pd.read_parquet(path_nodes)
            df_manifest = pd.read_parquet(path_manifest)

            # leitura das arestas
            df_edges_list = []
            if "file" in df_manifest.columns:
                for f in df_manifest["file"]:
                    full_path = os.path.join(source_dir, os.path.basename(f))
                    if os.path.exists(full_path):
                        df_edges_list.append(pd.read_parquet(full_path))
            else:
                df_edges_list.append(df_manifest)

            if not df_edges_list:
                print(f"❌ No edge partitions loaded for {full_base}{suffix}")
                continue

            df_edges = pd.concat(df_edges_list, ignore_index=True)

            # vetores jobs
            jr = df_jobs["jr"].to_numpy()
            jb = df_jobs["jb"].to_numpy()
            jl = df_jobs["jl"].to_numpy()
            jo = df_jobs["jo"].to_numpy()

            # vetores nós
            V_R = df_nodes["R"].to_numpy()
            V_B = df_nodes["B"].to_numpy()
            V_Busy = df_nodes["Busy"].to_numpy()
            V_Inactive = df_nodes["Inactive"].to_numpy()
            numNodes = len(V_R)
            edge_nodes = df_info["edge_nodes"].iloc[0]

            # lista de adjacência
            adjList = [[] for _ in range(numNodes)]
            for _, row in df_edges.iterrows():
                src, tgt, lat = int(row["source"]), int(row["target"]), float(row["latency"])
                adjList[src].append((tgt, lat))

            # ordenação jobs
            c0, c1, c2 = 60, 1, 39
            ordered = sorted(
                zip(range(len(jr)), jr, jb, jl, jo),
                key=lambda x: ((c0*(0.253*x[1]) + c1*(0.024*x[2]) - c2*(0.723*x[3])) / (c0+c1+c2)),
                reverse=True
            )
            job_ids, jr, jb, jl, jo = map(list, zip(*ordered))

            if len(jr) > numNodes:
                print(f"⚠️ Jobs ({len(jr)}) > nós ({numNodes}) → exclusividade impossível. Skip dataset.")
                continue

            # execuções
            times_execs_cpu = []
            times_execs_real = []

            for r in range(numRunnings):
                start_cpu = time.process_time()
                start_real = time.perf_counter()

                # === Passo 1: propostas Top-K em paralelo ===
                pool_args = [
                    (i, jr[i], jb[i], jl[i], jo[i],
                     V_R, V_B, adjList, numNodes, cut_comb_nodes, cut_sol)
                    for i in range(len(jr))
                ]
                with Pool(processes=cpu_count()) as pool:
                    job_proposals = pool.map(run_job, pool_args)
                job_proposals.sort(key=lambda x: x[0])

                # === Passo 2: coordenação ===
                available = [True] * numNodes
                v_all_cost, v_all_nodes, v_all_sol_feasible = [], [], []

                for job_idx, proposals in job_proposals:
                    chosen = None
                    feasible_count = sum(1 for _, _, f in proposals if f)
                    v_all_sol_feasible.append(feasible_count)

                    for cost, nodes, feasible in proposals:
                        if all(available[i] for i in nodes):
                            chosen = (cost, nodes, feasible)
                            break

                    if chosen is None:
                        source = int(jo[job_idx])
                        N_L = get_latencies(source, adjList, numNodes)
                        l_job = jl[job_idx] - t_c
                        best_single = None
                        for i in range(numNodes):
                            if not available[i]:
                                continue
                            sum_R = V_R[i]
                            sum_B = V_B[i]
                            sum_L = N_L[i]
                            cost, feasible = soft_cost(sum_R, sum_B, sum_L, jr[job_idx], jb[job_idx], l_job)
                            cand = (cost, (i,), feasible)
                            if (best_single is None) or (cand[0] < best_single[0]):
                                best_single = cand
                        chosen = best_single

                    cost, nodes, feasible = chosen
                    v_all_cost.append(cost)
                    v_all_nodes.append(sorted(nodes))
                    for i in nodes:
                        available[i] = False

                runtime_cpu = time.process_time() - start_cpu
                runtime_real = time.perf_counter() - start_real
                times_execs_cpu.append(runtime_cpu)
                times_execs_real.append(runtime_real)

                # === Saída TXT ===
                def format_float(x): return f"{x:18,.1f}"
                output_path = os.path.join(
                    target_dir, f"results_{r}_MNA_IoT_{full_base}{suffix}_multiprocessing.txt"
                )
                with open(output_path, 'w', encoding="utf-8") as out:
                    out.write(f"Input file: {jobs_file}\nNumber of jobs: {len(jr)}\n\n")
                    rows = (
                        [i, f"[{jr[i]}, {jb[i]}, {jl[i]}, {jo[i]}]",
                         format_float(v_all_cost[i]), str(v_all_nodes[i])]
                        for i in range(len(jr))
                    )
                    table = tabulate(rows, headers=["Job", "[Jr,Jb,Jl,Jo]", "OF", "Allocated nodes"],
                                     tablefmt="plain", colalign=("center", "left", "right", "right"))
                    out.write(table)
                    out.write(f"\n\nTotal OF: {format_float(np.sum(v_all_cost)).strip()}\n")
                    out.write(f"Runtime (CPU): {runtime_cpu:,.5f} sec\n")
                    out.write(f"Runtime (REAL): {runtime_real:,.5f} sec\n")

                    if r == numRunnings - 1:
                        # estatísticas + parquet
                        out.write("\n---------------- Times Execs (CPU) ----------------")
                        for i, t in enumerate(times_execs_cpu):
                            out.write(f"\n {i:4d}:  {t:,.5f}")
                        mean, sd = np.mean(times_execs_cpu), np.std(times_execs_cpu, ddof=(0 if numRunnings == 1 else 1))
                        out.write(f"\n mean: {mean:,.5f}\n   sd: {sd:,.5f}\n")

                        out.write("\n---------------- Times Execs (REAL) ----------------")
                        for i, t in enumerate(times_execs_real):
                            out.write(f"\n {i:4d}:  {t:,.5f}")
                        mean, sd = np.mean(times_execs_real), np.std(times_execs_real, ddof=(0 if numRunnings == 1 else 1))
                        out.write(f"\n mean: {mean:,.5f}\n   sd: {sd:,.5f}\n")

                        # parquet saída
                        prefix = os.path.join(target_dir, f"DI_MNA_{full_base}{suffix}")
                        df_info_out = pd.DataFrame({"edge_nodes": [edge_nodes]})
                        pq.write_table(pa.Table.from_pandas(df_info_out), f"{prefix}_info.parquet")
                        df_jobs_out = pd.DataFrame({"jr": jr, "jb": jb, "jl": jl, "jo": jo})
                        pq.write_table(pa.Table.from_pandas(df_jobs_out), f"{prefix}_jobs.parquet")
                        df_nodes_out = pd.DataFrame({"R": V_R, "B": V_B, "Busy": V_Busy, "Inactive": V_Inactive})
                        pq.write_table(pa.Table.from_pandas(df_nodes_out), f"{prefix}_nodes.parquet")
                        edge_data = [(src, tgt, lat) for src, neighbors in enumerate(adjList) for tgt, lat in neighbors]
                        df_edges_out = pd.DataFrame(edge_data, columns=["source", "target", "latency"])
                        pq.write_table(pa.Table.from_pandas(df_edges_out), f"{prefix}_edges.parquet")
                        df_results_out = pd.DataFrame({
                            "v_all_OF": v_all_cost,
                            "v_all_nodes": [json.dumps(n) for n in v_all_nodes],
                            "v_all_sol_feasible": v_all_sol_feasible,
                            "Min_FX": [sum(v_all_cost)] * len(v_all_cost)
                        })
                        pq.write_table(pa.Table.from_pandas(df_results_out), f"{prefix}_results.parquet")

            gc.collect()

#--------------------------------------------------------------------------------------------------------------------------------------------
# Start of execution (main):
#--------------------------------------------------------------------
if __name__ == "__main__":
    #---------------- Clear terminal window before program execution ---------------
    os.system('cls' if os.name == 'nt' else 'clear')
    # On Windows (nt) systems, use the cls command
    # On Unix/Linux and MacOS systems, use the clear command
    #-------------------------------------------------------------------------------

    # Setting default values (global)
    cut_comb_nodes = 5_000
    cut_sol = 2

    # Set the directory path where the .json files are located
    from sys import argv, path
    path.append(os.path.join(os.path.dirname(path[0]), "commandLine"))
    import commandLine as cl
    source_dir, target_dir = cl.get_target_dirs(argv)
    
    # Number of runnings of a given configuration
    numRunnings = 1

    # Create the output folder if it doesn't exist; keep if it already exists
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    run_mna_iot_batch(source_dir, target_dir, numRunnings, cut_comb_nodes, cut_sol)

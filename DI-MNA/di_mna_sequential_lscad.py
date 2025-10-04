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
path.append(os.path.join(os.path.dirname(path[0]), "commandLine"))
import commandLine as cl
source_dir, target_dir = cl.get_target_dirs(argv)

import json
import time
import numpy as np
from itertools import combinations
import heapq
import gc
from tabulate import tabulate
import pandas as pd
import pyarrow.parquet as pq
import pandas as pd
import pyarrow as pa

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
    os.system('cls' if os.name == 'nt' else 'clear')
    print("\n---------------------------------------------------------")
    print(f"\n Processing the file: {file}")
    print(f"\n Running: {r+1}/{numRunnings}")
    print(f"\n Job {job} [{jr[job]}, {jb[job]}, {jl[job]}, {jo[job]}]")
    print("\n---------------------------------------------------------\n")
    return

def preselect_nodes(available, N_R, N_B, N_L, jr_job, jb_job, l_job, numNodes, cut_comb_nodes, cut_sol):
    """
    Preselects combinations of up to `cut_sol` nodes that together satisfy job requirements,
    and returns up to `cut_comb_nodes` combinations sorted by minimum max latency.

    Parameters:
    available (list[bool]): Availability status of nodes.
    N_R (list[float]): Resource availability per node.
    N_B (list[float]): Bandwidth availability per node.
    N_L (list[float]): Latency per node.
    jr_job (float): Job resource requirement.
    jb_job (float): Job bandwidth requirement.
    l_job (float): Job latency constraint.
    numNodes (int): Total number of nodes.
    cut_comb_nodes (int): Maximum number of combinations to return.
    cut_sol (int): Maximum number of nodes allowed in a combination.

    Returns:
    list[tuple[int]]: List of node combinations satisfying the job.
    """
    valid_combinations = []
    # Generates all combinations of available IoT network nodes
    candidates = [i for i in range(numNodes) if available[i]]
    
    for r in range(1, cut_sol + 1):  # Quantity combinations from 1 to cut_sol
        # Cut Radius (Job Latency), latency threshold
        for combo in combinations(candidates[:cut_comb_nodes], r):
            total_R = sum(N_R[i] for i in combo)
            total_B = sum(N_B[i] for i in combo)
            max_L = max(N_L[i] for i in combo)

            if total_R >= jr_job and total_B >= jb_job and max_L <= l_job:
                valid_combinations.append((combo, max_L))

    # Only take the first `cut_comb_nodes` combinations
    top_combinations = [combo for combo, _ in valid_combinations[:cut_comb_nodes]]

    return top_combinations
#-----------------------------------------------------------------------------------------------

def run_mna_jobs(file, numRunnings, r, jr, jb, jl, jo, N_R, N_B, adjList, numNodes):
    n_jobs = len(jr)
    l_job = 0
    v_all_OF = []
    v_all_nodes = []
    v_all_sol_feasible = []
    available= [True] * numNodes  # Allocated nodes management. In the beginning, all nodes are available

    for job in range(n_jobs):
        # Prints to the Terminal the execution header of each input file and the current job
        # print_header(file, numRunnings, r, job, jr, jb, jl, jo) 
        source = jo[job]  # Stores the position value of the source node
        # Discounts the initial node connection time in the calculation (only 1 time for each job)
        l_job = jl[job] - t_c
        N_L = get_latencies(source, adjList, numNodes)

        Min_FX = INF
        better_mask = [0] * numNodes
        v_sol_feasible = 0 # Nº feasible solutions
        #---------------------------------------------------------------------------------------------------------------------
        # Hyperparameters for cutting combinations considered and maximum number of nodes in the generated solutions
        # cut_comb_nodes = 500 # Considers the cut_comb_nodes combinations, according to DI, for each job.
        # cut_sol = 2 # Considers a maximum of 2 nodes when allocating a job
        #---------------------------------------------------------------------------------------------------------------------
        filtered_nodes = preselect_nodes(available, N_R, N_B, N_L, jr[job], jb[job], l_job, numNodes, cut_comb_nodes, cut_sol)
        # Flatten combinations to get only node indices (Ordered according to DI criteria)
        filtered_nodes = set(i for combo in filtered_nodes for i in combo)
        # Reduces the search space to only the filtered nodes (vector_space_generator=filtered_nodes)
        for mask in vector_space_generator(filtered_nodes, numNodes, cut_sol):
            comb_nodes = [i for i in filtered_nodes if mask[i]]
            # Skip the current iteration of the loop if the comb_nodes list is empty
            if not comb_nodes:
               continue
            # Discards combinations that:
            # (1) not all(available[i] for i in comb_nodes) - have some unavailable nodes, OR
            # (2) len(comb_nodes) > 2 - have more than 2 active nodes
            if not all(available[i] for i in comb_nodes) or len(comb_nodes) > 2:
               continue

            sum_R = sum(N_R[i] for i in comb_nodes)
            sum_B = sum(N_B[i] for i in comb_nodes)
            sum_L = sum(N_L[i] for i in comb_nodes)

            if (sum_R >= jr[job] and sum_B >= jb[job] and sum_L <= l_job):
                f0 = sum_R - jr[job]
                f1 = sum_B - jb[job]
                f2 = l_job - sum_L
                OF = f0**2 + f1**2 - f2
                v_sol_feasible += 1 
                if OF <= Min_FX:
                    Min_FX = OF
                    better_mask = mask
                    
            else:    
                # For solutions that are not feasible, move on to the next configuration
                continue

        #---------------------------------------- End for mask -------------------------------------------------
        #-------------- Stores the number of feasible solutions for each job -----------------------------------
        v_all_sol_feasible.append(v_sol_feasible)

        if Min_FX < INF:
            allocated = [i for i in range(numNodes) if better_mask[i]]
            for i in allocated:
                available[i] = False
            v_all_OF.append(Min_FX)
            v_all_nodes.append(allocated)
        else:
            v_all_OF.append(0)
            v_all_nodes.append([])


        gc.collect()
    return v_all_OF, v_all_nodes, v_all_sol_feasible


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
    all_files = os.listdir(source_dir)

    # Configuration types
    type_suffix = {"lightweight": "_light", "heavyweight": "_heavy"}

    for cfg, suffix in type_suffix.items():
        # finds all job files for this configuration
        job_files = [f for f in all_files if f.endswith(f"{suffix}.parquet") and "jobs" in f]

        if not job_files:
            print(f"⚠️ No jobs files found for {cfg}")
            continue

        for jobs_file in job_files:
            full_base = jobs_file.replace(f"{suffix}.parquet", "")
            #print(f"\n📁 Dataset detectado: {full_base} ({cfg})")

            # Corresponding info file
            info_file = f"{full_base}{suffix}_info.parquet"
            if info_file not in all_files:
                print(f"⚠️ Missing info file: {info_file}")
                continue

            # Find correct nodes and edges_manifest by dataset prefix
            dataset_prefix = "_".join(full_base.split("_")[:2])
            nodes_file_candidates = [f for f in all_files if f.startswith(dataset_prefix) and f.endswith("_nodes.parquet")]
            manifest_file_candidates = [f for f in all_files if f.startswith(dataset_prefix) and f.endswith("_edges_manifest.parquet")]

            if not nodes_file_candidates or not manifest_file_candidates:
                print(f"⚠️ Missing nodes/manifest files for {dataset_prefix}")
                continue

            nodes_file = nodes_file_candidates[0]
            manifest_file = manifest_file_candidates[0]

            # Absolute Paths
            path_jobs = os.path.join(source_dir, jobs_file)
            path_info = os.path.join(source_dir, info_file)
            path_nodes = os.path.join(source_dir, nodes_file)
            path_manifest = os.path.join(source_dir, manifest_file)

            # Reads main files
            df_jobs = pd.read_parquet(path_jobs)
            df_info = pd.read_parquet(path_info)
            df_nodes = pd.read_parquet(path_nodes)
            df_manifest = pd.read_parquet(path_manifest)

            # Reads edge partitions, if any.
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

            # Variable extraction
            jr = df_jobs["jr"].to_numpy()
            jb = df_jobs["jb"].to_numpy()
            jl = df_jobs["jl"].to_numpy()
            jo = df_jobs["jo"].to_numpy()

            V_R = df_nodes["R"].to_numpy()
            V_B = df_nodes["B"].to_numpy()
            V_Busy = df_nodes["Busy"].to_numpy()
            V_Inactive = df_nodes["Inactive"].to_numpy()
            numNodes = len(V_R)
            edge_nodes = df_info["edge_nodes"].iloc[0]

            # Create adjacency list
            adjList = [[] for _ in range(numNodes)]
            for _, row in df_edges.iterrows():
                src, tgt, lat = int(row["source"]), int(row["target"]), float(row["latency"])
                adjList[src].append((tgt, lat))

            # Job ordering
            c0, c1, c2 = 60, 1, 39
            ordered_lists = sorted(zip(jr, jb, jl, jo),
                                   key=lambda x: ((c0*(0.253*x[0])+c1*(0.024*x[1])-c2*(0.723*x[2]))/(c0+c1+c2)),
                                   reverse=True)
            jr, jb, jl, jo = map(list, zip(*ordered_lists))

            # Executions
            times_execs = []
            for r in range(numRunnings):
                start = time.process_time()
                v_all_OF, v_all_nodes, v_all_sol_feasible = run_mna_jobs(
                    full_base, numRunnings, r, jr, jb, jl, jo, V_R, V_B, adjList, numNodes)
                runtime = time.process_time() - start
                times_execs.append(runtime)

                # Salva resultados em .txt e estatísticas na mesma abertura do arquivo
                def format_float(x): return f"{x:18,.1f}"
                output_path = os.path.join(target_dir, f"results_{r}_MNA_IoT_{full_base}{suffix}.txt")
                with open(output_path, 'w', encoding="utf-8") as out:
                    out.write(f"Input file: {jobs_file}\nNumber of jobs: {len(jr)}\n\n")
                    rows = (
                        [i, f"[{jr[i]}, {jb[i]}, {jl[i]}, {jo[i]}]", format_float(of), str(sorted(nodes))]
                        for i, (of, nodes) in enumerate(zip(v_all_OF, v_all_nodes))
                    )
                    table = tabulate(rows, headers=["Job", "[Jr,Jb,Jl,Jo]", "OF", "Allocated nodes"], tablefmt="plain")
                    out.write(table)
                    out.write(f"\n\nTotal OF: {format_float(np.sum(v_all_OF)).strip()}\nRuntime: {runtime:.5f} sec\n")

                    # Última execução: salva arquivos parquet e estatísticas dentro do mesmo with
                    if r == numRunnings - 1:
                        prefix = os.path.join(target_dir, f"MNA_IoT_{full_base}{suffix}")

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
                        out.write("-----------------------------------------------------------------\n")
#--------------------------------------------------------------------------------------------------------------------------------------------
# Start of execution (main):

#---------------- Clear terminal window before program execution ---------------
os.system('cls' if os.name == 'nt' else 'clear')
# On Windows (nt) systems, use the cls command
# On Unix/Linux and MacOS systems, use the clear command
#-------------------------------------------------------------------------------

# Setting default values (global)
cut_comb_nodes = 500
cut_sol = 2

# Set the directory path where the .json files are located

# Number of runnings of a given configuration
numRunnings = 10

# Create the output folder if it doesn't exist; keep if it already exists
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

run_mna_iot_batch(source_dir, target_dir, numRunnings)
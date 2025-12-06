''' 
MNA-IoT-DI (parallel threads, non-deterministic allocation)
Parallel candidate evaluation + concurrent allocation using per-node locks 
and .parquet input files.

Designed by Murilo Táparo - August 2025

Modified for max-parallelism (non-deterministic): October 10, 2025

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
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import threading

#---------------------------- Global variables ---------------------------------
INF = 10**12  # Infinite
t_c = 1       # Time connection (t_c=1 ms)
#-------------------------------------------------------------------------------

def vector_space_generator(filtered_nodes_sorted, numNodes, cut_sol):
    """
    Generates binary masks with up to 'cut_sol' active nodes, only at the indexes in filtered_nodes_sorted.
    """
    for k in range(1, cut_sol + 1):
        for combo in combinations(filtered_nodes_sorted, k):
            mask = [0] * numNodes
            for i in combo:
                mask[i] = 1
            yield mask

def get_latencies(source, adjList, numNodes):
    latencies = [INF] * numNodes
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

def get_positive_integer(prompt, default):
    while True:
        try:
            user_input = input(f"{prompt} ({default}): ")
            value = int(user_input) if user_input else default
            if value > 0:
                return value
            else:
                print("⚠️ Error: The number must be a positive value. Please try again.")
        except ValueError:
            print("⚠️ Error: Invalid input. Please enter an integer.")

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

def preselect_nodes_dynamic(available, N_R, N_B, N_L, jr_job, jb_job, l_job, numNodes, cut_comb_nodes, cut_sol):
    """
    Preselect combinations of up to `cut_sol` nodes that together satisfy job requirements,
    but consider only nodes current marked available (dynamic).
    Returns a sorted list of unique node indices that appear in top combinations.
    """
    valid_combinations = []
    candidates = [i for i in range(numNodes) if available[i]]
    # limit candidates deterministically by order of indices
    for r in range(1, cut_sol + 1):
        for combo in combinations(candidates[:cut_comb_nodes], r):
            total_R = sum(N_R[i] for i in combo)
            total_B = sum(N_B[i] for i in combo)
            max_L = max(N_L[i] for i in combo)
            if total_R >= jr_job and total_B >= jb_job and max_L <= l_job:
                valid_combinations.append((combo, max_L))
    top_combinations = [combo for combo, _ in valid_combinations[:cut_comb_nodes]]
    filtered_nodes = sorted({i for combo in top_combinations for i in combo})
    return filtered_nodes

def evaluate_candidates(job_index, source, jr_job, jb_job, jl_job, V_R, V_B, adjList, numNodes, cut_comb_nodes, cut_sol):
    """
    Compute candidate solutions for a job (independent compute).
    Returns sorted list of candidates (OF, mask, allocated_tuple) sorted by OF ascending.
    """
    N_L = get_latencies(source, adjList, numNodes)
    l_job = jl_job - t_c
    # Preselect over all nodes (we do dynamic availability checks when allocating)
    # Using all nodes speeds candidate generation; filtering to first cut_comb_nodes to limit explosion
    candidates = []
    # Use deterministic candidate generation order (combinations of node indices)
    candidates_nodes = list(range(numNodes))
    for mask in vector_space_generator(candidates_nodes[:cut_comb_nodes], numNodes, cut_sol):
        comb_nodes = [i for i in candidates_nodes if mask[i]]
        if not comb_nodes:
            continue
        if len(comb_nodes) > cut_sol:
            continue
        sum_R = sum(V_R[i] for i in comb_nodes)
        sum_B = sum(V_B[i] for i in comb_nodes)
        sum_L = sum(N_L[i] for i in comb_nodes)
        if (sum_R >= jr_job and sum_B >= jb_job and sum_L <= l_job):
            f0 = sum_R - jr_job
            f1 = sum_B - jb_job
            f2 = l_job - sum_L
            OF = f0**2 + f1**2 - f2
            allocated = tuple(sorted(comb_nodes))
            candidates.append((OF, mask, allocated))
    # sort by OF ascending (tie-breaker: lexicographic allocated tuple)
    candidates.sort(key=lambda x: (x[0], x[2]))
    return candidates

def try_allocate_candidate(allocated_tuple, node_locks, available_flags):
    """
    Try to acquire locks for all nodes in allocated_tuple (in ascending order).
    If success: mark available_flags False for those nodes and return True.
    If fail: release any acquired locks and return False.
    Uses non-blocking lock acquisition to favor throughput (no waiting).
    """
    acquired = []
    # Acquire locks in ascending order to reduce deadlock probability
    for node in allocated_tuple:
        lock = node_locks[node]
        got = lock.acquire(blocking=False)
        if not got:
            # release previously acquired locks
            for n in acquired:
                node_locks[n].release()
            return False
        acquired.append(node)
    # At this point, all locks acquired: verify availability and mark allocated
    for node in acquired:
        if not available_flags[node]:
            # someone else allocated it between our checks; release all and return False
            for n in acquired:
                node_locks[n].release()
            return False
    # mark as allocated (safe under locks)
    for node in acquired:
        available_flags[node] = False
    # release locks immediately after marking (node now marked busy)
    for n in acquired:
        node_locks[n].release()
    return True

def process_job_and_allocate(job_index, source, jr_job, jb_job, jl_job, V_R, V_B, adjList, numNodes,
                             cut_comb_nodes, cut_sol, node_locks, available_flags, max_retries=1):
    """
    Worker function executed by each thread:
    - compute candidates,
    - attempt to allocate best candidate(s) using per-node locks (non-blocking).
    Returns (job_index, OF_assigned, allocated_nodes_list, feasible_candidates_count)
    """
    candidates = evaluate_candidates(job_index, source, jr_job, jb_job, jl_job, V_R, V_B, adjList, numNodes, cut_comb_nodes, cut_sol)
    v_sol_feasible = len(candidates)
    if v_sol_feasible == 0:
        return job_index, 0, [], 0

    assigned_OF = 0
    assigned_nodes = []

    # Attempt allocation: iterate over candidates in order of OF (best first).
    # On failure to allocate all candidates, optionally retry a limited number of times.
    for attempt in range(max_retries):
        for OF, mask, allocated in candidates:
            # quick availability check (no lock) to skip obviously busy allocations
            if not all(available_flags[n] for n in allocated):
                continue
            success = try_allocate_candidate(allocated, node_locks, available_flags)
            if success:
                assigned_OF = OF
                assigned_nodes = list(allocated)
                # allocation done
                return job_index, assigned_OF, assigned_nodes, v_sol_feasible
        # if we reached here, no candidate allocated in this attempt; optionally loop again
    # no allocation succeeded
    return job_index, 0, [], v_sol_feasible

def run_mna_jobs_parallel_max_throughput(file, numRunnings, r, jr, jb, jl, jo, N_R, N_B, adjList, numNodes,
                                         cut_comb_nodes, cut_sol, max_workers=None, max_retries=1):
    """
    Launches one thread per job (bounded by max_workers) that computes candidates and attempts allocation
    using fine-grained per-node locks. This yields high parallelism and non-deterministic results.
    """
    n_jobs = len(jr)
    if max_workers is None:
        max_workers = max(1, min(64, multiprocessing.cpu_count() * 2))  # allow over-subscription if IO-bound

    # Prepare node locks and availability flags (shared)
    node_locks = [threading.Lock() for _ in range(numNodes)]
    available_flags = [True] * numNodes

    results = [None] * n_jobs
    futures_map = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for job in range(n_jobs):
            src = jo[job]
            fut = executor.submit(process_job_and_allocate,
                                  job, src, jr[job], jb[job], jl[job],
                                  N_R, N_B, adjList, numNodes,
                                  cut_comb_nodes, cut_sol, node_locks, available_flags, max_retries)
            futures_map[fut] = job

        # Collect results as they complete (order of completion is non-deterministic)
        for fut in as_completed(futures_map):
            try:
                job_idx, assigned_OF, assigned_nodes, v_sol_feasible = fut.result()
            except Exception as e:
                print(f"⚠️ Error in job future: {e}")
                continue
            results[job_idx] = (assigned_OF, assigned_nodes, v_sol_feasible)

    # Fill missing results (shouldn't happen) with defaults
    v_all_OF = []
    v_all_nodes = []
    v_all_sol_feasible = []
    for job in range(n_jobs):
        res = results[job]
        if res is None:
            v_all_OF.append(0)
            v_all_nodes.append([])
            v_all_sol_feasible.append(0)
        else:
            of, nodes, vf = res
            v_all_OF.append(of)
            v_all_nodes.append(nodes)
            v_all_sol_feasible.append(vf)

    # Force garbage collection
    gc.collect()
    return v_all_OF, v_all_nodes, v_all_sol_feasible

# Batch runner similar to your original but calling the parallel max-throughput runner
def run_mna_iot_batch(source_dir, target_dir, numRunnings, config_type, max_workers=None, max_retries=1):
    type_suffix = {
        "lightweight": "_light",
        "heavyweight": "_heavy"
    }
    suffix = type_suffix[config_type]
    all_files = os.listdir(source_dir)
    jobs_file = next((f for f in all_files if f.endswith(f"{suffix}.parquet") and "jobs" in f), None)
    info_file = next((f for f in all_files if f.endswith(f"{suffix}_info.parquet")), None)
    if not jobs_file or not info_file:
        print(f"⚠️ .parquet jobs/info files not found for {config_type}.")
        return
    full_base = jobs_file.replace(f"{suffix}.parquet", "")
    print(f"📁 Input base detected: {full_base}")
    nodes_file = next((f for f in all_files if f.endswith("_nodes.parquet")), None)
    if not nodes_file:
        print("⚠️ File nodes.parquet not found.")
        return
    path_nodes = os.path.join(source_dir, nodes_file)
    manifest_file = next((f for f in all_files if f.endswith("_edges_manifest.parquet")), None)
    if not manifest_file:
        print("⚠️ Edge manifest file not found.")
        return
    path_manifest = os.path.join(source_dir, manifest_file)
    path_jobs = os.path.join(source_dir, jobs_file)
    path_info = os.path.join(source_dir, info_file)

    # Read parquet files
    df_jobs = pd.read_parquet(path_jobs)
    df_info = pd.read_parquet(path_info)
    df_nodes = pd.read_parquet(path_nodes)

    df_manifest = pd.read_parquet(path_manifest)
    edges_files = df_manifest["file"].tolist()
    df_edges_list = []
    for f in edges_files:
        file_name = os.path.basename(f)
        full_path = os.path.join(source_dir, file_name)
        if not os.path.exists(full_path):
            print(f"⚠️ Edge file not found: {full_path}")
            continue
        df_edges_list.append(pd.read_parquet(full_path))
    if not df_edges_list:
        raise RuntimeError("❌ No edge partitions were loaded successfully. Check the files in source_dir.")
    df_edges = pd.concat(df_edges_list, ignore_index=True)

    # Extract variables
    jr = df_jobs["jr"].to_numpy()
    jb = df_jobs["jb"].to_numpy()
    jl = df_jobs["jl"].to_numpy()
    jo = df_jobs["jo"].to_numpy()

    V_R = df_nodes["R"].to_numpy()
    V_B = df_nodes["B"].to_numpy()
    V_Busy = df_nodes.get("Busy", pd.Series([False]*len(df_nodes))).to_numpy()
    V_Inactive = df_nodes.get("Inactive", pd.Series([False]*len(df_nodes))).to_numpy()
    numNodes = len(V_R)
    edge_nodes = df_info["edge_nodes"].iloc[0]

    # Build adjacency list
    adjList = [[] for _ in range(numNodes)]
    for _, row in df_edges.iterrows():
        src, tgt, lat = int(row["source"]), int(row["target"]), float(row["latency"])
        adjList[src].append((tgt, lat))

    # Job ordering (same as original)
    c0, c1, c2 = 60, 1, 39
    ordered_lists = sorted(zip(jr, jb, jl, jo),
        key=lambda x: ((c0*(0.253*x[0]) + c1*(0.024*x[1]) - c2*(0.723*x[2])) / (c0+c1+c2)),
        reverse=True)
    jr, jb, jl, jo = map(list, zip(*ordered_lists))

    # Hyperparameters (set by main)
    global_cut_comb_nodes = globals().get("cut_comb_nodes", 500)
    global_cut_sol = globals().get("cut_sol", 2)

    times_execs_cpu = []
    times_execs_real = []
    v_all_OFs = [] # Store the values ​​of all OFs from all runnings

    for r in range(numRunnings):
        start_cpu = time.process_time()
        start_real = time.perf_counter()

        v_all_OF, v_all_nodes, v_all_sol_feasible = run_mna_jobs_parallel_max_throughput(
            full_base, numRunnings, r, jr, jb, jl, jo,
            V_R, V_B, adjList, numNodes,
            global_cut_comb_nodes, global_cut_sol,
            max_workers=max_workers, max_retries=max_retries)

        runtime_cpu = time.process_time() - start_cpu
        runtime_real = time.perf_counter() - start_real

        times_execs_cpu.append(runtime_cpu)
        times_execs_real.append(runtime_real)
        v_all_OFs.append(float(np.sum(v_all_OF))) # <-- saves the total (float)

        def format_float(x): return f"{x:18,.1f}"
        output_path = os.path.join(target_dir, f"results_{r}_DI_MNA_{full_base}{suffix}results_parallel_threads.txt")

        with open(output_path, 'w', encoding="utf-8") as out:
            #--------------- Printing the scenario in the file header -----------------------
            out.write("----------------------------------------------------------------")
            out.write("\n----------------- DI-MNA-Parallel Threads ---------------------")
            out.write("\n---------------------------------------------------------------")
            out.write(f"\n Scenario: {numNodes} nds - {len(jr)} jobs")
            out.write("\n--------------------------------------------------------------\n\n")
            #--------------------------------------------------------------------------------
            out.write(f"Input file: {jobs_file}\n\n")
            #out.write(f"Number of jobs: {len(jr)}\n\n")
            rows = (
                [i, f"[{jr[i]}, {jb[i]}, {jl[i]}, {jo[i]}]", format_float(of), str(sorted(nodes))]
                for i, (of, nodes) in enumerate(zip(v_all_OF, v_all_nodes))
            )
            table = tabulate(rows, headers=["Job", "[ Jr,  Jb,  Jl,  Jo]", "OF", "Allocated nodes"],
                            tablefmt="plain", colalign=("center", "left", "right", "right"))
            out.write(table)
            out.write('\n\n\n')
            out.write(f"Total OF: {format_float(np.sum(v_all_OF)).strip()}\n\n")
            out.write(f"Runtime (CPU): {runtime_cpu:,.5f} sec\n")
            out.write(f"Runtime (REAL): {runtime_real:,.5f} sec\n")

            if r == numRunnings - 1:
                out.write("\n--------------------- Object Function (OF) -------------------------------")
                for i in range(numRunnings):
                    out.write(f"\n {i:4d}:  {v_all_OFs[i]:,.1f}")
                out.write("\n-----------------------------------------------------------------")
                out.write("\n--------------------- Times Execs (CPU) -------------------------------")
                for i in range(numRunnings):
                    out.write(f"\n {i:4d}:  {times_execs_cpu[i]:,.5f}")
                out.write("\n-----------------------------------------------------------------")
                mean_cpu = np.mean(times_execs_cpu)
                sd_cpu = np.std(times_execs_cpu, ddof=(0 if numRunnings == 1 else 1))
                out.write(f"\n mean (CPU):  {mean_cpu:,.5f}")
                out.write(f"\n   sd (CPU):  {sd_cpu:,.5f}")
                out.write("\n-----------------------------------------------------------------\n")

                out.write("\n--------------------- Times Execs (REAL) ------------------------------")
                for i in range(numRunnings):
                    out.write(f"\n {i:4d}:  {times_execs_real[i]:,.5f}")
                out.write("\n-----------------------------------------------------------------")
                mean_real = np.mean(times_execs_real)
                sd_real = np.std(times_execs_real, ddof=(0 if numRunnings == 1 else 1))
                out.write(f"\n mean (REAL):  {mean_real:,.5f}")
                out.write(f"\n   sd (REAL):  {sd_real:,.5f}")
                out.write("\n-----------------------------------------------------------------\n")

                # Save parquet outputs
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
                pq.write_table(pa.Table.from_pandas(df_results_out), f"{prefix}_results_parallel_threads.parquet")

# --------------------- Start of execution (main) -------------------------------
if __name__ == "__main__":
    os.system('cls' if os.name == 'nt' else 'clear')

    # Default hyperparameters
    default_cut_comb_nodes = 500
    default_cut_sol = 2

    default_in_f = r'/mnt/c/Projetos/lscad/MNA/inputs/parquet/1_000nds'
    default_out_f = r'/mnt/c/Projetos/lscad/MNA/DI-MNA/output'

    source_dir = input(f"Enter an source directory path or press ENTER for default (ex: {default_in_f}): ") or default_in_f
    target_dir = input(f"Enter an target directory path or press ENTER for default (ex: {default_out_f}): ") or default_out_f

    config_type = get_config_type()

    cut_comb_nodes = get_positive_integer("Value of the cutoff radius of the number of available node combinations of the IoT network: ", default_cut_comb_nodes)
    cut_sol = get_positive_integer("Value of the cutting radius of the maximum number of nodes considered per solution of each Job: ", default_cut_sol)

    default_runnings = 1
    numRunnings = get_positive_integer("Number of runnings: ", default_runnings)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Tune these for performance: number of threads and allocation aggressiveness
    cpu_count = multiprocessing.cpu_count()
    suggested_workers = max(2, cpu_count * 2)  # allow oversubscription
    print(f"\nDetected CPU cores: {cpu_count}. Suggested worker threads: {suggested_workers}.")
    max_workers = input(f"Max worker threads (press ENTER for suggested {suggested_workers}): ")
    max_workers = int(max_workers) if max_workers else suggested_workers

    # How many allocation attempts to retry per job (increase if many conflicts observed)
    max_retries = input("Max allocation attempts per job (default 1): ")
    max_retries = int(max_retries) if max_retries else 1

    run_mna_iot_batch(source_dir, target_dir, numRunnings, config_type, max_workers=max_workers, max_retries=max_retries)

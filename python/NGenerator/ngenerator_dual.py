''' NGenerator-IoT Network
It generates .json and .parquet input files, in light and heavy configurations, 
containing the values ​​(variables) to represent the node configurations and also the jobs to be performed in this IoT network.

Designed by Murilo Táparo - October 2023
Last updated - May 2025

Last modified: 06/25/2025'''

# -*- coding: utf-8 -*-
import os, gc, time, json, heapq, random
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import psutil
from tqdm import tqdm


def write_edges_chunk(edges_chunk, output_path, chunk_index, job_tag):
    start = time.perf_counter()

    df = pd.DataFrame(edges_chunk, columns=["source", "target", "latency"])
    table = pa.Table.from_pandas(df, preserve_index=False)
    filename = f"{output_path}_{job_tag}jobs_edges_part{chunk_index}.parquet"
    pq.write_table(table, filename)

    elapsed = time.perf_counter() - start
    mem = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    print(f"✅ Chunk {chunk_index} saved ({len(edges_chunk)} edges). "
          f"🕒 Time: {elapsed:.2f}s | 🧠 Memory: {mem:.2f} MB")

    return filename, len(edges_chunk)


def save_checkpoint(filename, chunk_index, edge_files):
    state = {
        "chunk_index": chunk_index,
        "edge_files": edge_files
    }
    with open(filename, "w") as f:
        json.dump(state, f)


def load_checkpoint(filename):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            state = json.load(f)
        print(f"🔁 Checkpoint loaded (chunk {state['chunk_index']})")
        edge_files = [tuple(e) for e in state["edge_files"]]
        return state["chunk_index"], edge_files
    else:
        print("📦 No checkpoints found. Starting from scratch.")
        return 0, []


def generate_graph_with_mst(numNodes, range_RN, range_BN, range_LN, edge_prob, k_neigh=5):
    for i in range(numNodes):
        R = random.randint(*range_RN)
        B = random.randint(*range_BN)
        yield ("node", i, {"R": R, "B": B})

    sampled_edges = set()
    edges_with_latency = []
    for u in range(numNodes):
        neighbors = set()
        while len(neighbors) < k_neigh:
            v = random.randint(0, numNodes - 1)
            if v != u:
                a, b = min(u, v), max(u, v)
                if (a, b) not in sampled_edges:
                    L = random.randint(*range_LN)
                    sampled_edges.add((a, b))
                    edges_with_latency.append((L, a, b))
                    neighbors.add(v)

    parent = list(range(numNodes))
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    def union(u, v):
        pu, pv = find(u), find(v)
        if pu == pv:
            return False
        parent[pu] = pv
        return True

    heap = list(edges_with_latency)
    heapq.heapify(heap)

    used = set()
    count = 0
    while heap and count < numNodes - 1:
        L, u, v = heapq.heappop(heap)
        if union(u, v):
            a, b = min(u, v), max(u, v)
            used.add((a, b))
            yield ("edge", (a, b), {"L": L})
            count += 1

    components = len(set(find(i) for i in range(numNodes)))
    if components > 1:
        print(f"⚠️ Incomplete MST: {components} components. Connecting...")
        attempts = 0
        while count < numNodes - 1 and attempts < 5_000_000:
            u, v = random.sample(range(numNodes), 2)
            if find(u) != find(v):
                a, b = min(u, v), max(u, v)
                if (a, b) not in used:
                    L = random.randint(*range_LN)
                    if union(u, v):
                        used.add((a, b))
                        yield ("edge", (a, b), {"L": L})
                        count += 1
            attempts += 1

    max_edges = (numNodes * (numNodes - 1)) // 2
    target_edges = int(edge_prob * max_edges)
    extra_needed = max(0, target_edges - count)

    extras_added = 0
    attempts = 0
    while extras_added < extra_needed and attempts < 10 * extra_needed:
        u, v = random.sample(range(numNodes), 2)
        a, b = min(u, v), max(u, v)
        if (a, b) in used:
            attempts += 1
            continue
        L = random.randint(*range_LN)
        yield ("edge", (a, b), {"L": L})
        used.add((a, b))
        extras_added += 1

def save_nodes_graph_to_parquet(
    output_path, numNodes, edge_nodes, range_RN, range_BN, range_LN,
    jr, jb, jl, jo, V_Busy, V_Inactive,
    edge_prob, numJob_light=None, numJob_heavy=None,
    chunk_limit=1_000_000, k_neigh=5,
    checkpoint_file="checkpoint.json"):

    V_R = [0] * numNodes
    V_B = [0] * numNodes
    adjList = [[] for _ in range(numNodes)]

    chunk_index, edge_files = load_checkpoint(checkpoint_file)

    generator = generate_graph_with_mst(numNodes, range_RN, range_BN, range_LN, edge_prob, k_neigh)
    edges_buffer = []

    print("🚀 Starting simulation with checkpointing...")
    pbar = tqdm(generator, desc="🔄 Graph processing", unit="item")

    for item_type, idx, attr in pbar:
        if item_type == "node":
            V_R[idx] = attr["R"]
            V_B[idx] = attr["B"]
        elif item_type == "edge":
            src, tgt = idx
            lat = attr["L"]
            edges_buffer.append((src, tgt, lat))
            edges_buffer.append((tgt, src, lat))

            adjList[src].append([tgt, lat])
            adjList[tgt].append([src, lat])

            if len(edges_buffer) >= chunk_limit:
                chunk_data = edges_buffer[:]
                edges_buffer.clear()
                print(f"📝 Salvando chunk {chunk_index} com {len(chunk_data)} arestas")
                filename, n_edges = write_edges_chunk(chunk_data, output_path, chunk_index, numJob_heavy)
                edge_files.append((filename, n_edges))
                save_checkpoint(checkpoint_file, chunk_index + 1, edge_files)
                chunk_index += 1
                gc.collect()

    if edges_buffer:
        chunk_data = edges_buffer[:]
        edges_buffer.clear()
        print(f"📝 Saving final chunk {chunk_index} with {len(chunk_data)} edges")
        filename, n_edges = write_edges_chunk(chunk_data, output_path, chunk_index, numJob_heavy)
        edge_files.append((filename, n_edges))
        save_checkpoint(checkpoint_file, chunk_index + 1, edge_files)
        chunk_index += 1
        gc.collect()

    # Save the Manifesto
    df_manifest = pd.DataFrame({
    "file": [os.path.basename(f) for f, n in edge_files],
    "num_edges": [n for f, n in edge_files]
    })
    pq.write_table(pa.Table.from_pandas(df_manifest),f"{output_path}_edges_manifest.parquet",compression="snappy")
    df_manifest.to_csv(f"{output_path}_edges_manifest.csv",index=False)

    # Nodes
    df_nodes = pd.DataFrame({
        "node": np.arange(numNodes),
        "R": V_R,
        "B": V_B,
        "Busy": V_Busy,
        "Inactive": V_Inactive
    })
    pq.write_table(pa.Table.from_pandas(df_nodes),
                   f"{output_path}_{numJob_heavy}jobs_nodes.parquet")

    # Jobs and info
    df_info = pd.DataFrame({"edge_nodes": [edge_nodes], "total_nodes": [numNodes]})
    df_jobs = pd.DataFrame({"jr": jr, "jb": jb, "jl": jl, "jo": jo})

    def save_json_config(fname, edge_nodes, jr_list, jb_list, jl_list, jo_list,
                         V_R, V_B, V_Busy, V_Inactive, adjList):
        try:
            with open(fname, "w") as json_file:
                json_file.write('{\n')
                json_file.write(f'  "edge_nodes": {edge_nodes},\n')
                json_file.write(f'  "jr": [{", ".join(map(str, jr_list))}],\n')
                json_file.write(f'  "jb": [{", ".join(map(str, jb_list))}],\n')
                json_file.write(f'  "jl": [{", ".join(map(str, jl_list))}],\n')
                json_file.write(f'  "jo": [{", ".join(map(str, jo_list))}],\n')
                json_file.write(f'  "V_R": [{", ".join(map(str, V_R))}],\n')
                json_file.write(f'  "V_B": [{", ".join(map(str, V_B))}],\n')
                json_file.write(f'  "V_Busy": [{", ".join(map(str, V_Busy))}],\n')
                json_file.write(f'  "V_Inactive": [{", ".join(map(str, V_Inactive))}],\n')
                json_file.write('  "adjList": [\n')

                for i, adj in enumerate(adjList):
                    line = "    [" + ", ".join(f"[{tgt}, {lat}]" for tgt, lat in adj) + "]"
                    if i < len(adjList) - 1:
                        json_file.write(f"{line},\n")
                    else:
                        json_file.write(f"{line}\n")

                json_file.write('  ]\n')
                json_file.write('}\n')

            print(f"✅ JSON configuration file saved successfully: {fname}")

        except IOError:
            raise Exception(f'❌ Error creating or opening file {fname} for writing.')

    # Light configuration
    if numJob_light:
        pq.write_table(pa.Table.from_pandas(df_jobs.head(numJob_light)),
                       f"{output_path}_{numJob_light}jobs_light.parquet")
        pq.write_table(pa.Table.from_pandas(df_info),
                       f"{output_path}_{numJob_light}jobs_light_info.parquet")
        save_json_config(
            f"{output_path}_{numJob_light}jobs_light_config.json",
            edge_nodes,
            df_jobs.head(numJob_light)["jr"].tolist(),
            df_jobs.head(numJob_light)["jb"].tolist(),
            df_jobs.head(numJob_light)["jl"].tolist(),
            df_jobs.head(numJob_light)["jo"].tolist(),
            V_R, V_B, V_Busy, V_Inactive, adjList
        )

    # Heavy configuration
    if numJob_heavy:
        pq.write_table(pa.Table.from_pandas(df_jobs),
                       f"{output_path}_{numJob_heavy}jobs_heavy.parquet")
        pq.write_table(pa.Table.from_pandas(df_info),
                       f"{output_path}_{numJob_heavy}jobs_heavy_info.parquet")
        save_json_config(
            f"{output_path}_{numJob_heavy}jobs_heavy_config.json",
            edge_nodes,
            df_jobs["jr"].tolist(),
            df_jobs["jb"].tolist(),
            df_jobs["jl"].tolist(),
            df_jobs["jo"].tolist(),
            V_R, V_B, V_Busy, V_Inactive, adjList
        )

    print("\n✅ Simulation completed successfully!")

def get_positive_integer(prompt, default):
    while True:
        try:
            user_input = input(f"{prompt} ({default}): ")
            value = int(user_input) if user_input else default
            if value > 0:
                return value
            else:
                print("⚠️ ERROR: The number must be a positive value. Please try again.")
        except ValueError:
            print("⚠️ ERROR:: Invalid input. Please enter an integer.")


def main():
    os.system('cls' if os.name == 'nt' else 'clear')

    default_out_f = r'C:\Users\Murilo\Documents\DOUTORADO\UFMS-FACOM\Material-Ricardo\Encontros-Ricardo\Encontro_57-05-08-2025\NGenerator\output'
    output_folder = input(f"Enter a directory path or press ENTER for default (ex: {default_out_f}): ") or default_out_f
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # -------------------------------- Nodes parameters -------------------------------------------
    default_node = 15
    default_EN = (7, 14)
    default_RN = (20, 700) # (20, 400)
    default_BN = (20, 5000) # (20, 4000)
    default_LN = (2, 100)

    print("---------------- IoT Network Parameters ---------------------------\n")
    numNodes = get_positive_integer("Number of nodes", default_node)
    input_EN = input(f"Range for edge nodes (ex: {default_EN[0]} {default_EN[1]}): ")
    range_EN = tuple(map(int, input_EN.split())) if input_EN else default_EN
    edge_nodes = list(range(range_EN[0], range_EN[1] + 1))
    input_RN = input(f"Range for Resource (R) (ex: {default_RN[0]} {default_RN[1]}): ")
    range_RN = tuple(map(int, input_RN.split())) if input_RN else default_RN
    input_BN = input(f"Range for Bandwidth (B) (ex: {default_BN[0]} {default_BN[1]}): ")
    range_BN = tuple(map(int, input_BN.split())) if input_BN else default_BN
    input_LN = input(f"Range for Latency (L) (ex: {default_LN[0]} {default_LN[1]}): ")
    range_LN = tuple(map(int, input_LN.split())) if input_LN else default_LN
    print("-------------------------------------------------------------------")

    V_Busy = np.zeros(numNodes).astype(int)
    V_Inactive = np.zeros(numNodes).astype(int)

    # -------------------------------- Jobs parameters ------------------------------------------
    default_RJ = (10, 350)
    default_BJ = (10, 2500)
    default_LJ = (30, 250)

    print("---------------------- Job Parameters -----------------------------\n")
    lightweight = 0.2
    heavyweight = 0.7
    default_numJob_light = int(lightweight * numNodes)
    default_numJob_heavy = int(heavyweight * numNodes)
    numJob_light = get_positive_integer("Number of jobs in lightweight configuration", default_numJob_light)
    numJob_heavy = get_positive_integer("Number of jobs in heavyweight configuration", default_numJob_heavy)

    input_RJ = input(f"Range for Resource (R) (ex: {default_RJ[0]} {default_RJ[1]}): ")
    range_RJ = tuple(map(int, input_RJ.split())) if input_RJ else default_RJ
    input_BJ = input(f"Range for Bandwidth (B) (ex: {default_BJ[0]} {default_BJ[1]}): ")
    range_BJ = tuple(map(int, input_BJ.split())) if input_BJ else default_BJ
    input_LJ = input(f"Range for Latency (L) (ex: {default_LJ[0]} {default_LJ[1]}): ")
    range_LJ = tuple(map(int, input_LJ.split())) if input_LJ else default_LJ

    jr = [random.randint(*range_RJ) for _ in range(numJob_heavy)]
    jb = [random.randint(*range_BJ) for _ in range(numJob_heavy)]
    jl = [random.randint(*range_LJ) for _ in range(numJob_heavy)]
    jo = [random.choice(edge_nodes) for _ in range(numJob_heavy)]
    print("-------------------------------------------------------------------")

    output_base = os.path.join(output_folder, f"input_{numNodes}nds")

    checkpoint_path = "checkpoint.json"
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"🗑️ Previous checkpoint '{checkpoint_path}' removed.")

    try:
        save_nodes_graph_to_parquet(
            output_path=output_base,
            numNodes=numNodes,
            edge_nodes=edge_nodes,
            range_RN=range_RN,
            range_BN=range_BN,
            range_LN=range_LN,
            jr=jr,
            jb=jb,
            jl=jl,
            jo=jo,
            V_Busy=V_Busy.tolist(),
            V_Inactive=V_Inactive.tolist(),
            edge_prob=0.15, # 15% # recommended value (dense networks) for 1e-6 performance
            numJob_light=numJob_light,
            numJob_heavy=numJob_heavy,
            chunk_limit=1_000_000,
            k_neigh=5, # No. of neighbors per node
            checkpoint_file=checkpoint_path
        )
    except Exception as e:
        raise Exception(f"⚠️ ERROR during saving Parquet files: {e}") from e

    print(f"\n✅ Data saved successfully in: {output_base}_*.parquet")


if __name__ == "__main__":
    main()

import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

def jobs_configs(nds, all_files):
    type_suffix = {"lightweight": "_light", "heavyweight": "_heavy"}

    for cfg, suffix in type_suffix.items():
        # finds all job files for this configuration
        job_files = [f for f in all_files 
                     if f.startswith(f"input_{nds}") and
                        f.endswith(f"{suffix}.parquet") and
                        "jobs" in f]
        
        info_files = [f for f in all_files 
                     if f.startswith(f"input_{nds}") and
                        f.endswith(f"{suffix}_info.parquet") and
                        "jobs" in f]

        if not job_files:
            print(f"⚠️ No jobs files found for {nds} {cfg}")
            continue

        yield job_files[0], info_files[0], suffix
    

def networks(all_files):
    
    # manifest_files = sorted([os.path.join(f) for f in all_files if f.endswith("_edges_manifest.parquet")])
    nodes_files = sorted([f for f in all_files if f.endswith("_nodes.parquet")])
    
    for i in range(len(nodes_files)):
        nds = nodes_files[i].split("_")[1]
        edge_files = [f for f in all_files if f"{nds}_edges_part" in f]
        yield nds, nodes_files[i], edge_files
    

def get_file_paths(source_dir):

    all_files = os.listdir(source_dir)

    for nds, nodes_file, edge_files in networks(all_files):
        for j_cfg, j_info, suffix in jobs_configs(nds, all_files):
            yield (nds, nodes_file, edge_files, j_cfg, j_info, suffix)

def full_base_name(jobs_file, suffix):
    splited = jobs_file.split("_")
    return f"{splited[1]}_{splited[2]}{suffix}"

def read_input(args, source_dir):

    nds, path_nodes, edge_files, path_jobs, path_info, suffix = args

    path_nodes = os.path.join(source_dir, path_nodes)
    path_jobs = os.path.join(source_dir, path_jobs)
    path_info = os.path.join(source_dir, path_info)


    with ThreadPoolExecutor() as executor:

        future_nodes = executor.submit(pd.read_parquet, path_nodes)
        future_jobs = executor.submit(pd.read_parquet, path_jobs)
        future_info = executor.submit(pd.read_parquet, path_info)

        df_jobs = future_jobs.result()
        df_info = future_info.result()
        df_nodes = future_nodes.result()

    df_edges_list = []
    
    for f in edge_files:
        df_edges_list.append(pd.read_parquet(os.path.join(source_dir, f)))

    if not df_edges_list:
        print(f"❌ No edge partitions loaded for {nds}{suffix}")
        return None
    
    elif len(df_edges_list) != 1:
        df_edges = pd.concat(df_edges_list, ignore_index=True)
    else:
        df_edges = df_edges_list[0]

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

    print("colocou nas variáveis")

    adjList = [[] for _ in range(numNodes)]

    for row in df_edges[['source', 'target', 'latency']].itertuples(index=False, name=None):
        src, tgt, lat = row
        adjList[int(src)].append((int(tgt), lat))
    
    return jr, jb, jl, jo, V_R, V_B, V_Busy, V_Inactive, numNodes, edge_nodes, adjList
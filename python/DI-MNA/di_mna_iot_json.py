''' MNA-IoT-DI
It runs, from a .json input file, generating solutions (stored in .txt files), 
as well as a new .json output file containing the parameters of the best solution 
found by this algorithm.

Designed by Murilo Táparo - November 2023

Last modified: 06/09/2025'''
# -*- coding: utf-8 -*-
# MNA-IoT mem optimization

import os
import json
import time
import numpy as np
from itertools import combinations
import heapq
import gc
from tabulate import tabulate

#---------------- Clear terminal window before program execution ---------------
os.system('cls' if os.name == 'nt' else 'clear')
# On Windows (nt) systems, use the cls command
# On Unix/Linux and MacOS systems, use the clear command
#-------------------------------------------------------------------------------
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
    for k in range(1, cut_sol + 1):  # combinations from 1 to cut_sol active nodes
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

def get_positive_integer(prompt, default):
    while True:
        try:
            user_input = input(f"{prompt} ({default}): ")
            # If the user does not enter anything, it uses the default value.
            value = int(user_input) if user_input else default

            if value > 0:
                return value  # Returns the value if it is positive
            else:
                print("Erro: The number must be a positive value. Please try again.")
        except ValueError:  # Checks if value is an integer
            print("Erro: Invalid input. Please enter an integer.")

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
        print_header(file, numRunnings, r, job, jr, jb, jl, jo) 
        source = jo[job]  # Stores the position value of the source node
        # Discounts the initial node connection time in the calculation (only 1 time for each job)
        l_job = jl[job] - t_c
        N_L = get_latencies(source, adjList, numNodes)

        Min_FX = INF
        better_mask = [0] * numNodes
        v_sol_feasible = 0 # Nº feasible solutions
        # Considers the cut_comb_nodes combinations, according to DI, for each job
        cut_comb_nodes = 500 # Parameter None = network (total)
        cut_sol = 2 # Considers a maximum of 2 nodes when allocating a job
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

    files = [f for f in os.listdir(source_dir) if f.endswith('.json')]
    for file in files:
        source_path = os.path.join(source_dir, file)
        full_name = os.path.splitext(file)[0]
        # Stores the total time of each running from a given input file
        times_execs = []
        
        with open(source_path, 'r') as f:
            json_data = json.load(f)

        jr = np.array(json_data['jr'])
        jb = np.array(json_data['jb'])
        jl = np.array(json_data['jl'])
        jo = np.array(json_data['jo'])
        N_R = np.array(json_data['V_R'])
        N_B = np.array(json_data['V_B'])
        adjList = json_data['adjList']
        edge_nodes = json_data['edge_nodes']  # Nodes that are part of the edge (source nodes)- user defined (set to zero index)
        numNodes = len(N_R)

        #-------------------------------------------------------------------------------------------------------------------------------------------
        #----- Sort the values ​​of: jr, jb, jl, and jo simultaneously, by decreasing value of jb ----------------------------------------------------
        #---- Demand Index (DI) using weights: 60, 1 and 39, in parameters: jr, jb and jl, respectively --------------------------------------------
        #---- Coefficient values: c0=60, c1=1 and c2=39 --------------------------------------------------------------------------------------------
        c0 = 60
        c1 = 1
        c2 = 39
        ordered_lists = sorted(zip(jr, jb, jl, jo), key=lambda x: ((c0*x[0] + c1*x[1] - c2*x[2]) / (c0+c1+c2)), reverse=True)
        # reverse=False (ascending order) and reverse=True (descending order)
        # Sort the lists simultaneously (zip command) by increasing value of jl
        #---- Normalized coefficient values ​​of: R=0.253, B=0.024 and L=0.723 corresponding to the variables: x[0], x[1] and x[2], respectively -----
        ordered_lists = sorted(zip(jr, jb, jl, jo), key=lambda x: ((c0*(0.253*x[0]) + c1*(0.024*x[1]) - c2*(0.723*x[2]))/(c0+c1+c2)), reverse=True)

        # Unpacking the sorted values ​​and assigning them to the same lists: jr, jb, jl, and jo
        jr, jb, jl, jo = map(list, zip(*ordered_lists))
        #-------------------------------------------------------------------------------------------------------------------------------------------

        for r in range(numRunnings):
            
            # Starts counting the time of each execution
            start = time.process_time() 
            v_all_OF, v_all_nodes, v_all_sol_feasible = run_mna_jobs(file, numRunnings, r, jr, jb, jl, jo, N_R, N_B, adjList, numNodes)
            # The time count for each execution ends
            runtime = time.process_time() - start

            # American format with thousands separator, 1 decimal place, width 18
            def format_float(x):
                return f"{x:18,.1f}" 

            output_path= os.path.join(target_dir, f"results_{str(r)}_MNA_IoT_{full_name}.txt")
            try:
                fileID = open(output_path, 'w', encoding="utf-8")
            except IOError:
                raise Exception('Error creating or opening file for writing.')

            with open(output_path, 'w') as out:
                out.write(f"Input file: {file}\n\n")
                out.write(f"Number of jobs: {len(jr)}\n\n")

                rows = (
                    [
                        i,
                        f"[{jr[i]}, {jb[i]}, {jl[i]}, {jo[i]}]",
                        format_float(of),
                        str(sorted(nodes))
                    ]
                    for i, (of, nodes) in enumerate(zip(v_all_OF, v_all_nodes))
                )

                table = tabulate(
                    rows,
                    headers=["Job", "[ Jr,  Jb,  Jl,  Jo]", "OF     ", "Allocated nodes"],
                    tablefmt="plain",
                    colalign=("center", "left", "right", "right")
                )

                out.write(table)
                out.write('\n\n\n')
                # `.strip()` removes any whitespace (or newline characters) from the beginning and end of the resulting string.
                out.write(f"Total OF: {format_float(np.sum(v_all_OF)).strip()}\n\n")
                out.write(f"Runtime: {runtime:,.5f} sec\n") 

                times_execs.append(runtime)
                #---------------------------------------------------------------------------------------------
                # Prints to the file, only in the last execution, all the times (s) of the OF values ​​obtained
                if (r==(numRunnings-1)):
                    out.write("\n--------------------- Times Execs -------------------------------")
                    for i in range(numRunnings):
                        # Aligns 4 digits of the run index and ensures that numeric values ​​are at least 16 characters wide
                        out.write(f"\n {i:4d}:  {times_execs[i]:,.5f}")
                    out.write("\n-----------------------------------------------------------------")  
                    mean = np.mean(times_execs)
                    if (numRunnings==1):
                        sd = np.std(times_execs, ddof=0)  # For population (n), ddof=0 (Delta Degrees of Freedom)
                    else:    
                        sd = np.std(times_execs, ddof=1)  # For population (n-1), ddof=1 (Delta Degrees of Freedom)
                    out.write(f"\n mean:  {mean:,.5f}")
                    out.write(f"\n   sd:  {sd:,.5f}")   
                    out.write("\n-----------------------------------------------------------------\n")
                    #---------------------------------------------------------------------------------------------------------
                    # Stores the initial values ​​of V_R, V_B, V_Busy and V_Inactive in the output .json file (NSGA3_hyb Input)
                    V_R = np.array(json_data['V_R'])
                    V_B = np.array(json_data['V_B'])
                    V_Busy = np.array(json_data['V_Busy'])
                    V_Inactive = np.array(json_data['V_Inactive'])
                    #---------------------------------------------------------------------------------------------------------
                    #--------------------------------------------------------------------------------------------------------- 
                    # Close the file: results.txt
                    fileID.close()
                    #----------------------------------------------------------------------------------------------------------
                    #--------------------------------------------------------------------------------------------- 
                    # Save IoT network configuration to .json output file (run on NSGA-III_Hybrid)
                    # Specify the paths to save the .json output file
                    name_file_json = f'{target_dir}/MNA_IoT_{full_name}.json'
                    
                    try:
                        fileIDJSON = open(name_file_json, 'w', encoding="utf-8")
                    except IOError:
                        raise Exception('Error creating or opening file for writing.')
                    #--------------------------------------------------------------------------------------------
                    # Formatting data for printing in the requested format
                    formatted_output = (
                        '{\n'  # Adds the opening key of the JSON object
                        f'  "edge_nodes": {edge_nodes},\n'
                        f'  "jr": [{", ".join(map(str, jr))}],\n' 
                        f'  "jb": [{", ".join(map(str, jb))}],\n' 
                        f'  "jl": [{", ".join(map(str, jl))}],\n' 
                        f'  "jo": [{", ".join(map(str, jo))}],\n' 
                        f'  "V_R": [{", ".join(map(str, V_R))}],\n'
                        f'  "V_B": [{", ".join(map(str, V_B))}],\n'
                        f'  "V_Busy": [{", ".join(map(str, V_Busy))}],\n'
                        f'  "V_Inactive": [{", ".join(map(str, V_Inactive))}],\n'
                        f'  "v_all_OF": [{", ".join(map(str, v_all_OF))}],\n'
                        f'  "v_all_nodes": [{", ".join(map(str, v_all_nodes))}],\n'
                        f'  "Min_FX": [{", ".join(map(str, [sum(v_all_OF)]))}],\n'
                        f'  "v_all_sol_feasible": [{", ".join(map(str, v_all_sol_feasible))}],\n'
                        f'  "adjList": [\n'
                    )

                    # Save the new content in the input_NSGA3.json file, with encoding="utf-8" for special characters
                    try:
                        with open(name_file_json, "w", encoding="utf-8") as fjson:
                            fjson.write(formatted_output)  # Writes the header of the JSON structure

                            # Saving the adjacency list in the desired format to the output.json file
                            # Iterate over neighbors and add comma only where needed
                            for i, neighbors in enumerate(adjList):
                                if i < len(adjList) - 1:
                                    fjson.write(f"    {json.dumps(neighbors)},\n")  # Adds comma after each element except the last one
                                else:
                                    fjson.write(f"    {json.dumps(neighbors)}\n")   # The last element does not receive a comma

                            fjson.write("  ]\n")  # Close adjList
                            fjson.write("}")      # Close principal JSON 

                    except Exception as e:
                        print(f"Error saving file: {e}")
                    #----------------------------------------------------------------------------------------------------------------------- 
                    # Close the file: input_NSGA3.json
                    fileIDJSON.close()
                    #------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------
# Start of execution (main):

# Set the directory path where the .json files are located
# Setting default values
default_in_f = r'C:\Users\Murilo\Documents\DOUTORADO\UFMS-FACOM\Material-Ricardo\Encontros-Ricardo\Encontro_57-05-08-2025\25nds\input'
default_out_f = r'C:\Users\Murilo\Documents\DOUTORADO\UFMS-FACOM\Material-Ricardo\Encontros-Ricardo\Encontro_57-05-08-2025\25nds\output'

# Prompts user for directory path or uses default value if ENTER is pressed
source_dir = input(f"Enter an source directory path or press ENTER for default (ex: {default_in_f}): ") or default_in_f

target_dir = input(f"Enter an target directory path or press ENTER for default (ex: {default_out_f}): ") or default_out_f

# Setting default values
default_runnings = 1

# Number of runnings of a given configuration
numRunnings = get_positive_integer("Number of runnings: ", default_runnings)
# Create the output folder if it doesn't exist; keep if it already exists
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

run_mna_iot_batch(source_dir, target_dir, numRunnings)
""" NSGA-III Enhanced
This is similar to the standard NSGA-III. However, it always considers the last best solution,
the lowest value of the objective function (OF) as the final answer.

Designed by Murilo Táparo - January 2025

Last modified: 06/30/2025 """

# -*- coding: utf-8 -*-
# Commented out IPython magic to ensure Python compatibility.
# %pip -q install deap
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use backend 'Agg' to prevent displaying plots
from deap import base, creator, tools, algorithms
from deap.tools.indicator import hv
from functools import partial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import heapq
import random
import json
import os
import sys
from sys import argv

sys.path.append(os.path.join(os.path.dirname(sys.path[0]), "commandLine"))

import commandLine as cl

#---------------- Clear terminal window before program execution ---------------
os.system('cls' if os.name == 'nt' else 'clear')
# On Windows (nt) systems, use the cls command
# On Unix/Linux and MacOS systems, use the clear command
#-------------------------------------------------------------------------------
# -------------------- Infinite value for int ----------------------------------
INF = 10**12  # Infinite Constant for int
# ------------------- Function Declarations ------------------------------------

# NSGA3 Hyperparameter Default Values
default_params = {
  'population_size': 80, # The number of individuals in each generation of the population. Typical values: 200 to 500 initially and increase only if you notice weak convergence or a poorly explored front.
  'generations': 120, # The total number of generations the algorithm will execute. Each generation involves selection, crossover, and mutation to create a new population.
  'cx_prob': 0.7, # The probability of crossover between two individuals. This value indicates the fraction of the population selected to perform the crossover operation.
  'indpb': 0.1 # The probability of mutation for each gene (or variable) of an individual. If an individual is selected for mutation, this probability determines the chance of each gene being altered.
}              # Typical value: 1/n_genes.

# Function to read parameters from JSON file
def read_hyper_parameters(directory):
    file_name = 'hyper_parameters.json'
    file_path = os.path.join(directory, file_name)

    #print(f"Checking if file {file_name} exists in path: {file_path}")

    if os.path.exists(file_path):
        #print(f"File {file_name} found. Trying to read parameters.")
        with open(file_path, 'r') as file:
            try:
                params = json.load(file)
                #print(f"Parameters read from file: {params}")
                return params
            except json.JSONDecodeError as e:
                #print(f"Error reading JSON file: {e}")
                return default_params  # If there is an error, return the default values
    else:
        #print(f"File {file_name} not found in directory {directory}. Using default values.")
        return default_params  # If the file does not exist, use the default values

def mark_busy(cont_nos_rede, v_nodes_network, V_Busy):
    pos = 0  # Specifies the node nº in the network
    for i in range(cont_nos_rede):
        pos = v_nodes_network[i]
        V_Busy[pos] = 1  # Assigns the node at position i as occupied.
    return V_Busy


# Function to check if solution A dominates solution B
def dominates(A, B):
    return all(a <= b for a, b in zip(A, B)) and any(a < b for a, b in zip(A, B))

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

# Function to remove edges surrounding the inactive node
def remove_edges_involving_node(adjList, inactive_node):
    for i in range(len(adjList)):
        if i != inactive_node:
            adjList[i] = [connection for connection in adjList[i] if connection[0] != inactive_node]

# Define the objective function
def objectiveFunction(individual, l_job, job, jr, jb, N_R, N_B, N_L, v_sol_f1, v_sol_f2, v_sol_f3, v_sol_fx, v_sol_ind, numNodes):

    # Initializing the value of the lists of objective function components
    f = np.zeros(3)
    individual = np.array(individual) # Convert individual to np.array
    # Initializing value of the objective function
    OF = 0

    f[0] = (np.sum(N_R[:numNodes] * individual[:numNodes])) - jr[job] # Minimize
    f[1] = (np.sum(N_B[:numNodes] * individual[:numNodes])) - jb[job] # Minimize
    f[2] = l_job - (np.sum(N_L[:numNodes] * individual[:numNodes])) # Maximize

    if (f[0] >= 0 and f[1] >= 0 and f[2] >= 0):
        # The calculation of OF considers the attributes: R, B and L
        OF = f[0]**2 + f[1]**2 - f[2]
        #----------- Storing the solutions -----------------------
        v_sol_f1.append(f[0])
        v_sol_f2.append(f[1])
        v_sol_f3.append(f[2])
        v_sol_fx.append(OF)
        v_sol_ind.append(individual)

    # Returns the values ​​of each of the objectives
    return f[0], f[1], f[2]

# Constraint function - returns values: c[0], c[1] e c[2] <= 0
def jobConstraints(individual, l_job, job, jr, jb, N_R, N_B, N_L, numNodes, OF_previous_best):

    c = np.zeros(4)  # 4 Constraints
    individual = np.array(individual) # Convert individual to np.array
    c[0] = jr[job] - (np.sum(N_R[:numNodes] * individual[:numNodes]))
    c[1] = jb[job] - (np.sum(N_B[:numNodes] * individual[:numNodes]))
    c[2] = (np.sum(N_L[:numNodes] * individual[:numNodes])) - l_job

    # New objective function calculation (same as objectiveFunction)
    f0 = (np.sum(N_R[:numNodes] * individual[:numNodes])) - jr[job]
    f1 = (np.sum(N_B[:numNodes] * individual[:numNodes])) - jb[job]
    f2 = l_job - (np.sum(N_L[:numNodes] * individual[:numNodes]))

    OF = f0**2 + f1**2 - f2

    # New constraint: OF cannot be greater than OF_previous_best
    c[3] = OF - OF_previous_best

    # The return values ​​of c[0], c[1], c[2] and c[3] <=0
    return c[0], c[1], c[2], c[3]

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

# -------------------------- End Functions --------------------------------------

# -------------------------- Main Function --------------------------------------
def main(path_input, path_output, cut_sol, numRunnings):
 #-------------------------------------------------------------------------------
 # Load input data from a JSON file
 with open(path_input, 'r') as f:
      data = json.load(f)
 #-------------------------------------------------------------------------------
 #-----------------Reading variables from the input file-------------------------
 # ---------------------------- Job Input Parameters ----------------------------
 jr = np.array(data['jr'])
 jb = np.array(data['jb'])
 jl = np.array(data['jl'])
 jo = np.array(data['jo'])
 # ------------------------------------------------------------------------------
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
 # ------------------------------------------------------------------------------
 n_jobs = len(jr)  # Nº of jobs
 # t_c = time_connection = 1 (1 ms) is the connection time of any IoT application to the source node of the network
 t_c = 1
 #-------------------------------------------------------------------------------------------------------------------------------------------
 Min_FX = INF # The initial value of Min_FX = INF
 OF_execs = [] # Stores all best OF values ​​from all runs
 times_execs = [] # Stores the time (s) values ​​of all executions
 v_all_nodes_execs = [[] for _ in range(numRunnings)] # Stores all best node values ​​from all runs
 mean = 0  # Stores the average time value of all executions
 sd = 0 # Stores the standard deviation value of the time of all runnings
 # ===========================================================================================
 # Initialization of "best solution" variables (before any conditional comparison)
 # This prevents IndexError in the first iteration or if no feasible solution is found
 # ===========================================================================================
 v_all_sol_feasible_better = [[] for _ in range(n_jobs)]    # feasible solutions per job
 v_all_OF_better = [0.0 for _ in range(n_jobs)]              # objective function values per job
 v_all_nodes_better = [[] for _ in range(n_jobs)]            # allocated nodes per job
 v_all_times_better = [0.0 for _ in range(n_jobs)]           # processing times per job
 num_empty_sublists_nsga3_better = n_jobs # Stores the number of empty sublists of the best solution found
 # -----------------------------------------------------------------------------

 for r in range(numRunnings):

  input_file_name =  os.path.basename(path_input).split('.')[0]

  # Specify the path to save the file
  file_path = f'{path_output}/results_{str(r)}_{input_file_name}.txt'

  output_file_name = os.path.basename(file_path).split('.')[0]
  name_file = output_file_name +'.txt'

  try:
    fileID = open(file_path, 'w')
  except IOError:
    raise Exception('Error creating or opening file for writing.')
  l_job = 0 # Latency Job: variable that stores the latency value of a job minus t_c (time connection)
  # --------------------------------------------------------------------------------
  # Create n_jobs empty lists to store the first 3 solutions of each job and,
  # then perform all the combinations to find the smallest value of OF
  #SS = [[] for _ in range(n_jobs)]  # Create n_jobs empty lists inside SS
  # Cut is the variable that indicates the number of solutions for each job to be considered
  # in the value of the final OF solution
  #cut = 3
  # --------------------------------------------------------------------------------
  # Initial values ​​of vectors R and B
  N_R = np.array(data['V_R'])
  N_B = np.array(data['V_B'])
  #----------------- No. of IoT network nodes (Variables)--------------------------
  numNodes = len(N_R)
  #--------------------------------------------------------------------------------
  #--------------- Printing the scenario in the file header -----------------------
  fileID.write("--------------------------------------------------------------")
  fileID.write("\n-------------------- NSGA-III Enhanced -----------------------")
  fileID.write("\n--------------------------------------------------------------")
  fileID.write(f"\n Scenario: {numNodes} nds - {n_jobs} jobs")
  fileID.write("\n--------------------------------------------------------------")
  #--------------------------------------------------------------------------------
  # Initial node values ​​regarding the status: free/busy ---------------------------
  # The values ​​represent: 0 (free node) and 1 (busy node) -------------------------
  V_Busy = np.array(data['V_Busy'])
  #--------------------------------------------------------------------------------
  # Initial values ​​of the nodes regarding the state: Active/Inactive --------------
  # The values ​​represent: 0 (Active node) and 1 (Inactive node) -------------------
  V_Inactive = np.array(data['V_Inactive'])
  #--------------------------------------------------------------------------------
  # Counter of the total number of allocated nodes of the entire network
  cont_nos_rede = 0
  # Stores all the indexes of the allocated nodes of the entire network
  v_nodes_network = []
  #--------- Representation of IoT network with adjacency list as a cell ---------
  #-------- IoT network representation with adjacency list as a cell -------------
  # Initializing adjacency lists as a list
  adjList = data['adjList']

  #---- Step 1: Convert the adjacency list to an edge list ----
  # Initializing the edge list
  edges = []
  #------------------------------------------------------------------------------
  # Stores the smallest values ​​found by NSGA-III for each Job in a given execution
  best_sol_population_job = []
 #------------------------------------------------------------------------------
  # Making a given node inactive: replacing the adjacency list with a loop to itself and infinite latency
  for i in range(numNodes):
      if V_Inactive[i]: # 0 = False e 1 = True
        adjList[i] = [[i, INF]]
        # Remove surrounding edges from a node
        remove_edges_involving_node(adjList, i)

  # Traversing the adjacency list to build the edge list
  for i in range(len(adjList)):
      connections = adjList[i]
      for j in range(len(connections)):
          node = connections[j][0]
          latency = connections[j][1]
          # Adding non-duplicate edges
          if i < node:  # i because Python is 0-indexed
              edges.append([i, node, latency])

  #-------------------------------------------------------------------------------
  #-------------------------------------------------------------------------------
  #------------------------------  total elapsed time ----------------------------
  total_time = 0
  #-------------------------------------------------------------------------------
  #------------------------------ Total OF ---------------------------------------
  OF_total = 0
  #-------------------------------------------------------------------------------
  # -------------------- Define the parameters NSGA-III --------------------------

  # Reading parameters from file or using default values
  hyper_parameters = read_hyper_parameters(input_folder)

  # Updating parameters in code
  population_size = hyper_parameters.get('population_size', default_params['population_size'])    # initial population on which the algorithm will operate

  generations = hyper_parameters.get('generations', default_params['generations'])    # number of generations (iterations) - defines how many times the selection, crossover, mutation and evaluation process will be repeated (default=200).

  cx_prob = hyper_parameters.get('cx_prob', default_params['cx_prob'])    # defines how many times the selection, crossover, mutation and evaluation process will be repeated - If it is 0.30, it means that 30% of the selected parent pairs will be crossed.

  mut_prob = (1/numNodes) # frequency with which the mutation will be applied to the generated descendants - If it is (1/numNodes), it means that (1/numNodes)% of the descendants will undergo mutation (default: 1% to 5%).

  indpb = hyper_parameters.get('mut_prob', default_params['indpb']) # In uniform crossover, each gene from two parents has a probability (usually 0.9 - equals 90%) of being exchanged, resulting in offspring with genes randomly inherited from either parent (default: 70% to 90%).

  majority = round((numNodes / 2) + 1) # The criterion used was simple majority = half + 1

  # Define the Hall of Fame
  #hof = hall_of_fame = tools.HallOfFame(1)  # Stores the best individual
  # ------------------------------------------------------------------------------
  # --------------------------- Configure DEAP -----------------------------------
  creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, 1.0)) # -1.0 = Minimize, 1.0 = Maximize, 0.0 = Without a specific optimization direction (free)
  creator.create("Individual", list, fitness=creator.FitnessMulti)
  #-------------------------------------------------------------------------------
  # The bounded_crossover function performs the crossovers using cxUniform and cxOnePoint, and ensures that the genes of the individuals remain within the interval [0,1)
  def bounded_crossover(ind1, ind2, majority, indpb):
      """
      Applies crossover to the genes of two individuals.

      Parameters:
      ind1 (list of int): The first individual.
      ind2 (list of int): The second individual.
      indpb (float): Mutation probability for each gene.
      """

      cont = 0 # counts how many elements of the tuple are different
      for elem1, elem2 in zip(ind1, ind2):
          if (elem1 != elem2):
              cont += 1

      # Modifies only randomly generated individuals, with amount of (elements<=majority) with an individual from the population
      if (cont <= majority) and (cont > 0):
        # cxTwoPoint e cxOnePoint: Swap entire segments of genes, ensuring that values ​​remain within range
        #tools.cxOnePoint(ind1, ind2)
        # The cxUniform operator performs crossover by randomly swapping the genes of the parents. This keeps the gene values ​​within the range [0,1) because the swapped genes are just copied from the parents.
        # cxUniform: Keeps values ​​within 0 or 1 because it only exchanges the genes from the parents.
        tools.cxUniform(ind1, ind2, indpb)

      return ind1, ind2

  #-------------------------------------------------------------------------------

  def bounded_mutation(individual, majority, numNodes):
      """
      Applies mutation to the genes of an individual.

      Parameters:
      individual (list of int): The individual to be mutated.
      indpb (float): Probability of mutation for each gene.

      Returns:
      individual (list of int): The mutated individual.
      """
      cont = 0 # counts how many elements of the tuple are different
      modify = []  # generates an element from an individual and modifies it randomly
      # Compares the randomly generated individual with an individual from the population
      for i in range(numNodes):
          modify.append(random.randint(0, 1))

          if individual[i] != modify[i]:
              cont += 1
      # Modifies only randomly generated individuals, with qde of (elements<=majority) with an individual from the population
      # Up to 50% of the number of elements in a tuple that make up an individual in the population
      if (cont <= majority) and (cont > 0):
          for i in range(numNodes):
            # Apply the mutation to the individual
              individual[i] = modify[i]

      return individual, # Returning the mutated individual as a tuple (1 element)
  #-------------------------------------------------------------------------------

  #----------------------------------------------------------------------------------------------
  # Defining the attr_int function to generate int numbers: 0 or 1
  # Hybrid version (90% normal + 10% guaranteed)
  def attr_int():
      return 1 if random.random() < 0.1 else 0  # Increases the initial chance of activating nodes

  def individual_with_min_ones():
    ind = [0] * numNodes
    ones_indices = random.sample(range(numNodes), cut_sol)
    for idx in ones_indices:
        ind[idx] = 1
    return creator.Individual(ind)

  def hybrid_individual():
    if random.random() < 0.1:
        return individual_with_min_ones()
    else:
        return tools.initRepeat(creator.Individual, attr_int, numNodes)
  #----------------------------------------------------------------------------------------------

  toolbox = base.Toolbox()
  # ------------------------------------------------------------------------------
  # Registering the attr_int function in the toolbox, toolbox.attr_int() to generate numbers: 0 or 1
  #toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=numNodes)
  toolbox.register("individual", hybrid_individual)
  toolbox.register("population", tools.initRepeat, np.array, toolbox.individual)
  toolbox.register("mate", bounded_crossover, majority=majority, indpb=indpb)
  #-------------------------------------------------------------------------------
  # Each gene has a 50% chance of being mutated. If the gene is not chosen for mutation, it remains unchanged.
  toolbox.register("mutate", bounded_mutation, majority=majority, numNodes=numNodes)

  # 3 = Nº Objectives (R e B) and 30 = Number of divisions made in hyperspace to create the reference points
  toolbox.register("select", tools.selNSGA3, ref_points=tools.uniform_reference_points(3, 30))

  # Logbook and statistics configuration
  # ind: This is the parameter that represents the input argument to this anonymous function.
  # In the context of DEAP (a library of evolutionary algorithms), ind is likely to be an individual in the population.
  #stats = tools.Statistics(lambda ind: ind.fitness.values)
  #stats.register("avg", np.mean)
  #stats.register("std", np.std)
  #stats.register("min", np.min)
  #stats.register("max", np.max)

  #logbook = tools.Logbook()
  #logbook.header = ["gen", "nevals"] + stats.fields
  #-------------------------------------------------------------------------------
  # Stores last nº of feasible solutions
  v_all_sol_feasible = []
  # Stores all the best OF values ​​for each job
  v_all_OF = []
  # Stores all best v_nodes values ​​for each job
  v_all_nodes = []
  # Stores all best v_nodes values ​​for each job
  v_all_times = []
  #-------------------------------------------------------------------------------
  # List of lists containing the search space of Latency of all nodes in the layer
  N_L = np.array([])
  # Counts the number of empty sublists provided by NSGA3_Enhanced in a running
  num_empty_sublists_nsga3 = 0
  #------------------------ Start of execution of each job -----------------------
  for job in range(n_jobs):
      #---------------------------------------------------------------------------
      #------------------------- start_time, end_time time -----------------------
      start_time = 0
      end_time = 0
      # Start of time measurement
      # start_time  # Timing not directly translated, can use time.time() if needed
      start_time = time.time()
      source = jo[job]  # Stores the position value of the source node
      # Discounts the initial node connection time in the calculation (only 1 time for each job)
      l_job = jl[job] - t_c
      N_L = get_latencies(source, adjList, numNodes)
      #------------- Printing job attribute values ​​to the screen -----------------
      #print("\n---------------------------------------------------------")
      #print(f"\n Scenario: {numNodes} nds - {n_jobs} jobs")
      #print(f"\n Running: {r+1}/{numRunnings}")
      #print(f"\n Job {job}[{jr[job]}, {jb[job]}, {jl[job]}, {jo[job]}] waiting...")
      #print("\n---------------------------------------------------------\n")
      #---------------------------------------------------------------------------
      # ------------------------ Variables initialization ------------------------
      # Output value of the population generated by NSGA3
      population = []
      # Output value of the population generated by NSGA3, without replication in the solutions
      final_population_with_replicate = []
      # Get unique rows based on unique values' indexes
      final_population = []
      #---------------------------------------------------------------------------
      # Initializing population
      population = toolbox.population(n=population_size)
      population = [toolbox.individual() for _ in range(population_size)]
      #---------------------------------------------------------------------------
      #-------------------- Nodes allocated to each job --------------------------
      # Separately stores the indexes of the nodes that make up the solution for each job
      v_nodes = []
      # Stores the number of nodes allocated in the network, in the best solution, for the execution of a given job
      size_v_nodes = 0
      # --------------------------------------------------------------------------
      # Stores the position of the minimum solution (best solution)
      better_pos = 0
      # Nº of feasible solutions final_population_with_replicate (used in graphics)
      #num_solutions = 0
      # Nº of feasible solutions final_population (sem replicação)
      n_sol_feasible = 0
      # Stores the individual with the smallest value of OF
      best_sol_ind = []
      # Stores the value of the best solution
      best_sol = 0
      # Index of vectors NSS_R and NSS_B that stores the position of the minimum solution
      # List of solution values (used in graphics)
      #solutions_val = []
      # Lists used to store the values ​​of objective functions and their positions
      v_sol_f1 = []  # f1 (Resources)
      v_sol_f2 = []  # f2 (Bandwidth)
      v_sol_f3 = []  # f3 (Latency)
      v_sol_fx = []  # Stores the value of the objective function: fx=f1^2+f2^2-f3
      v_sol_ind = []  # Stores the individuals of feasible solutions (binary tuples)
      #---------------------------------------------------------------------------
      #------------------------------------------------------ Register the evaluate function and call the partial function with the parameters of a given job ------------------------------------------------------------------------------
      # Pass the objectiveFunction function directly to toolbox.register, and use partial to pre-configure the function parameters.
      # Registering the objective function in the toolbox
      toolbox.register("evaluate", partial(objectiveFunction, l_job=l_job, job=job, jr=jr, jb=jb, N_R=N_R, N_B=N_B, N_L=N_L, v_sol_f1=v_sol_f1, v_sol_f2=v_sol_f2, v_sol_f3=v_sol_f3, v_sol_fx=v_sol_fx, v_sol_ind=v_sol_ind, numNodes=numNodes))
      # Decorating the evaluation function, using the lambda function that checks whether an individual (ind) satisfies the constraints.
      # The penalty function to return a fitness value (INF) for any constraint violation, ensuring that those individuals will not be selected
      toolbox.decorate("evaluate", tools.DeltaPenality(lambda ind: jobConstraints(ind, l_job, job, jr, jb, N_R, N_B, N_L, numNodes, Min_FX), INF))
      #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      # ------------------------------------------------- NSGA-III function call ----------------------------------------------------------------------------------------------------------------------
      #----------------------------------- Register the evaluate function and call the partial function with the parameters of a given job ------------------------------------------------------------
      # evaluate - calculates the fitness of an individual or a solution.
      # It is making use of statistics (stats=stats) or (stats=None)
      #logbook =
      algorithms.eaMuPlusLambda(population, toolbox, mu=population_size, lambda_=population_size, cxpb=cx_prob, mutpb=mut_prob, ngen=generations, stats=None, halloffame=None, verbose=False)
      #-----------------------------------------------------------------------------------------------------------------------------------
      # Obtaining the values ​​of the objective functions (R, B and L)
      #print("Logbook History:")
      #for record in logbook:
      #   print(record)
      #---------------------------------------------------------------------------
      # Creating final_population_with_replicate for the current run
      # .tolist() in Python is used to convert a NumPy array (or similar structures) into a Python list
      final_population_with_replicate = np.array(list(zip(v_sol_fx, v_sol_ind, v_sol_f1, v_sol_f2, v_sol_f3)), dtype=object).tolist()
      #---------------------------------------------------------------------------
      # Accessing values ​​correctly
      values = [t[0] for t in final_population_with_replicate]
      individuals = [t[1] for t in final_population_with_replicate]

      # Convert individuals to a tuple for processing
      # It may happen that different tuples produce the same value in the OF
      #individuals_as_tuples = [tuple(individual) for individual in individuals]

      # Finding unique indexes using a set (individuals variable), sorted by values
      seen = set()
      unique_indexes = []
      for i, indiv in enumerate(values):
      #for i, indiv in enumerate(individuals_as_tuples):
          if indiv not in seen:
              seen.add(indiv)
              unique_indexes.append(i)

      # Creating a new list without duplicates
      final_population = np.array([(values[i], individuals[i]) for i in unique_indexes], dtype=object)

      # Sorted final_population by v_sol_fx (first field)
      final_population = sorted(final_population, key=lambda x: x[0])
      final_population = np.array(final_population)  # Garante que é um array NumPy
      size_final_population = len(final_population)

      if size_final_population > 0:
        # The best individual is initially the first
        best_sol_ind = final_population[0]
        best_sol = best_sol_ind[0]

        # Locates the position of the best individual
        better_pos = np.where(final_population[:, 0] == best_sol_ind[0])[0][0]
        individuals = final_population[better_pos, 1]

        job_R = jr[job]
        job_B = jb[job]
        cont = 0
        t_sol = False

        while (job_R > 0 or job_B > 0) and (cont < size_final_population):
            num_ones_individual = sum(1 for i in individuals if i == 1)
            cont_nodes_not_busy = sum(1 for k in range(numNodes) if individuals[k] == 1 and V_Busy[k] == 0)
            # The condition restricts the solutions to at most cut_sol node per job
            if (num_ones_individual <= cut_sol) and (num_ones_individual <= cont_nodes_not_busy):
                v_nodes.clear()
                for j in range(numNodes):
                    if individuals[j] == 1 and V_Busy[j] == 0:
                        v_nodes.append(j)

                size_v_nodes = len(v_nodes)
                S_R = sum(N_R[pos] for pos in v_nodes)
                S_B = sum(N_B[pos] for pos in v_nodes)

                if (size_v_nodes > 0) and (S_R >= jr[job]) and (S_B >= jb[job]):
                    for pos_element in v_nodes:
                        if N_R[pos_element] >= job_R:
                            N_R[pos_element] -= job_R
                            job_R = 0
                        else:
                            job_R -= N_R[pos_element]
                            N_R[pos_element] = 0

                        if N_B[pos_element] >= job_B:
                            N_B[pos_element] -= job_B
                            job_B = 0
                        else:
                            job_B -= N_B[pos_element]
                            N_B[pos_element] = 0

                    if job_R == 0 and job_B == 0:
                        t_sol = True
                        break

            # If you haven't found a solution yet
            t_sol = False
            v_nodes.clear()
            cont += 1

            if cont < size_final_population:
                best_sol_ind = final_population[cont]
                best_sol = best_sol_ind[0]
                better_pos = np.where(final_population[:, 0] == best_sol_ind[0])[0][0]
                individuals = final_population[better_pos, 1]
                job_R = jr[job]
                job_B = jb[job]
            else:
                best_sol = 0
                break

        # ------------------------- Post processing after while ------------------------- #
        if t_sol:
            size_v_nodes = len(v_nodes)
            for i in range(size_v_nodes):
                v_nodes_network.append(v_nodes[i])
                cont_nos_rede += 1

            V_Busy = mark_busy(cont_nos_rede, v_nodes_network, V_Busy)
            n_sol_feasible = len(final_population)
            best_sol_population_job.append(best_sol_ind)
            v_all_sol_feasible.append(n_sol_feasible)
            v_all_OF.append(best_sol)
            v_all_nodes.append(v_nodes.copy())
            v_nodes.clear()
            OF_total += best_sol
        else: # t_sol:
            n_sol_feasible = 0
            best_sol = 0
            v_all_sol_feasible.append(n_sol_feasible)
            v_all_OF.append(best_sol)
            v_all_nodes.append([])
            v_nodes.clear()
      else: # if size_final_population > 0:
        n_sol_feasible = 0
        best_sol = 0
        v_all_sol_feasible.append(n_sol_feasible)
        v_all_OF.append(best_sol)
        v_all_nodes.append([])
        v_nodes.clear()

      # Total time per job
      end_time = time.time() - start_time
      v_all_times.append(end_time)
      total_time += end_time

  #------------------- End for job in range(n_jobs): ---------------------------------------------------------------------
  # ========== AUpdates the best solution on first running ========== #
  if r == 0:
    Min_FX = OF_total
    num_empty_sublists_nsga3_better = sum(1 for sublist in v_all_nodes if len(sublist) == 0)

    v_all_sol_feasible_better = v_all_sol_feasible.copy()
    v_all_OF_better = v_all_OF.copy()
    v_all_nodes_better = [list(nodes) for nodes in v_all_nodes]
    v_all_times_better = v_all_times.copy()

  # ========== Assessment of conditions for replacement ========== #
  cond1 = (Min_FX >= OF_total)
  num_empty_sublists_nsga3 = sum(1 for sublist in v_all_nodes if len(sublist) == 0)
  cond2 = (bool(v_all_nodes) and (num_empty_sublists_nsga3_better >= num_empty_sublists_nsga3))

# ========== Selecting the solution to be printed ========== #
  if cond1 and cond2:
    # Update the best solution found
    Min_FX = OF_total
    num_empty_sublists_nsga3_better = num_empty_sublists_nsga3

    # Copies current solutions as the best
    # v_all_sol_feasible_better = v_all_sol_feasible.copy()
    v_all_OF_better = v_all_OF.copy()
    v_all_nodes_better = [list(nodes) for nodes in v_all_nodes]
    v_all_times_better = v_all_times.copy()

  # ========== Printing to output (always the "best" solution so far) ========== #
  output_content = f"\nInput file: {input_file_name}\n\n"
  output_content += f"Number of jobs: {n_jobs}\n\n"
  output_content += " Job   [ Jr,  Jb,  Jl,  Jo]                 OF           Allocated nodes\n"

  for job in range(n_jobs):
    job_details = f"[{jr[job]}, {jb[job]}, {jl[job]}, {jo[job]}]"
    of_value = f"{v_all_OF_better[job]:,.1f}"
    allocated_nodes = v_all_nodes_better[job]
    output_content += f" {job:^5} {job_details:<27} {of_value:>14} {str(allocated_nodes):>20}\n"

  # ========== Final updates ========== #
  OF_total = np.sum(v_all_OF_better)
  OF_execs.append(float(OF_total))
  v_all_nodes_execs[r] = [list(nodes) for nodes in v_all_nodes_better]
  total_time = np.sum(v_all_times_better)
  #-------- End: if (cond1 and cond2): ------------------------------------------------------------------------------------------
  #---------------------------------------------------------------------------------------------------------------------------------------------------------------
  #print('-----------------------------------------------------------------------------------------')
  #print(f'The results are in the file: {name_file}')
  #print('-----------------------------------------------------------------------------------------')
  #-------------------------------------------------------------------------------------------------
  # Print the Job, OF and, allocated node
  fileID.write(output_content)
  # OF_total is a NumPy array with a single value, it needs to be converted to a scalar before formatting
  fileID.write(f"\nTotal OF: {OF_total:,.1f}")
  fileID.write(f"\n\nRuntime: {total_time:,.4f} seconds")
  #----------------------------------------------------------------------------------------------------------
  #--------- Adds total_time to the times_execs variable (contains the times of all executions) -------------
  times_execs.append(total_time)
  #----------------------------------------------------------------------------------------------------------
  # Prints to the file, only in the last execution, all the times (s) of the OF values ​​obtained
  if (r==(numRunnings-1)):
    fileID.write("\n--------------------------------------------------------------------------------------\n")
    fileID.write("\n--------------- Solutions generated by NSGA-III Enhanced -----------------------------")
    fileID.write("\n----------------------------------- OF Execs -----------------------------------------")
    fileID.write("\n" + "\n".join([f" {i:4d}: {OF_execs[i]:,.1f}" for i in range(numRunnings)]))
    fileID.write("\n--------------------------------------------------------------------------------------")
    fileID.write("\n------------------------------ v_all_nodes_execs -------------------------------------")
    fileID.write("\n" + "\n".join([f" {i:4d}: {list(map(list, v_all_nodes_execs[i]))}" for i in range(numRunnings)]))
    fileID.write("\n--------------------------------------------------------------------------------------")
    fileID.write("\n-------------------------------- Times Execs -----------------------------------------")
    fileID.write("\n" + "\n".join([f" {i:4d}: {times_execs[i]:,.4f}" for i in range(numRunnings)]))
    fileID.write("\n--------------------------------------------------------------------------------------")
    mean = np.mean(times_execs)
    if (numRunnings==1):
         sd = np.std(times_execs, ddof=0)  # For population (n), ddof=0 (Delta Degrees of Freedom)
    else:
         sd = np.std(times_execs, ddof=1)  # For population (n-1), ddof=1 (Delta Degrees of Freedom)

    fileID.write(f"\n mean:  {mean:,.4f}")
    fileID.write(f"\n   sd:  {sd:,.4f}")
    fileID.write("\n--------------------------------------------------------------------------------------\n")
  #------------------------------------------------------------------------------------------------------------
  # Close the file: results.txt
  fileID.close()
#----------------------------------------------------------------------------------------------------------------------------------------
# Start of execution (main):

# Setting default values (global)
#default_cut_sol = 2
#default_runnings = 1

# Set the directory path where the .json files are located
# Setting default values
#default_in_f = r'C:\Users\Murilo\Documents\DOUTORADO\UFMS-FACOM\Material-Ricardo\Encontros-Ricardo\Encontro_55-08-07-2025\25nds\input'
#default_out_f = r'C:\Users\Murilo\Documents\DOUTORADO\UFMS-FACOM\Material-Ricardo\Encontros-Ricardo\Encontro_55-08-07-2025\25nds\output'

# Prompts user for directory path or uses default value if ENTER is pressed
#input_folder = input(f"Enter an input directory path or press ENTER for default (ex: {default_in_f}): ") or default_in_f

#output_folder = input(f"Enter an output directory path or press ENTER for default (ex: {default_out_f}): ") or default_out_f


# input_folder = '/home/jonatas/high-performance-execution/ExperimentSix/input/light/'

# output_folder = '/home/jonatas/high-performance-execution/ExperimentSix/metaheuristic/enhanced/light/'

input_folder, output_folder = cl.get_target_dirs(argv)

# Collecting the maximum number of nodes considered per solution of each Job
#cut_sol = get_positive_integer("Value of the cutting radius of the maximum number of nodes considered per solution of each Job: ", default_cut_sol)

cut_sol = 2

# Number of runnings of a given configuration
#numRunnings = get_positive_integer(f"Number of runnings: ", default_runnings)

numRunnings = 5


# Create the output folder if it doesn't exist; keep if it already exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# List all .json files in input directory
json_files = [f for f in os.listdir(input_folder) if f.endswith('.json')]

# Process each .json file and generate a .txt result file
for json_file in json_files:
    # Full path of input file
    input_path = os.path.join(input_folder, json_file)
    main(input_path, output_folder, cut_sol, numRunnings)

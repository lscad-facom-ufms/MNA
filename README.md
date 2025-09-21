# MNA Repository

Repository of the Multi-Node Allocation heuristics, metaheuristics, and input data.  
All the source code files were written in the Python language (python3). The input instance data are in the `.json` and `.parquet` formats.  
The current version of the source code is **1.0.0 (September 2025)**.

## This repository comprises:
- The source-code file of the **DI-MNA heuristic** (`DI-MNA` folder);
- The source-code files of the **NSGA3-Hyb** and **NSGA3-Enh metaheuristics** (`NSGA-III` folder);
- The source-code file of the **Branch-and-Bound algorithm** (`Branch_and_Bound` folder);
- The source-code file of the **NGenerator tool**, which allows the generation of new IoT network inputs (`NGenerator` folder);
- The source-code file of the **NViewer tool**, which allows the graphical view of the IoT network inputs (`NViewer` folder);
- **Input files** for DI-MNA, the metaheuristics, and the B&B algorithm (`inputs` folder).

---

## Getting Started and Execution

Follow the steps below to download the repository and run the algorithms.

### 1. Cloning the Repository

First, clone this repository to your local machine using the following command:

```bash
git clone https://github.com/lscad-facom-ufms/MNA.git
cd MNA-Repository
```

### 2. Downloading Input Files (Git LFS)

The input files in the `inputs` folder are large and are managed using **Git Large File Storage (LFS)**. After cloning, you need to pull these files.

If you don't have Git LFS installed, you must install it first. You can find instructions at [git-lfs.github.com](https://git-lfs.github.com).

After installation, run the following command from the root of the repository to download the actual data files:

```bash
git lfs pull
```

### 3. Running the Scripts

All algorithms and tools were developed in **Python 3**. Each component listed above has its own execution script, generally located inside its respective folder.

---

## Command-Line Arguments

The execution of the main algorithms (**DI-MNA** and **NSGA-III**) requires two command-line arguments to specify the input and output directories.

- `--in, --input <input_dir>`: (Required) Sets the directory where the input files are located.
- `--out, --output <output_dir>`: (Required) Sets the directory where the results will be saved.
- `-h, --help`: Displays help information and exits.

---

## Command Structure

To run an algorithm, use the following structure:

```bash
python3 <path_to_script>.py --in <input_directory> --out <output_directory>
```

---

## Examples

Here are some examples of how to run the scripts from the repository's root directory:

### Example: Running the DI-MNA heuristic
```bash
python3 DI-MNA/MNA-IoT.py --in ./inputs/instance_set_1 --out ./results/di_mna_results
```

### Example: Running the NSGA3-Enh metaheuristic
```bash
python3 NSGA-III/NSGA3-IoT-Enhanced.py --in ./inputs/instance_set_2 --out ./results/nsga3_enh_results
```

---

The other heuristics scripts, such as **Branch-and-Bound** and **NSGA3-Hyb**  follow a similar execution pattern. Please refer to their respective folders for the specific main script file.


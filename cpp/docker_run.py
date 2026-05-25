import subprocess
import os

base_in = "/app/inputs/parquet/"
base_out = "/app/output/"

executable_path = "/app/out/build/docker-sycl/dimna_sycl/di_mna_sycl" 

inputs = ["10nds", "100nds", "1_000nds", "10_000nds"]
runs = "11"
alfas = ["2"]
betas = ["500"]

for input_name in inputs:
    for a in alfas:
        for b in betas:

            print(input_name, "cs:", a, "ccn:", b, "...")

            out_path = base_out + input_name + "/cs" + a + "/"
            if not os.path.exists(out_path):
                os.makedirs(out_path)

            subprocess.run([executable_path,
                            "--in", base_in + input_name,
                            "--out", out_path,
                            "--runs", runs,
                            "--cs", a,
                            "--ccn", b])

    print(input_name, "done.")

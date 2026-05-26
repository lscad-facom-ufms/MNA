import subprocess
import os

# base_in = "../inputs/parquet/"
# base_out = "./output/"
 
base_in = "/app/inputs/parquet/"
base_out = "/app/output/full_explore/"

executable_path = "/app/out/build/docker-sycl/dimna/di_mna" 
inputs = ["10_nds", "100nds", "1_000nds", "10_000nds"]

runs = "1"

alfas = ["2"]
betas = ["50000"]

for input in inputs:
    for a in alfas:
        for b in betas:

            print(input, "cs:", a, "ccn:", b, "...")

            out_path = base_out + input + "/cs" + a + "/"
            if not os.path.exists(out_path):
                os.makedirs(out_path)

            subprocess.run([executable_path,
                            "--in", base_in + input,
                            "--out", out_path,
                            "--runs", runs,
                            "--cs", a,
                            "--ccn", b])

    print(input, "done.")

import subprocess
import os

base_in = "../inputs/parquet/"
base_out = "./output/"


inputs = ["10nds", "100nds", "1_000nds"]

runs = "1"

alfas = ["1", "2", "3"]
betas = ["500", "1000", "5000"]

for input in inputs[2:3]:
    for a in alfas:
        for b in betas:

            print(input, "cs:", a, "ccn:", b, "...")

            out_path = base_out + input + "/" + a + "/"
            if not os.path.exists(out_path):
                os.makedirs(out_path)

            subprocess.run(["./build/canon_dimna/canon_di_mna",
                            "--in", base_in + input,
                            "--out", out_path,
                            "--runs", runs,
                            "--cs", a,
                            "--ccn", b])

    print(input, "done.")

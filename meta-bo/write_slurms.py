#!/usr/bin/env python3
"""Generate SLURM scripts by replacing placeholders with values from CSV file."""

import pandas as pd
import os
import shutil

# Read the CSV file
df = pd.read_csv("next_candidates.csv")

# Read the template SLURM script
with open("metabo.slurm", "r") as f:
    template = f.read()

# Remove and recreate the cands directory
if os.path.exists("cands"):
    shutil.rmtree("cands")
os.makedirs("cands")

# Generate SLURM scripts for each candidate
for i, row in df.iterrows():
    # Replace placeholders with actual values
    script = template.replace("<GPRPTEAPOT>", str(row["gpr_p"]))
    script = script.replace("<GPRVALTEAPOT>", str(row["gpr_val_factor"]))
    script = script.replace("<ONORMTEAPOT>", str(row["onorm_factor"]))

    # Save to file
    filename = f"cands/cand{i}.slurm"
    with open(filename, 'w') as f:
        f.write(script)

    print(f"Generated {filename}")

print(f"Created {len(df)} SLURM scripts in the 'cands' directory")

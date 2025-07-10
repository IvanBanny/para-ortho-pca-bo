#!/usr/bin/env python3
"""Generate SLURM scripts by replacing placeholders with provided values."""

import os
import shutil
import argparse


OPCABO_DEFAULT_CONFIG = {"gpr_p": 0.42, "gpr_val_factor": 0.0, "onorm_factor": 5.812329}
OPCABO_BATCH_CONFIG = {
    1: {"gpr_p": 0.42, "gpr_val_factor": 0.0, "onorm_factor": 5.812329},
    5: {"gpr_p": 0.520197, "gpr_val_factor": 0.027186, "onorm_factor": 7.952228},
    10: {"gpr_p": 0.456144, "gpr_val_factor": 0.0, "onorm_factor": 6.875751},
    20: {"gpr_p": 0.471504, "gpr_val_factor": 0.0, "onorm_factor": 7.803236},
    42: {"gpr_p": 0.740087, "gpr_val_factor": 0.070968, "onorm_factor": 7.55555},
}


def parse_arguments():
    """Parse command line arguments for experiment configuration."""
    parser = argparse.ArgumentParser(
        description="Generate SLURM scripts for the main experiments."
    )

    parser.add_argument(
        "--algorithms",
        type=str,
        nargs="+",
        default=["vanilla", "pca", "opca"],
        help="src to test (default: vanilla pca opca)"
    )

    parser.add_argument(
        "--batch",
        type=int,
        nargs="+",
        default=[1, 5, 10, 20, 42],
        help="Batch sizes to test (default: 1 5 10 20 42)"
    )

    return parser.parse_args()


def main():
    """Generate SLURM scripts by replacing placeholders with provided values."""
    args = parse_arguments()

    print("\nBayesian Optimization Experiment Configuration:")
    print(f"  algorithms: {args.algorithms}")
    print(f"  Batch sizes: {args.batch}\n")

    # Read the template SLURM script
    with open("exp.slurm", "r") as f:
        template = f.read()

    # Remove and recreate the cands directory
    if os.path.exists("slurms"):
        shutil.rmtree("slurms")
    os.makedirs("slurms")

    # Generate SLURM scripts for each candidate
    for algo in args.algorithms:
        for batch in args.batch:
            # Replace placeholders with actual values
            script = template.replace("<ATEAPOT>", str(algo))
            script = script.replace("<BTEAPOT>", str(batch))

            opcapo_config = OPCABO_DEFAULT_CONFIG
            if batch in OPCABO_BATCH_CONFIG.keys():
                opcapo_config = OPCABO_BATCH_CONFIG[batch]

            script = script.replace("<GPRPTEAPOT>", str(opcapo_config["gpr_p"]))
            script = script.replace("<GPRVALTEAPOT>", str(opcapo_config["gpr_val_factor"]))
            script = script.replace("<ONORMTEAPOT>", str(opcapo_config["onorm_factor"]))

            # Save to file
            filename = f"slurms/{algo}-b{batch}.slurm"
            with open(filename, 'w') as f:
                f.write(script)

            print(f"Generated {filename}")

    print(f"\nCreated {len(args.algorithms) * len(args.batch)} SLURM scripts in the 'slurms' directory\n")


if __name__ == '__main__':
    main()

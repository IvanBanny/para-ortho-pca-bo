#!/usr/bin/env python3
import os
import shutil
from pathlib import Path


def count_lines_in_file(file_path):
    """Count lines in a file, handling potential encoding issues."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)
    except UnicodeDecodeError:
        # Try with different encoding if UTF-8 fails
        with open(file_path, 'r', encoding='latin-1') as f:
            return sum(1 for _ in f)


def main():
    meta_bo_path = Path("../meta-bo-data")

    if not meta_bo_path.exists():
        print(f"Directory '{meta_bo_path}' does not exist")
        return

    valid_line_counts = {201, 351, 651}
    deleted_count = 0
    kept_count = 0
    error_count = 0

    print(f"Scanning directories in {meta_bo_path}...")

    # Iterate through all subdirectories in meta-bo
    for subdir in meta_bo_path.iterdir():
        if not subdir.is_dir():
            continue

        try:
            # Find the nested directory (should be only one)
            nested_dirs = [d for d in subdir.iterdir() if d.is_dir()]

            if len(nested_dirs) != 1:
                print(f"Warning: {subdir} doesn't have exactly one subdirectory, skipping")
                error_count += 1
                continue

            nested_dir = nested_dirs[0]

            # Find .dat files in the nested directory
            dat_files = list(nested_dir.glob("*.dat"))

            if len(dat_files) != 1:
                print(f"Warning: {nested_dir} doesn't have exactly one .dat file, skipping")
                error_count += 1
                continue

            dat_file = dat_files[0]
            line_count = count_lines_in_file(dat_file)

            # Check if line count matches valid counts
            if line_count not in valid_line_counts:
                print(f"Deleting {subdir} (dat file has {line_count} lines)")
                shutil.rmtree(subdir)
                deleted_count += 1
            else:
                print(f"Keeping {subdir} (dat file has {line_count} lines)")
                kept_count += 1

        except Exception as e:
            print(f"Error processing {subdir}: {e}")
            error_count += 1

    print(f"\nSummary:")
    print(f"Directories deleted: {deleted_count}")
    print(f"Directories kept: {kept_count}")
    print(f"Errors encountered: {error_count}")


if __name__ == "__main__":
    # Safety confirmation
    response = input("This will permanently delete directories. Are you sure? (yes/no): ")
    if response.lower() == 'yes':
        main()
    else:
        print("Operation cancelled.")

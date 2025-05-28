import os
import json
import re


def modify_json_files(base_dir="experiment"):
    """
    Traverse all subdirectories in the experiment folder and modify JSON files
    to replace specific patterns in the experiment_attributes.
    """
    if not os.path.exists(base_dir):
        print(f"Directory '{base_dir}' not found.")
        return

    modified_count = 0

    # Get all subdirectories
    subdirs = [d for d in os.listdir(base_dir)
               if os.path.isdir(os.path.join(base_dir, d))]

    print(f"Found {len(subdirs)} subdirectories to process...")

    for subdir in subdirs:
        subdir_path = os.path.join(base_dir, subdir)

        # Find JSON files in this subdirectory
        json_files = [f for f in os.listdir(subdir_path)
                      if f.endswith('.json')]

        if not json_files:
            print(f"No JSON file found in {subdir}")
            continue

        if len(json_files) > 1:
            print(f"Multiple JSON files found in {subdir}, processing first one: {json_files[0]}")

        json_file_path = os.path.join(subdir_path, json_files[0])

        try:
            # Read the JSON file
            with open(json_file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Perform the replacements
            original_content = content
            content = content.replace('{"self.doe_params": "{', '{"doe_params": "dict{')
            content = content.replace('{"torch_config": "{', '{"torch_config": "dict{')

            # Check if any changes were made
            if content != original_content:
                # Write back the modified content
                with open(json_file_path, 'w', encoding='utf-8') as f:
                    f.write(content)

                modified_count += 1
                print(f"Modified: {json_file_path}")
            else:
                print(f"No changes needed: {json_file_path}")

        except Exception as e:
            print(f"Error processing {json_file_path}: {str(e)}")

    print(f"\nCompleted! Modified {modified_count} files out of {len(subdirs)} subdirectories.")


if __name__ == "__main__":
    modify_json_files()

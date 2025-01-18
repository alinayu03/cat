import os
import subprocess
import glob
from my_faiss_functions import (
    process_code_file,
    run_funct,
    initialize_faiss,
)

def index_existing_codebase(root_dir="."):
    py_files = glob.glob(f"{root_dir}/**/*.py", recursive=True)
    for py_file in py_files:
        process_code_file(py_file)

def get_changed_files():
    cmd = ["git", "diff", "--name-only", "HEAD^", "HEAD"]
    output = subprocess.check_output(cmd).decode("utf-8").strip()
    changed_files = [line for line in output.split("\n") if line.endswith(".py")]
    return changed_files

def analyze_changed_files(changed_files):
    results = []
    for cf in changed_files:
        with open(cf, "r", encoding="utf-8") as f:
            new_code_input = f.read()
        # Use run_funct (which uses your analyze_* methods)
        result = run_funct(main_code_file="all_code.txt", new_code_input=new_code_input)
        results.append((cf, result))
    return results

def main():
    # 1. Initialize the FAISS index
    initialize_faiss()

    # 2. Index all existing code
    index_existing_codebase(root_dir=".")

    # 3. Identify changed files
    changed_files = get_changed_files()
    if not changed_files:
        print("No .py files changed in this commit.")
        return

    # 4. Run the similarity checks on each changed file
    all_results = analyze_changed_files(changed_files)

    # 5. Decide what to do with the results (fail the build, print warnings, etc.)
    for filename, result in all_results:
        print(f"\n=== Redundancy Check for {filename} ===")
        print(result)

        # Example condition: if we have "redundant_code": True or similarity above 0.7, fail
        if isinstance(result, dict):
            # If it's a class result
            if result.get("redundant_code", False):
                raise SystemExit(
                    f"ERROR: {filename} is highly similar to existing code (score={result.get('class_similarity_score')})."
                )
            # If it's a method result
            if result.get("redundant_code", False):
                raise SystemExit(
                    f"ERROR: {filename} method is highly similar to existing code (score={result.get('highest_similarity')})."
                )

if __name__ == "__main__":
    main()

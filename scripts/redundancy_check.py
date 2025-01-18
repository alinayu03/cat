import os
import subprocess
import ast
import json
import torch
import faiss
import numpy as np
from transformers import AutoModel, AutoTokenizer

# Set up embedding model
CHECKPOINT = "Salesforce/codet5p-110m-embedding"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)
model = AutoModel.from_pretrained(CHECKPOINT, trust_remote_code=True).to(DEVICE)

D = 256  # Embedding dimension
index_methods = None
metadata_methods = []

def initialize_faiss():
    """Initializes FAISS index for method-level similarity search."""
    global index_methods
    index_methods = faiss.IndexFlatIP(D)  # Dot product similarity


def embed_text(text):
    """Generates embeddings from text using CodeT5."""
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        embedding = model(**tokens).last_hidden_state.mean(dim=1)  # Mean pooling
    return embedding.cpu().numpy()


def extract_methods_from_code(code):
    """Extracts methods and their code from Python source code."""
    tree = ast.parse(code)
    method_chunks = {}

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            method_name = f"def {node.name}"
            method_chunks[method_name] = ast.unparse(node)

    return method_chunks


def process_codebase(base_dir):
    """Processes the entire codebase and builds the FAISS index."""
    global index_methods, metadata_methods
    initialize_faiss()

    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                with open(filepath, "r", encoding="utf-8") as f:
                    code = f.read()
                methods = extract_methods_from_code(code)

                for method_name, method_code in methods.items():
                    embedding = embed_text(method_code)
                    index_methods.add(embedding)
                    metadata_methods.append({
                        "chunk_name": method_name,
                        "chunk_code": method_code,
                        "file_path": filepath
                    })


def get_changed_files():
    """Uses git to determine which files have been changed in the current push."""
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD~1"],
            stdout=subprocess.PIPE,
            text=True,
            check=True
        )
        changed_files = result.stdout.strip().split("\n")
    except subprocess.CalledProcessError:
        print("⚠️ Falling back to listing all files (e.g., first commit).")
        result = subprocess.run(
            ["git", "ls-files"],
            stdout=subprocess.PIPE,
            text=True,
            check=True
        )
        changed_files = result.stdout.strip().split("\n")

    return [f for f in changed_files if f.endswith(".py")]


def analyze_changes(changed_files):
    """Analyzes the changes and compares them with the existing codebase."""
    results = []

    for file in changed_files:
        with open(file, "r", encoding="utf-8") as f:
            code = f.read()

        methods = extract_methods_from_code(code)

        for method_name, method_code in methods.items():
            embedding = embed_text(method_code)
            distances, indices = index_methods.search(embedding, 5)

            method_results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx < len(metadata_methods):
                    similar_method = metadata_methods[idx]
                    method_results.append({
                        "similar_method_name": similar_method["chunk_name"],
                        "similar_method_code": similar_method["chunk_code"],
                        "similar_method_file": similar_method["file_path"],
                        "similarity_score": round(float(distance), 4)
                    })

            results.append({
                "method_name": method_name,
                "similarity_results": method_results
            })

    return results


if __name__ == "__main__":
    base_dir = os.getenv("GITHUB_WORKSPACE", ".")  # Root of the repository
    print("📁 Processing the entire codebase...")
    process_codebase(base_dir)

    print("📋 Identifying changed files...")
    changed_files = get_changed_files()
    if not changed_files:
        print("✅ No Python files changed in this push.")
        exit(0)

    print(f"🔄 Analyzing changes in {len(changed_files)} file(s): {changed_files}")
    results = analyze_changes(changed_files)

    print("📊 Redundancy analysis results:")
    print(json.dumps(results, indent=4))

    # Save results to a file
    results_file = os.path.join(base_dir, "redundancy_results.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    # Exit with a failure code if redundancy is detected
    has_redundancy = any(
        result["similarity_results"] and
        any(r["similarity_score"] >= 0.7 for r in result["similarity_results"])
        for result in results
    )

    if has_redundancy:
        print("❌ Redundancy detected. Check redundancy_results.json for details.")
        exit(1)
    else:
        print("✅ No significant redundancy detected.")
        exit(0)

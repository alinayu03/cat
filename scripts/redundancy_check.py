#!/usr/bin/env python3

import os
import subprocess
import glob
import ast
import random
import time
import math
import numpy as np
import torch
import faiss
from colorama import Fore
from transformers import AutoModel, AutoTokenizer

############################################
#  CONFIG & GLOBALS
############################################

CHECKPOINT = "Salesforce/codet5p-110m-embedding"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize CodeT5
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)
model = AutoModel.from_pretrained(CHECKPOINT, trust_remote_code=True).to(DEVICE)

CONTEXT_WINDOW = 512
D = 256  # Dimension of the embeddings

# FAISS index objects
index_methods = None
metadata_methods = []
metadata_classes = []

############################################
#  FAISS Initialization
############################################

def initialize_faiss():
    """
    Initializes a new FAISS index for storing embeddings (dot-product / inner-product).
    """
    global index_methods
    index_methods = faiss.IndexFlatIP(D)  # dot-product
    # Reset metadata in case it was used before
    metadata_methods.clear()
    metadata_classes.clear()


############################################
#  Embedding & Utility Functions
############################################

def embed_text(text: str) -> np.ndarray:
    """
    Generate a single [1 x D] embedding from a text string using CodeT5.
    """
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=CONTEXT_WINDOW).to(DEVICE)
    with torch.no_grad():
        # model(...) returns (last_hidden_state, ...)
        # but some CodeT5 variants might differ. You may need to adjust
        # if your embedding approach is different. For now, we take [0] for demonstration.
        embedding = model(**tokens)[0]

    # Convert to numpy, shape [1, D]
    return embedding[:, 0, :].cpu().numpy()


def compute_pairwise_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Computes cosine similarity between two embeddings (shape [1, D] each).
    """
    e1 = embedding1.flatten()
    e2 = embedding2.flatten()
    num = np.dot(e1, e2)
    den = (np.linalg.norm(e1) * np.linalg.norm(e2)) + 1e-12
    return float(num / den)


############################################
#  Code Parsing & Indexing
############################################

def extract_methods_and_classes(code: str):
    """
    Parses Python code with `ast` to extract top-level classes and methods.
    
    Returns:
        - method_chunks: dict of {method_name: method_code_str}
        - class_chunks: dict of {class_name: [list of method_code_str]}
    """
    tree = ast.parse(code)
    method_chunks = {}
    class_chunks = {}

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            class_name = node.name
            class_chunks[class_name] = []
            for child in node.body:
                if isinstance(child, ast.FunctionDef):
                    method_code = ast.unparse(child)
                    method_name = f"def {child.name}"
                    method_chunks[method_name] = method_code
                    class_chunks[class_name].append(method_code)

        elif isinstance(node, ast.FunctionDef):
            method_name = f"def {node.name}"
            method_chunks[method_name] = ast.unparse(node)

    return method_chunks, class_chunks


def process_code_file(filepath: str):
    """
    Reads a file and indexes its methods/classes into the FAISS index.
    """
    global index_methods, metadata_methods, metadata_classes

    if index_methods is None:
        initialize_faiss()  # safeguard, if not initialized

    with open(filepath, "r", encoding="utf-8") as f:
        code = f.read()

    extracted_methods, extracted_classes = extract_methods_and_classes(code)

    # Index each method individually
    for method_name, method_code in extracted_methods.items():
        emb = embed_text(method_code)
        index_methods.add(emb)
        metadata_methods.append({
            "chunk_name": method_name,
            "chunk_code": method_code,
            "class_name": None  # top-level (no class)
        })

    # Index entire classes as a single embedding (the concatenation of their methods)
    for class_name, methods in extracted_classes.items():
        if not methods:
            continue

        class_embedding = embed_text("\n".join(methods))
        index_methods.add(class_embedding)
        metadata_methods.append({
            "chunk_name": class_name,
            "chunk_code": "\n".join(methods),
            "class_name": class_name
        })

        # Also store the class + methods in metadata_classes
        metadata_classes.append({
            "class_name": class_name,
            "methods": methods
        })


############################################
#  Searching for Similar Methods
############################################

def search_similar_methods(query_code: str, top_k=5):
    """
    Given some snippet of code (presumably a method), returns the top_k similar methods/classes
    from the FAISS index, sorted by approximate similarity score.
    """
    if index_methods is None or index_methods.ntotal == 0:
        return []

    query_emb = embed_text(query_code)
    # Search in the index for top_k
    distances, indices = index_methods.search(query_emb, top_k)
    results = []

    for i, idx in enumerate(indices[0]):
        if 0 <= idx < len(metadata_methods):
            candidate = metadata_methods[idx]["chunk_code"]
            # We do a secondary check: compute actual similarity using cosine to ensure correctness
            # (Because the index uses IP, might differ from pure cosine if norms differ.)
            candidate_emb = embed_text(candidate)
            sim = compute_pairwise_similarity(query_emb, candidate_emb)
            results.append((
                metadata_methods[idx]["chunk_name"],
                candidate,
                round(sim, 4)
            ))

    # Sort by similarity descending
    results.sort(key=lambda x: x[2], reverse=True)
    return results


def get_class_methods_from_metadata():
    """
    Builds a dict: class_name -> [list_of_method_code]
    from the globally collected metadata_methods.
    """
    class_dict = {}
    for entry in metadata_methods:
        cls_name = entry["class_name"]
        if cls_name is None:
            continue
        # If the chunk_name == class_name, it means the entire class code. We skip that chunk
        # for method-level comparisons.
        if entry["chunk_name"] == cls_name:
            continue
        class_dict.setdefault(cls_name, []).append(entry["chunk_code"])
    return class_dict


############################################
#  Similarity Analysis Methods
############################################

def analyze_class_similarity(class_code: str, similarity_threshold=0.7, coverage_threshold=0.6):
    """
    Given a new class definition (as a string),
    compare it to existing classes in the index.

    Returns a dictionary:
    {
      "redundant_code": bool,
      "coverage_map": [...],
      "best_coverage": float,
      "most_similar_class": str,
      "class_similarity_score": float,
      ...
    }
    """
    # 1) Extract methods from the input class code
    _, extracted_classes = extract_methods_and_classes(class_code)
    if not extracted_classes:
        return {
            "redundant_code": False,
            "reason": "No class or methods found in the input code."
        }

    # We'll only handle the first class if there's multiple
    input_class_name = list(extracted_classes.keys())[0]
    input_class_methods = extracted_classes[input_class_name]
    if not input_class_methods:
        return {
            "redundant_code": False,
            "reason": "No methods found in the class."
        }

    # 2) Get the existing classes & methods
    indexed_classes = get_class_methods_from_metadata()
    if not indexed_classes:
        return {
            "redundant_code": False,
            "reason": "No indexed classes to compare against."
        }

    # We'll embed all input methods
    input_method_embeddings = [embed_text(m_code) for m_code in input_class_methods]
    total_methods = len(input_method_embeddings)

    # We'll track each existing class's average similarity
    class_similarity_scores = {}

    for existing_class_name, existing_methods in indexed_classes.items():
        existing_method_embeddings = [embed_text(m) for m in existing_methods]

        sum_best_sims = 0.0
        method_best_sims = []
        for input_emb in input_method_embeddings:
            best_sim = 0.0
            for existing_emb in existing_method_embeddings:
                sim = compute_pairwise_similarity(input_emb, existing_emb)
                if sim > best_sim:
                    best_sim = sim
            method_best_sims.append(best_sim)
            sum_best_sims += best_sim

        avg_sim = sum_best_sims / total_methods if total_methods > 0 else 0.0
        class_similarity_scores[existing_class_name] = {
            "average_similarity": avg_sim,
            "method_best_sims": method_best_sims
        }

    # Find the class with highest average similarity
    most_similar_class = None
    class_similarity_score = 0.0
    for cls_name, sim_info in class_similarity_scores.items():
        if sim_info["average_similarity"] > class_similarity_score:
            class_similarity_score = sim_info["average_similarity"]
            most_similar_class = cls_name

    # Compute coverage map = best similarity for each method across *all classes*
    coverage_map = []
    for input_emb in input_method_embeddings:
        best_overall = 0.0
        for existing_class_name, existing_methods in indexed_classes.items():
            for existing_method_code in existing_methods:
                existing_emb = embed_text(existing_method_code)
                sim = compute_pairwise_similarity(input_emb, existing_emb)
                if sim > best_overall:
                    best_overall = sim
        coverage_map.append(best_overall)

    # fraction of methods above threshold
    coverage_score = sum(1 for s in coverage_map if s >= similarity_threshold) / total_methods
    best_coverage = max(coverage_map) if coverage_map else 0.0
    is_redundant = (coverage_score >= coverage_threshold)

    return {
        "redundant_code": is_redundant,
        "coverage_map": [round(s, 4) for s in coverage_map],
        "best_coverage": round(best_coverage, 4),
        "most_similar_class": most_similar_class,
        "class_similarity_score": round(class_similarity_score, 4)
    }


def analyze_method_similarity(method_code: str, similarity_threshold=0.7):
    """
    Given a single method (no class definition), see if it matches any existing method.
    
    Returns a dict:
    {
      "redundant_code": bool,
      "highest_similarity": float,
      "most_similar_method": str,
      "complementary_class": str
    }
    """
    matches = search_similar_methods(method_code, top_k=5)
    if not matches:
        return {
            "redundant_code": False,
            "highest_similarity": 0.0,
            "most_similar_method": None,
            "complementary_class": None
        }

    best_similarity = max(match[2] for match in matches)
    most_similar_method = max(matches, key=lambda x: x[2])[0]
    is_redundant = (best_similarity >= similarity_threshold)

    # Optionally, attempt to see if this method "belongs" in some existing class
    complementary_class = None
    if not is_redundant:
        new_method_emb = embed_text(method_code)
        best_class_name = None
        best_class_sim = 0.0
        for cdata in metadata_classes:
            class_name = cdata["class_name"]
            methods = cdata["methods"]
            if not methods:
                continue
            sims = []
            for m_code in methods:
                existing_emb = embed_text(m_code)
                sims.append(compute_pairwise_similarity(new_method_emb, existing_emb))
            avg_sim = sum(sims) / len(sims)
            # Example threshold to decide if it "belongs" to this class
            if avg_sim > 0.48 and avg_sim > best_class_sim:
                best_class_sim = avg_sim
                best_class_name = class_name

        complementary_class = best_class_name

    return {
        "redundant_code": is_redundant,
        "highest_similarity": round(best_similarity, 4),
        "most_similar_method": most_similar_method,
        "complementary_class": complementary_class
    }


############################################
#  Main Driver
############################################

def run_funct(existing_code_dummy_file, new_code_input):
    """
    In your original usage, you had a single file 'all_code.txt' that you embed.
    Here, we assume you've already indexed all code in the repo EXCEPT this new snippet.
    Then we see if 'new_code_input' is a class or method.
    """

    # We won't call process_code_file again since we already indexed everything
    # (But your original code might do so. Decide how you want it.)

    is_class = new_code_input.strip().startswith("class ")
    if is_class:
        return analyze_class_similarity(new_code_input)
    else:
        return analyze_method_similarity(new_code_input)


############################################
#  Index Existing Code & Analyze Changes
############################################

def index_existing_codebase(root_dir="."):
    """
    Recursively find all .py files and index them in the FAISS index.
    """
    py_files = glob.glob(f"{root_dir}/**/*.py", recursive=True)
    for py_file in py_files:
        # Optionally filter out tests, venv, etc.
        process_code_file(py_file)


def get_changed_files():
    """
    Returns a list of changed .py files in the last commit.
    If there's no previous commit, or if run in ephemeral env, might fail or return empty.
    """
    cmd = ["git", "diff", "--name-only", "HEAD^", "HEAD"]
    try:
        output = subprocess.check_output(cmd).decode("utf-8").strip()
    except subprocess.CalledProcessError:
        # If there's no previous commit, fallback or handle differently
        return []
    changed_files = [line for line in output.split("\n") if line.endswith(".py")]
    return changed_files


def analyze_changed_files(changed_files):
    """
    Reads each changed file, runs `run_funct()` to decide redundancy.
    """
    results = []
    for cf in changed_files:
        with open(cf, "r", encoding="utf-8") as f:
            new_code_input = f.read()
        result = run_funct(existing_code_dummy_file="all_code.txt", new_code_input=new_code_input)
        results.append((cf, result))
    return results


def main():
    # 1) Initialize (or re-initialize) the FAISS index
    initialize_faiss()

    # 2) Index all existing .py files in the repo
    #    (Skipping or including the changed file is up to you. 
    #     Often you'd want to index everything except the changed one.)
    index_existing_codebase(root_dir=".")

    # 3) Identify changed files from the last commit
    changed_files = get_changed_files()
    if not changed_files:
        print("No .py files changed in this commit.")
        return

    # 4) Analyze changes
    all_results = analyze_changed_files(changed_files)

    # 5) Report & optionally fail if redundant
    for filename, result in all_results:
        print(f"\n=== Redundancy Check for {filename} ===")
        print(result)

        # If analyzing a class, you'll see "redundant_code" and "class_similarity_score"
        # If analyzing a method, you'll see "redundant_code" and "highest_similarity"
        # Adjust thresholds or logic as needed

        if isinstance(result, dict):
            # Class check
            if result.get("redundant_code", False):
                coverage = result.get("best_coverage", 0.0)
                sim_score = result.get("class_similarity_score", 0.0)
                raise SystemExit(
                    f"ERROR: {filename} is highly similar to existing code. "
                    f"(coverage={coverage}, avg_sim={sim_score})"
                )

            # Method check
            if result.get("redundant_code", False):
                best_sim = result.get("highest_similarity", 0.0)
                raise SystemExit(
                    f"ERROR: {filename} has a method too similar to existing code "
                    f"(similarity={best_sim})."
                )

    # If we get here, no redundancy triggered the threshold
    print("No redundant code detected above thresholds. Build passes.")


if __name__ == "__main__":
    main()

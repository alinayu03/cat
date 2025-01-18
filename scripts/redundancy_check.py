import os
import torch
import faiss
import json
import ast
import numpy as np
import math
from transformers import AutoModel, AutoTokenizer

# Set up embedding model
CHECKPOINT = "Salesforce/codet5p-110m-embedding"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)
model = AutoModel.from_pretrained(CHECKPOINT, trust_remote_code=True).to(DEVICE)

CONTEXT_WINDOW = 512
D = 256  # Embedding dimension

# FAISS Indexes
index_methods = None
metadata_methods = []
metadata_classes = []


def initialize_faiss():
    """Initializes FAISS index for method-level similarity search."""
    global index_methods
    index_methods = faiss.IndexFlatIP(D)  # Dot product similarity


def extract_methods_and_classes(code):
    """Extracts both method and class definitions from the input code."""
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


def embed_text(text):
    """Generates embeddings from text using CodeT5."""
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=CONTEXT_WINDOW).to(DEVICE)
    with torch.no_grad():
        embedding = model(**tokens)[0]  # Extract embedding
    return embedding.cpu().numpy().reshape(1, -1)


def process_code_file(filepath):
    global index_methods, metadata_methods, metadata_classes
    if index_methods is None:
        initialize_faiss()

    with open(filepath, "r", encoding="utf-8") as file:
        code = file.read()

    extracted_methods, extracted_classes = extract_methods_and_classes(code)

    # Index methods individually
    for method_name, method_code in extracted_methods.items():
        embedding = embed_text(method_code)
        index_methods.add(embedding)
        metadata_methods.append({
            "chunk_name": method_name,
            "chunk_code": method_code,
            "class_name": None
        })

    # Index entire classes by embedding all their methods together
    for class_name, methods in extracted_classes.items():
        if not methods:
            continue
        # 1) Add to FAISS (just like you had before)
        class_embedding = embed_text("\n".join(methods))
        index_methods.add(class_embedding)
        metadata_methods.append({
            "chunk_name": class_name,
            "chunk_code": "\n".join(methods),
            "class_name": class_name
        })

        # 2) Also store the methods in metadata_classes for later
        #    so that the "complementary_class" logic can find them
        metadata_classes.append({
            "class_name": class_name,
            "methods": methods  # list of "def foo" code strings
        })


def search_similar_methods(query_code, top_k=5):
    """Searches for similar methods in the FAISS index."""
    if index_methods is None or index_methods.ntotal == 0:
        return []

    query_embedding = embed_text(query_code)
    distances, indices = index_methods.search(query_embedding, top_k)
    results = []

    for i, idx in enumerate(indices[0]):
        if idx < len(metadata_methods):
            candidate_code = metadata_methods[idx]["chunk_code"]
            similarity = float(torch.tensor(query_embedding) @ torch.tensor(embed_text(candidate_code)).T)
            results.append((metadata_methods[idx]["chunk_name"], candidate_code, round(similarity, 4)))

    results.sort(key=lambda x: x[2], reverse=True)
    return results


def compute_pairwise_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Computes cosine similarity between two embeddings (shape [1, D]).
    """
    # Flatten them to 1D
    e1 = embedding1.flatten()
    e2 = embedding2.flatten()
    num = np.dot(e1, e2)
    den = (np.linalg.norm(e1) * np.linalg.norm(e2)) + 1e-12  # small epsilon to avoid /0
    return float(num / den)


def get_class_methods_from_metadata():
    """
    Returns a dictionary of class_name -> list of method_code
    extracted from global 'metadata_methods'.
    """
    class_dict = {}
    for entry in metadata_methods:
        cls_name = entry["class_name"]
        # Some entries might be top-level methods, so class_name could be None.
        if cls_name:
            # If it's the entire class chunk, we skip or filter it out;
            # we only want actual *method* chunks.
            # But if you stored each method with the same class_name, it will appear here individually.
            class_dict.setdefault(cls_name, []).append(entry["chunk_code"])
    return class_dict


def analyze_class_similarity(class_code, similarity_threshold=0.7, coverage_threshold=0.6):
    """
    Compares the incoming class's methods to every existing class,
    picking the most similar class and computing an average similarity score.

    Returns a dictionary with:
        - redundant_code: bool indicating if coverage >= coverage_threshold
        - coverage_map: list of best-match similarities for each method in the input class
        - best_coverage: max similarity among coverage_map
        - most_similar_class: name of the class with the highest average similarity
        - class_similarity_score: that highest average similarity
    """
    # 1) Extract methods from input class_code
    _, extracted_classes = extract_methods_and_classes(class_code)
    if not extracted_classes:
        return {
            "redundant_code": False,
            "reason": "No class or methods found in the input code."
        }

    # For simplicity, handle the first (or only) class from extracted_classes
    input_class_name = list(extracted_classes.keys())[0]
    input_class_methods = extracted_classes[input_class_name]  # list of method-code strings

    # 2) Get existing classes and their methods from metadata
    indexed_classes = get_class_methods_from_metadata()
    if not indexed_classes:
        return {
            "redundant_code": False,
            "reason": "No indexed classes to compare against."
        }

    # Prepare to track method-level coverage (how many methods pass the threshold)
    coverage_map = []  # store the best similarity found for each input method

    # We'll also track each existing class's "average similarity" to the input class
    class_similarity_scores = {}

    # Pre-embed all input methods to avoid repeated calls
    input_method_embeddings = [embed_text(m_code) for m_code in input_class_methods]
    total_methods = len(input_method_embeddings)

    # 3) For each existing class, compute the average of "best method matches"
    for existing_class_name, existing_methods in indexed_classes.items():
        # Pre-embed existing class's methods
        existing_method_embeddings = [embed_text(m_code) for m_code in existing_methods]

        # For each input method embedding, find the best match in this existing class
        sum_of_best_similarities = 0.0
        method_best_sims = []  # store best sim for each method in this class

        for input_embedding in input_method_embeddings:
            best_sim = 0.0
            for existing_embedding in existing_method_embeddings:
                sim = compute_pairwise_similarity(input_embedding, existing_embedding)
                if sim > best_sim:
                    best_sim = sim
            method_best_sims.append(best_sim)
            sum_of_best_similarities += best_sim

        # The average similarity for "input_class" vs "existing_class_name"
        avg_class_sim = sum_of_best_similarities / total_methods if total_methods > 0 else 0.0
        class_similarity_scores[existing_class_name] = {
            "average_similarity": avg_class_sim,
            "method_best_sims": method_best_sims  # optional tracking
        }

    # 4) Identify the class with the highest average similarity
    most_similar_class = None
    class_similarity_score = 0.0

    for cls_name, sim_info in class_similarity_scores.items():
        if sim_info["average_similarity"] > class_similarity_score:
            class_similarity_score = sim_info["average_similarity"]
            most_similar_class = cls_name


    for input_embedding in input_method_embeddings:
        # find best similarity across *all methods of all classes*
        best_overall = 0.0
        for existing_class_name, existing_methods in indexed_classes.items():
            for existing_method_code in existing_methods:
                sim = compute_pairwise_similarity(input_embedding, embed_text(existing_method_code))
                if sim > best_overall:
                    best_overall = sim
        coverage_map.append(best_overall)

    # fraction of methods above the threshold
    coverage_score = sum(1 for s in coverage_map if s >= similarity_threshold) / total_methods if total_methods else 0
    best_coverage = max(coverage_map) if coverage_map else 0.0
    is_redundant = (coverage_score >= coverage_threshold)

    # Final result
    return {
        "redundant_code": is_redundant,
        "coverage_map": [round(s, 4) for s in coverage_map],
        "best_coverage": round(best_coverage, 4),
        "most_similar_class": most_similar_class,
        "class_similarity_score": round(class_similarity_score, 4)
    }


def analyze_method_similarity(method_code, similarity_threshold=0.7):
    """
    Analyzes method redundancy and finds a complementary class if applicable.
    """
   
    matches = search_similar_methods(method_code, top_k=5)

    if not matches:
        # If no matches at all, no direct redundancy
        return {
            "redundant_code": False,
            "highest_similarity": 0.0,
            "most_similar_method": None,
            "complementary_class": None
        }

    # 2) Determine best match
    best_similarity = max(sim for _, _, sim in matches)
    most_similar_method = max(matches, key=lambda x: x[2])[0]

    # 3) Decide if the new method is "redundant" based on a threshold
    is_redundant = (best_similarity >= similarity_threshold)

    # 4) If not redundant, try to see if it "belongs" in some existing class
    complementary_class = None
    if not is_redundant:
        # Embed the new method once, to avoid repeated embeddings
        new_method_embedding = embed_text(method_code)  # shape [1, D]

        # metadata_classes might be a list of dictionaries, e.g.:
        # [
        #   {"class_name": "SomeClass", "methods": ["def foo(): ...", ...]},
        #   {"class_name": "OtherClass", "methods": [...]}
        # ]
       
        best_class_name = None
        best_class_sim = 0.0

        for existing_class in metadata_classes:
            class_name = existing_class["class_name"]
            class_methods = existing_class["methods"]
            if not class_methods:
                continue

            # Embed each method in this class, compute similarity to the new method
            sims = []
            for m_code in class_methods:
                existing_method_emb = embed_text(m_code)
                cos_sim = compute_pairwise_similarity(new_method_embedding, existing_method_emb)
                sims.append(cos_sim)

            avg_class_sim = sum(sims) / len(sims)

            if avg_class_sim > 0.48 and avg_class_sim > best_class_sim:
                best_class_sim = avg_class_sim
                best_class_name = class_name

        complementary_class = best_class_name

    # Return final results
    return {
        "redundant_code": is_redundant,
        "highest_similarity": round(best_similarity, 4),
        "most_similar_method": most_similar_method,
        "complementary_class": complementary_class
    }


def run_funct(main_code_file, new_code_input):
    """Processes new code input and determines redundancy using method-based class comparison."""
    process_code_file(main_code_file)

    is_class = new_code_input.strip().startswith("class ")
    if is_class:
        return analyze_class_similarity(new_code_input)
    else:
        return analyze_method_similarity(new_code_input)


if __name__ == "__main__":
    code_file = code_file = os.path.join(os.getenv('GITHUB_WORKSPACE', '.'), 'all_code.txt')


    test_cases = {
        # =========== REDUNDANT CLASSES ===========
        "Redundant Class 1 (RotatingCatAnimator)": """class RotatingCatAnimator(BaseAnimator):
            def animate(self):
                if not self.frame_manager:
                    return
                cycle = 0
                while cycle < self.repeat:  
                    idx = 0
                    while idx < len(self.frame_manager):
                        self._clear_console()
                        print(Fore.YELLOW + f"Cycle {cycle+1}/{self.repeat}, " + Fore.CYAN + f"Frame {idx+1}/{len(self.frame_manager)}\\n")
                        print(Fore.GREEN + str(self.frame_manager.get_frames()[idx]))
                        time.sleep(self.delay)
                        idx += 1
                    cycle += 1
        """,

        "Redundant Class 2 (MirroredSpinningCatAnimator)": """class MirroredSpinningCatAnimator(BaseAnimator):
            def animate(self):
                if not self.frame_manager:
                    return
                for cycle in range(self.repeat):
                    frames = self.frame_manager.get_frames()
                    for idx in range(len(frames) * 2):  # Doubles frame count, mirrors the second half
                        self._clear_console()
                        frame = frames[idx % len(frames)]
                        print(Fore.YELLOW + f"Cycle {cycle+1}/{self.repeat}, " + Fore.CYAN + f"Frame {idx+1}/{len(frames)*2}\\n")
                        print(Fore.GREEN + ("\\n".join(line[::-1] for line in str(frame).split("\\n")) if idx >= len(frames) else str(frame)))
                        time.sleep(self.delay)
        """,

        # =========== NOVEL CLASSES ===========
        "Novel Class 1 (RainbowCatAnimator)": """class RainbowCatAnimator(BaseAnimator):
            def __init__(self, frame_manager, delay=0.4, repeat=1, clear_screen=True, rainbow_speed=0.3):
                super().__init__(frame_manager, delay, repeat, clear_screen)
                self.rainbow_speed = rainbow_speed
                self.colors = [Fore.RED, Fore.YELLOW, Fore.GREEN, Fore.CYAN, Fore.BLUE, Fore.MAGENTA]

            def _rainbow_text(self, text, frame_idx):
                color_cycle = self.colors[frame_idx % len(self.colors)]
                return "\\n".join(color_cycle + line for line in text.split("\\n"))

            def _add_trail(self, text, frame_idx):
                trail = "." * (frame_idx % 5)
                return "\\n".join(line + trail for line in text.split("\\n"))

            def animate(self):
                if not self.frame_manager:
                    return
                for cycle in range(self.repeat):
                    for idx, frame in enumerate(self.frame_manager.get_frames()):
                        self._clear_console()
                        print(Fore.YELLOW + f"Rainbow Cycle {cycle+1}/{self.repeat}")
                        print(Fore.CYAN + f"Frame {idx+1}/{len(self.frame_manager)}\\n")
                        styled_text = self._rainbow_text(str(frame), idx)
                        styled_text = self._add_trail(styled_text, idx)
                        print(styled_text)
                        time.sleep(self.rainbow_speed)
        """,

        "Novel Class 2 (ShadowCatAnimator)": """class ShadowCatAnimator(BaseAnimator):
            def __init__(self, frame_manager, delay=0.5, repeat=1, clear_screen=True, shadow_offset=3):
                super().__init__(frame_manager, delay, repeat, clear_screen)
                self.shadow_offset = shadow_offset

            def _apply_shadow(self, text):
                shadow = " " * self.shadow_offset + Fore.BLACK + text.replace("\\n", "\\n" + " " * self.shadow_offset)
                return text + "\\n" + shadow

            def _dim_text(self, text, intensity=0.5):
                return "".join(c.lower() if random.random() < intensity else c for c in text)

            def animate(self):
                if not self.frame_manager:
                    return
                for cycle in range(self.repeat):
                    for idx, frame in enumerate(self.frame_manager.get_frames()):
                        self._clear_console()
                        print(Fore.YELLOW + f"Shadow Cycle {cycle+1}/{self.repeat}")
                        print(Fore.CYAN + f"Frame {idx+1}/{len(self.frame_manager)}\\n")
                        dimmed = self._dim_text(str(frame))
                        shadowed = self._apply_shadow(dimmed)
                        print(Fore.LIGHTWHITE_EX + shadowed)
                        time.sleep(self.delay)
        """,

        # =========== REDUNDANT METHODS ===========
        "Redundant Method 1 (_shake_lines)": """def _shake_lines(self, text):
            lines = text.split("\\n")
            adjusted_lines = []
            for line in lines:
                shift = random.randint(-self.amplitude, self.amplitude)
                adjusted_lines.append((" " * shift + line) if shift > 0 else line)
            return "\\n".join(adjusted_lines)
        """,

        "Redundant Method 2 (_flip_text)": """def _flip_text(self, text):
            lines = text.split("\\n")
            return "\\n".join(lines[::-1])
        """,

        # =========== NOVEL METHODS ===========
        "Novel Method 1 (_invert_colors)": """def _invert_colors(self, text):
            return "".join(Fore.BLACK + c if c.isalnum() else Fore.WHITE + c for c in text)
        """,

        "Novel Method 2 (_fade_out_effect)": """def _fade_out_effect(self, text, step, max_steps):
            fade_ratio = step / max_steps
            return "".join(c if random.random() > fade_ratio else "." for c in text)
        """,

        "Novel Method 3": """def _pulse_brightness(self, text, frame_idx):
            pulse_factor = 0.5 * (1 + math.sin(2 * math.pi * frame_idx / 30))  
            brightness_levels = [Fore.LIGHTBLACK_EX, Fore.LIGHTWHITE_EX, Fore.WHITE]  
            brightness_idx = int(pulse_factor * (len(brightness_levels) - 1))
            return "".join(brightness_levels[brightness_idx] + c for c in text)
    """

    }

    # ======= RUN TEST CASES =======
    print("\n====== Running Tests for Classes & Methods ======\n")
    for test_name, q_sim in test_cases.items():
        print(f"\n### {test_name} ###")
        result = run_funct(code_file, q_sim)
        print(result)
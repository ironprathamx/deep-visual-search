
import json
import pickle
import time
from pathlib import Path

from search import VisualSearchEngine, MODEL_NAME, PRETRAINED


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "dataset" / "ukbench"
RESULTS_DIR = ROOT / "results"

NUM_IMAGES = 10200
IMAGES_PER_GROUP = 4
TOP_K = 11

QUERY_INDICES = range(0, NUM_IMAGES, IMAGES_PER_GROUP)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_image_path(index):
    return DATASET_DIR / f"ukbench{index:05d}.jpg"


def get_relevant_images(query_index):
    group_start = (
        query_index // IMAGES_PER_GROUP
    ) * IMAGES_PER_GROUP

    return {
        f"ukbench{i:05d}.jpg"
        for i in range(
            group_start,
            group_start + IMAGES_PER_GROUP,
        )
        if i != query_index
    }


def validate_dataset():
    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {DATASET_DIR}"
        )

    missing = [
        get_image_path(i).name
        for i in range(NUM_IMAGES)
        if not get_image_path(i).is_file()
    ]

    if missing:
        preview = ", ".join(missing[:10])
        raise FileNotFoundError(
            f"Missing {len(missing)} UKBench images. "
            f"Examples: {preview}"
        )


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():
    print("=" * 70)
    print("FINAL FULL UKBENCH EVALUATION - DEEP CLIP")
    print("=" * 70)

    print(f"Dataset images      : {NUM_IMAGES}")
    print(f"Queries evaluated   : {len(QUERY_INDICES)}")
    print(f"Images per group    : {IMAGES_PER_GROUP}")
    print("Relevant per query  : 3 other group images")
    print(f"Top-K               : {TOP_K}")
    print(f"Model               : {MODEL_NAME}")
    print(f"Pretrained weights  : {PRETRAINED}")
    print("Ranking             : Cosine similarity")

    print("=" * 70)

    validate_dataset()

    # Load the existing index and model.
    # This does not rebuild the 10,200-image index.
    engine = VisualSearchEngine()

    num_queries = len(QUERY_INDICES)

    total_relevant = {k: 0 for k in (1, 4, 10)}
    hits = {k: 0 for k in (1, 4, 10)}
    reciprocal_rank_sum = 0.0
    ns_sum = 0.0
    failed_queries = []

    overall_start = time.time()

    for query_number, query_index in enumerate(
        QUERY_INDICES, start=1
    ):
        query_path = get_image_path(query_index)
        query_filename = query_path.name
        relevant = get_relevant_images(query_index)

        query_start = time.time()

        try:
            output = engine.search_by_image(
                query_path,
                top_k=TOP_K,
                INCLUDE_QUERY=True
            )

            # Normalize paths to filenames so comparisons
            # work whether search returns paths or filenames.
            ranked_images = [
                Path(result["path"]).name
                for result in output
            ]

            ns_score = sum(
                image in relevant or image==query_filename
                for image in ranked_images[:4]
            )
            ns_sum += ns_score
            ranked_images = [
                image for image in ranked_images
                if image != query_filename
            ]
            for k in (1, 4, 10):
                top_k_images = ranked_images[:k]

                relevant_count = sum(
                    image in relevant
                    for image in top_k_images
                )

                total_relevant[k] += relevant_count

                if relevant_count > 0:
                    hits[k] += 1

            reciprocal_rank = 0.0

            for rank, image in enumerate(
                ranked_images, start=1
            ):
                if image in relevant:
                    reciprocal_rank = 1.0 / rank
                    break

            reciprocal_rank_sum += reciprocal_rank

            if (
                query_number == 1
                or query_number % 25 == 0
                or query_number == num_queries
            ):
                elapsed = time.time() - overall_start
                average_time = elapsed / query_number
                remaining = num_queries - query_number
                eta = remaining * average_time

                print(
                    f"[{query_number:4d}/{num_queries}] "
                    f"{query_filename} | "
                    f"N-S: {ns_score}/4 | "
                    f"Top-4 relevant: "
                    f"{total_relevant[4] / max(1, query_number - len(failed_queries)):.2f} "
                    f"(running mean) | "
                    f"Time: "
                    f"{time.time() - query_start:.2f}s | "
                    f"ETA: {eta / 60:.1f} min"
                )

        except Exception as exc:
            failed_queries.append({
                "index": query_index,
                "filename": query_filename,
                "error": str(exc),
            })

            print(
                f"[ERROR] {query_number}/{num_queries} "
                f"{query_filename}: {exc}"
            )

    successful_queries = num_queries - len(failed_queries)

    if successful_queries == 0:
        print("No successful queries.")
        return

    # ========================================================
    # FINAL METRICS
    # ========================================================

    recall = {
        k: total_relevant[k] / (3 * successful_queries)
        for k in (1, 4, 10)
    }

    precision = {
        k: total_relevant[k] / (k * successful_queries)
        for k in (1, 4, 10)
    }

    hit_rate = {
        k: hits[k] / successful_queries
        for k in (1, 4, 10)
    }

    mrr = reciprocal_rank_sum / successful_queries
    ns_score = ns_sum / successful_queries
    total_time = time.time() - overall_start
    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL FULL UKBENCH EVALUATION - DEEP CLIP")
    print("=" * 70)

    print(f"Queries evaluated: {successful_queries}")
    print(f"Images in dataset: {NUM_IMAGES}")
    print(f"Failed queries: {len(failed_queries)}")

    print("\n--- UKBench N-S ---")
    print(f"Mean N-S score: {ns_score:.4f} / 4")
    print(f"Mean N-S percentage: {ns_score / 4 * 100:.2f}%")

    print("\n--- Retrieval Performance ---")

    for k in (1, 4, 10):
        print(
            f"Recall@{k}: {recall[k]:.4f} "
            f"({recall[k] * 100:.2f}%)"
        )

    for k in (1, 4, 10):
        print(f"Precision@{k}: {precision[k]:.4f}")

    print(f"MRR: {mrr:.4f}")

    print("\n--- Hit Rates ---")

    for k in (1, 4, 10):
        print(f"Hit@{k}: {hit_rate[k] * 100:.2f}%")

    print("\n--- Runtime ---")
    print(f"Total evaluation time: {total_time / 60:.2f} minutes")
    print(
        f"Average time per query: "
        f"{total_time / successful_queries:.2f} seconds"
    )

    if failed_queries:
        print("\n--- Failed Queries ---")
        for failure in failed_queries:
            print(
                f"{failure['index']}: "
                f"{failure['filename']} -> "
                f"{failure['error']}"
            )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    summary = {
        "configuration": {
            "dataset": "UKBench",
            "dataset_size": NUM_IMAGES,
            "queries_attempted": num_queries,
            "queries_successful": successful_queries,
            "images_per_group": IMAGES_PER_GROUP,
            "model": MODEL_NAME,
            "pretrained": PRETRAINED,
            "embedding_dimensions": int(
                engine.embeddings.shape[1]
            ),
            "ranking": "cosine_similarity",
            "top_k": TOP_K,
            "query_excluded": False,
            "ns_protocol": (
                "Count same-group images among the top 4 results, "
                "including the query image if it appears"
            ),
        },
        "metrics": {
            "N-S": ns_score,
            "N-S_percentage": ns_score / 4,
            "recall@1": recall[1],
            "recall@4": recall[4],
            "recall@10": recall[10],
            "precision@1": precision[1],
            "precision@4": precision[4],
            "precision@10": precision[10],
            "mrr": mrr,
            "hit@1": hit_rate[1],
            "hit@4": hit_rate[4],
            "hit@10": hit_rate[10],
            "total_evaluation_time_seconds": total_time,
            "average_query_time_seconds": (
                total_time / successful_queries
            ),
        },
        "failed_queries": failed_queries,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    json_path = RESULTS_DIR / "deep_clip_eval_summary.json"
    pickle_path = RESULTS_DIR / "deep_clip_eval_summary.pkl"

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    with open(pickle_path, "wb") as file:
        pickle.dump(summary, file)

    print("\nSaved:")
    print(f"  {json_path}")
    print(f"  {pickle_path}")
    print("=" * 70)
    print("EVALUATION COMPLETE")


if __name__ == "__main__":
    main()

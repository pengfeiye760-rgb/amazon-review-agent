import pandas as pd

from tools import classify_reviews


EVAL_DATA_PATH = "data/labelled_eval_reviews.csv"
OUTPUT_PATH = "data/evaluation_results.csv"


def run_evaluation():
    df = pd.read_csv(EVAL_DATA_PATH)

    if "human_label" not in df.columns:
        raise ValueError("The evaluation CSV must contain a 'human_label' column.")

    # Ensure required fields exist for the classifier
    if "review_title" not in df.columns:
        df["review_title"] = ""

    if "rating" not in df.columns:
        df["rating"] = None

    if "review_id" not in df.columns:
        df["review_id"] = [f"E{i+1:03d}" for i in range(len(df))]

    classified_df = classify_reviews(df)

    classified_df["is_correct"] = (
        classified_df["agent_category"] == classified_df["human_label"]
    )

    total = len(classified_df)
    correct = int(classified_df["is_correct"].sum())
    accuracy = correct / total if total > 0 else 0

    print("Amazon Review Analysis Agent Evaluation")
    print("--------------------------------------")
    print(f"Total examples: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.2%}")

    print("\nIncorrect cases:")
    incorrect_cases = classified_df[classified_df["is_correct"] == False]

    if len(incorrect_cases) == 0:
        print("No incorrect cases.")
    else:
        print(
            incorrect_cases[
                [
                    "review_id",
                    "customer_comment",
                    "human_label",
                    "agent_category",
                    "agent_confidence",
                    "agent_evidence",
                ]
            ].to_string(index=False)
        )

    classified_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nDetailed evaluation results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    run_evaluation()
import pandas as pd

from tools import (
    validate_schema,
    classify_reviews,
    compute_category_statistics,
    extract_evidence_examples,
    generate_qc_recommendations,
)


def observe_dataset(df: pd.DataFrame):
    """Observe the uploaded review dataset and summarize its condition."""
    total_rows = len(df)
    total_columns = len(df.columns)

    comment_missing_rate = 0
    low_rating_rate = 0

    if "customer_comment" in df.columns and total_rows > 0:
        comment_missing_rate = df["customer_comment"].isna().mean() * 100

    if "rating" in df.columns and total_rows > 0:
        low_rating_rate = (df["rating"] <= 3).mean() * 100

    return {
        "total_rows": total_rows,
        "total_columns": total_columns,
        "columns": list(df.columns),
        "comment_missing_rate": round(comment_missing_rate, 2),
        "low_rating_rate": round(low_rating_rate, 2),
    }


def decide_analysis_path(observation, schema_result):
    """Decide which analysis path the agent should follow."""
    warnings = []

    if not schema_result["is_valid"]:
        warnings.append(
            f"Missing required columns: {schema_result['missing_columns']}. "
            "The analysis may be incomplete."
        )

    if observation["comment_missing_rate"] > 30:
        warnings.append(
            "More than 30% of customer comments are missing. "
            "The agent will rely more on review titles and ratings."
        )

    if observation["low_rating_rate"] > 30:
        analysis_focus = "defect-focused"
    else:
        analysis_focus = "balanced"

    return {
        "analysis_focus": analysis_focus,
        "warnings": warnings,
    }


def run_review_analysis_agent(df: pd.DataFrame):
    """
    Main agent workflow:
    1. Observe dataset
    2. Validate schema
    3. Decide analysis path
    4. Classify reviews
    5. Compute statistics
    6. Generate evidence and QC recommendations
    """
    observation = observe_dataset(df)
    schema_result = validate_schema(df)
    decision = decide_analysis_path(observation, schema_result)

    if not schema_result["is_valid"]:
        return {
            "observation": observation,
            "schema_result": schema_result,
            "decision": decision,
            "classified_reviews": None,
            "category_statistics": None,
            "evidence_examples": None,
            "qc_recommendations": None,
        }

    classified_reviews = classify_reviews(df)
    category_statistics = compute_category_statistics(classified_reviews)
    evidence_examples = extract_evidence_examples(classified_reviews)
    qc_recommendations = generate_qc_recommendations(category_statistics)

    unknown_rate = 0
    if len(classified_reviews) > 0:
        unknown_rate = (
            classified_reviews["agent_category"].eq("unknown").mean() * 100
        )

    if unknown_rate > 30:
        decision["warnings"].append(
            "More than 30% of reviews were classified as unknown. "
            "The result may require manual review."
        )

    return {
        "observation": observation,
        "schema_result": schema_result,
        "decision": decision,
        "classified_reviews": classified_reviews,
        "category_statistics": category_statistics,
        "evidence_examples": evidence_examples,
        "qc_recommendations": qc_recommendations,
    }
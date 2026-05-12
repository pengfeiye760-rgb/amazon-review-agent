import json
import re
from pathlib import Path

import pandas as pd


MEMORY_PATH = Path("memory.json")


def load_memory():
    """Load defect taxonomy and safety rules from memory.json."""
    with open(MEMORY_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def validate_schema(df: pd.DataFrame):
    """Check whether the uploaded CSV contains the minimum required columns."""
    required_columns = ["review_id", "rating", "review_title", "customer_comment"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    return {
        "is_valid": len(missing_columns) == 0,
        "missing_columns": missing_columns,
        "available_columns": list(df.columns),
    }


def clean_text(text):
    """Clean review text for rule-based classification."""
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def combine_review_text(row):
    """Combine review title and customer comment as the observation text."""
    title = clean_text(row.get("review_title", ""))
    comment = clean_text(row.get("customer_comment", ""))
    return f"{title} {comment}".strip()


def classify_review_by_rules(text):
    """
    Classify one review into a predefined defect taxonomy.

    This is a deterministic rule-based classifier.
    It keeps the first prototype simple and reduces hallucination risk.
    """
    text = clean_text(text)

    if len(text) < 8:
        return {
            "category": "unknown",
            "confidence": 0.3,
            "evidence": text,
        }

    rules = [
        (
            "charging_pin_contact_issue",
            [
                "charging pin",
                "charging pins",
                "prong",
                "prongs",
                "contact with the charging base",
                "make contact",
                "retracted position",
                "charging contact",
            ],
            0.9,
        ),
        (
            "charging_base_failure",
            [
                "charging base stopped",
                "base stopped working",
                "base no longer works",
                "charging base is bad",
                "charging stand doesn’t always",
                "charging stand doesn't always",
                "base charge doesn’t work",
                "base charger is not working",
                "charging station no longer works",
                "faulty charging sensors",
                "salt shaker no longer charges at the base",
            ],
            0.88,
        ),
        (
            "charging_failure",
            [
                "would not recharge",
                "will not charge",
                "stopped charging",
                "stopped recharging",
                "won't charge",
                "doesn't charge",
                "does not charge",
                "no charging indicator",
                "tried other cords",
                "different cables",
                "plugging it into different outlets",
            ],
            0.86,
        ),
        (
            "battery_life_issue",
            [
                "battery life goes weak",
                "battery dies",
                "doesn't hold a charge",
                "does not hold a charge",
                "consistently charge",
                "charge pretty fast",
                "ran out of energy",
            ],
            0.78,
        ),
        (
            "stopped_working",
            [
                "stopped working",
                "stopped after",
                "does not last",
                "poor quality",
                "one grinder stopped",
                "salt grinder also stopped",
                "pepper grinder stopped",
                "no longer works",
                "worked for 3 times",
                "wouldn't grind",
                "would not grind",
                "se daño",
            ],
            0.86,
        ),
        (
            "grinding_power_issue",
            [
                "grinds slowly",
                "too slowly",
                "weak",
                "not as steady",
                "not grind",
                "wouldn't grind",
                "would not grind",
                "grinding is variable",
            ],
            0.8,
        ),
        (
            "coarseness_adjustment_issue",
            [
                "not grind to fine",
                "not grind the salt or pepper to a fine",
                "coarse grind",
                "too coarse",
                "fine coarseness",
                "coarseness setting",
                "grain fine enough",
            ],
            0.8,
        ),
        (
            "salt_or_pepper_jamming",
            [
                "gets stuck",
                "pepper gets stuck",
                "salt crystals",
                "have to shake",
                "restart",
                "jamming",
                "jammed",
            ],
            0.78,
        ),
        (
            "messy_dispensing",
            [
                "mess",
                "messy",
                "salt and pepper everywhere",
                "puts salt and pepper everywhere",
                "a lot of salt",
                "too much salt",
                "empty after use",
                "bumping it",
            ],
            0.82,
        ),
        (
            "small_capacity_or_refill_issue",
            [
                "difficult to fill",
                "filling the grinders is difficult",
                "don’t hold a lot",
                "don't hold a lot",
                "small compartment",
                "compartment for the salt and pepper are very small",
                "hard to clean",
                "base is hard to clean",
            ],
            0.76,
        ),
        (
            "material_or_build_quality_issue",
            [
                "cheap",
                "flimsy",
                "not very solid",
                "not sturdy",
                "plastic",
                "lightweight",
                "light weight",
                "too pricey for the quality",
                "quality is pretty cheap",
                "cheap feel",
            ],
            0.82,
        ),
        (
            "surface_stain_issue",
            [
                "oil stains",
                "permanent oil spots",
                "finish is terrible",
                "permanently stain",
                "shows blemishes",
                "matte finish picks up",
            ],
            0.82,
        ),
        (
            "customer_service_positive",
            [
                "customer service",
                "company responded",
                "sent me a replacement",
                "replacement within",
                "product support was excellent",
                "stood by their product",
                "support is awesome",
            ],
            0.76,
        ),
        (
            "positive_feedback",
            [
                "excellent quality",
                "works flawlessly",
                "works perfectly",
                "works great",
                "great product",
                "love this",
                "easy to use",
                "easy to refill",
                "holds a charge",
                "long charge",
                "sleek",
                "stylish",
                "modern",
                "beautiful",
                "recommend",
                "would buy again",
                "good value",
                "amazing",
                "reliable",
                "durable",
                "perfect",
            ],
            0.75,
        ),
    ]

    for category, keywords, confidence in rules:
        for keyword in keywords:
            if keyword in text:
                return {
                    "category": category,
                    "confidence": confidence,
                    "evidence": keyword,
                }

    return {
        "category": "unknown",
        "confidence": 0.4,
        "evidence": text[:80],
    }


def classify_reviews(df: pd.DataFrame):
    """Classify all reviews and return a new dataframe with agent outputs."""
    result_df = df.copy()

    combined_texts = result_df.apply(combine_review_text, axis=1)
    classifications = combined_texts.apply(classify_review_by_rules)

    result_df["agent_observation_text"] = combined_texts
    result_df["agent_category"] = classifications.apply(lambda x: x["category"])
    result_df["agent_confidence"] = classifications.apply(lambda x: x["confidence"])
    result_df["agent_evidence"] = classifications.apply(lambda x: x["evidence"])

    return result_df


def compute_category_statistics(classified_df: pd.DataFrame):
    """Compute category counts and percentages using pandas."""
    total = len(classified_df)

    stats = (
        classified_df["agent_category"]
        .value_counts()
        .reset_index()
    )
    stats.columns = ["category", "count"]
    stats["percentage"] = (stats["count"] / total * 100).round(2)

    return stats


def extract_evidence_examples(classified_df: pd.DataFrame, max_examples_per_category=2):
    """Extract representative review examples for each category."""
    examples = {}

    for category in classified_df["agent_category"].unique():
        subset = classified_df[classified_df["agent_category"] == category].head(max_examples_per_category)
        examples[category] = subset[
            ["review_id", "rating", "review_title", "customer_comment", "agent_evidence"]
        ].to_dict(orient="records")

    return examples


def generate_qc_recommendations(stats_df: pd.DataFrame):
    """Generate simple QC recommendations based on computed category statistics."""
    recommendation_map = {
        "charging_failure": "Run direct USB-C charging tests across multiple low-amp chargers and check charging indicators.",
        "charging_base_failure": "Stress-test the charging base, dock alignment, and dual-unit charging reliability.",
        "charging_pin_contact_issue": "Inspect spring-loaded charging pins for salt/pepper debris tolerance and contact durability.",
        "battery_life_issue": "Add charge-retention and discharge-cycle testing before shipment.",
        "stopped_working": "Run 2-4 week accelerated usage simulation to detect early-life motor or PCB failure.",
        "grinding_power_issue": "Test grinding torque with peppercorns, coarse salt, and mixed loads.",
        "coarseness_adjustment_issue": "Calibrate the burr adjustment range and verify fine/coarse output consistency.",
        "salt_or_pepper_jamming": "Test jamming risk with different salt crystal sizes and peppercorn types.",
        "messy_dispensing": "Review button placement, dispensing control, and post-use residue leakage.",
        "small_capacity_or_refill_issue": "Review refill opening size, container capacity, and cleaning accessibility.",
        "material_or_build_quality_issue": "Review housing material, weight, stability, and perceived build quality.",
        "surface_stain_issue": "Test matte surface resistance against oil, moisture, and food stains.",
    }

    recommendations = []

    for _, row in stats_df.iterrows():
        category = row["category"]

        if category in recommendation_map:
            recommendations.append(
                {
                    "category": category,
                    "count": int(row["count"]),
                    "percentage": float(row["percentage"]),
                    "recommendation": recommendation_map[category],
                }
            )

    return recommendations
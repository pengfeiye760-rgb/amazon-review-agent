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
    """
    Check whether the uploaded CSV contains the minimum required columns.

    The agent is allowed to recover from minor schema differences:
    - case_id can be used instead of review_id
    - review_title is optional and can be filled as empty text
    """
    required_columns = ["rating", "customer_comment"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    has_review_id = "review_id" in df.columns or "case_id" in df.columns
    if not has_review_id:
        missing_columns.append("review_id or case_id")

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
                "se dano",
                "dañó",
                "dano",
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
    """
    Classify all reviews and return a new dataframe with agent outputs.

    This function also performs light schema recovery:
    - If review_id is missing but case_id exists, case_id is copied into review_id.
    - If review_title is missing, it is filled with an empty string.
    """
    result_df = df.copy()

    if "review_id" not in result_df.columns and "case_id" in result_df.columns:
        result_df["review_id"] = result_df["case_id"]

    if "review_title" not in result_df.columns:
        result_df["review_title"] = ""

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

    if total == 0:
        return pd.DataFrame(columns=["category", "count", "percentage"])

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

    required_display_columns = [
        "review_id",
        "rating",
        "review_title",
        "customer_comment",
        "agent_evidence",
    ]

    for col in required_display_columns:
        if col not in classified_df.columns:
            classified_df[col] = ""

    for category in classified_df["agent_category"].unique():
        subset = classified_df[
            classified_df["agent_category"] == category
        ].head(max_examples_per_category)

        examples[category] = subset[required_display_columns].to_dict(
            orient="records"
        )

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

def infer_voc_theme(category):
    """Map agent category to a higher-level VOC theme."""
    theme_map = {
        "charging_failure": "Power & Charging",
        "charging_base_failure": "Power & Charging",
        "charging_pin_contact_issue": "Power & Charging",
        "battery_life_issue": "Power & Charging",
        "stopped_working": "Reliability & Durability",
        "grinding_power_issue": "Grinding Performance",
        "coarseness_adjustment_issue": "Grinding Performance",
        "salt_or_pepper_jamming": "Grinding Performance",
        "messy_dispensing": "Usability & Mess Control",
        "small_capacity_or_refill_issue": "Refill, Capacity & Cleaning",
        "material_or_build_quality_issue": "Build Quality & Perceived Value",
        "surface_stain_issue": "Appearance & Surface Finish",
        "positive_feedback": "Positive Experience",
        "customer_service_positive": "Customer Service",
        "unknown": "Unknown / Needs Review",
    }

    return theme_map.get(category, "Unknown / Needs Review")


def infer_voc_sub_issue(category, evidence):
    """Generate a more specific VOC sub-issue label."""
    evidence = clean_text(evidence)

    if category == "charging_failure":
        return "Direct charging failure / no recharge"
    if category == "charging_base_failure":
        return "Charging dock or base failure"
    if category == "charging_pin_contact_issue":
        return "Charging pin contact or spring tolerance issue"
    if category == "battery_life_issue":
        return "Weak battery retention or short usage time"
    if category == "stopped_working":
        return "Early-life product failure"
    if category == "grinding_power_issue":
        return "Weak grinding torque or slow grinding"
    if category == "coarseness_adjustment_issue":
        return "Inconsistent fine/coarse adjustment"
    if category == "salt_or_pepper_jamming":
        return "Salt or pepper jamming during use"
    if category == "messy_dispensing":
        return "Uncontrolled dispensing or residue mess"
    if category == "small_capacity_or_refill_issue":
        return "Small capacity, refill difficulty, or cleaning difficulty"
    if category == "material_or_build_quality_issue":
        return "Cheap, flimsy, or low-value material perception"
    if category == "surface_stain_issue":
        return "Surface staining or finish durability issue"
    if category == "customer_service_positive":
        return "Positive replacement or support experience"
    if category == "positive_feedback":
        return "Positive design, usability, or value feedback"

    if len(evidence) < 8:
        return "Insufficient evidence"

    return "Unclear issue requiring manual review"


def infer_sentiment(category, rating):
    """Infer review sentiment from category and rating."""
    try:
        rating_value = float(rating)
    except Exception:
        rating_value = None

    if category in ["positive_feedback", "customer_service_positive"]:
        return "positive"

    if category == "unknown":
        if rating_value is not None and rating_value >= 4:
            return "unclear_positive"
        if rating_value is not None and rating_value <= 2:
            return "unclear_negative"
        return "unknown"

    if rating_value is not None:
        if rating_value <= 2:
            return "strong_negative"
        if rating_value == 3:
            return "mixed_or_moderate_negative"

    return "negative"


def infer_severity(category, rating, confidence):
    """Infer severity level for VOC prioritisation."""
    critical_categories = {
        "charging_failure",
        "charging_base_failure",
        "charging_pin_contact_issue",
        "stopped_working",
    }

    high_categories = {
        "battery_life_issue",
        "grinding_power_issue",
        "coarseness_adjustment_issue",
        "salt_or_pepper_jamming",
    }

    try:
        rating_value = float(rating)
    except Exception:
        rating_value = None

    if category in ["positive_feedback", "customer_service_positive"]:
        return "low"

    if category == "unknown":
        return "review_needed"

    if category in critical_categories:
        if rating_value is not None and rating_value <= 2:
            return "critical"
        return "high"

    if category in high_categories:
        if rating_value is not None and rating_value <= 2:
            return "high"
        return "medium"

    if rating_value is not None and rating_value <= 2:
        return "high"

    if confidence < 0.6:
        return "review_needed"

    return "medium"


def infer_customer_pain_point(category):
    """Describe customer pain point in business language."""
    pain_point_map = {
        "charging_failure": "Customer cannot recharge the product reliably.",
        "charging_base_failure": "Customer loses confidence in the charging dock or base.",
        "charging_pin_contact_issue": "Customer must adjust the grinder to make charging contact.",
        "battery_life_issue": "Customer experiences short battery life or frequent recharging.",
        "stopped_working": "Customer experiences early product failure after limited use.",
        "grinding_power_issue": "Customer perceives the grinder as weak, slow, or ineffective.",
        "coarseness_adjustment_issue": "Customer cannot achieve the expected grind size.",
        "salt_or_pepper_jamming": "Customer has to shake, restart, or clear stuck ingredients.",
        "messy_dispensing": "Customer experiences uncontrolled salt/pepper output or mess.",
        "small_capacity_or_refill_issue": "Customer finds refill, capacity, or cleaning inconvenient.",
        "material_or_build_quality_issue": "Customer perceives the product as cheap or not worth the price.",
        "surface_stain_issue": "Customer sees visible stains or surface marks after normal use.",
        "positive_feedback": "Customer values the product design, convenience, and usability.",
        "customer_service_positive": "Customer values responsive support or replacement service.",
        "unknown": "Customer feedback is too vague or insufficient for reliable classification.",
    }

    return pain_point_map.get(category, "Customer issue requires manual review.")


def infer_root_cause_hypothesis(category):
    """Generate possible root-cause hypotheses for supplier discussion."""
    root_cause_map = {
        "charging_failure": "USB-C charging circuit, charging indicator, cable compatibility, or battery charging control may be unstable.",
        "charging_base_failure": "Dock alignment, charging base PCB, magnetic positioning, or dual-unit base reliability may be inconsistent.",
        "charging_pin_contact_issue": "Spring-loaded pins may have weak rebound, poor tolerance, contamination risk, or insufficient contact pressure.",
        "battery_life_issue": "Battery capacity, charging/discharging cycle quality, or standby drain may not meet expectation.",
        "stopped_working": "Motor, PCB, switch, wiring, or early-life assembly reliability may need stress testing.",
        "grinding_power_issue": "Motor torque, burr friction, ingredient load, or gear transmission may be insufficient.",
        "coarseness_adjustment_issue": "Burr adjustment tolerance, ceramic burr alignment, or calibration range may be inconsistent.",
        "salt_or_pepper_jamming": "Ingredient channel design, burr gap, or salt crystal compatibility may cause blockage.",
        "messy_dispensing": "Button sensitivity, output control, residue leakage, or grinder orientation may need redesign.",
        "small_capacity_or_refill_issue": "Container volume, refill opening, or cleaning access may be inconvenient.",
        "material_or_build_quality_issue": "Housing material, weight, finish, or perceived value may not match price expectation.",
        "surface_stain_issue": "Matte coating or surface treatment may lack oil and moisture resistance.",
        "positive_feedback": "Design, one-hand operation, appearance, and convenience are likely selling points.",
        "customer_service_positive": "Replacement and support process can reduce dissatisfaction risk.",
        "unknown": "Root cause cannot be inferred without clearer customer evidence.",
    }

    return root_cause_map.get(category, "Root cause requires manual investigation.")


def infer_business_impact(category, severity):
    """Estimate business impact for prioritisation."""
    if severity == "critical":
        return "High return risk and rating damage"
    if severity == "high":
        return "Likely negative review driver"
    if severity == "medium":
        return "Usability or satisfaction risk"
    if severity == "review_needed":
        return "Needs manual review before action"
    if category in ["positive_feedback", "customer_service_positive"]:
        return "Can be used as selling point or retention signal"

    return "Low immediate risk"


def add_voc_fields(classified_df: pd.DataFrame):
    """
    Add VOC-level analysis fields to classified review results.

    VOC fields include:
    - theme
    - sub-issue
    - sentiment
    - severity
    - customer pain point
    - root-cause hypothesis
    - business impact
    """
    voc_df = classified_df.copy()

    voc_df["voc_theme"] = voc_df["agent_category"].apply(infer_voc_theme)

    voc_df["voc_sub_issue"] = voc_df.apply(
        lambda row: infer_voc_sub_issue(
            row.get("agent_category", "unknown"),
            row.get("agent_evidence", ""),
        ),
        axis=1,
    )

    voc_df["voc_sentiment"] = voc_df.apply(
        lambda row: infer_sentiment(
            row.get("agent_category", "unknown"),
            row.get("rating", None),
        ),
        axis=1,
    )

    voc_df["voc_severity"] = voc_df.apply(
        lambda row: infer_severity(
            row.get("agent_category", "unknown"),
            row.get("rating", None),
            row.get("agent_confidence", 0),
        ),
        axis=1,
    )

    voc_df["customer_pain_point"] = voc_df["agent_category"].apply(
        infer_customer_pain_point
    )

    voc_df["root_cause_hypothesis"] = voc_df["agent_category"].apply(
        infer_root_cause_hypothesis
    )

    voc_df["business_impact"] = voc_df.apply(
        lambda row: infer_business_impact(
            row.get("agent_category", "unknown"),
            row.get("voc_severity", "review_needed"),
        ),
        axis=1,
    )

    return voc_df


def compute_voc_summary(voc_df: pd.DataFrame):
    """Compute VOC summary by theme, sub-issue, severity, and sentiment."""
    if voc_df is None or voc_df.empty:
        return {
            "theme_summary": pd.DataFrame(),
            "sub_issue_summary": pd.DataFrame(),
            "severity_summary": pd.DataFrame(),
            "sentiment_summary": pd.DataFrame(),
        }

    total = len(voc_df)

    theme_summary = (
        voc_df["voc_theme"]
        .value_counts()
        .reset_index()
    )
    theme_summary.columns = ["voc_theme", "count"]
    theme_summary["percentage"] = (theme_summary["count"] / total * 100).round(2)

    sub_issue_summary = (
        voc_df[["voc_theme", "voc_sub_issue"]]
        .value_counts()
        .reset_index(name="count")
    )
    sub_issue_summary["percentage"] = (
        sub_issue_summary["count"] / total * 100
    ).round(2)

    severity_summary = (
        voc_df["voc_severity"]
        .value_counts()
        .reset_index()
    )
    severity_summary.columns = ["voc_severity", "count"]
    severity_summary["percentage"] = (
        severity_summary["count"] / total * 100
    ).round(2)

    sentiment_summary = (
        voc_df["voc_sentiment"]
        .value_counts()
        .reset_index()
    )
    sentiment_summary.columns = ["voc_sentiment", "count"]
    sentiment_summary["percentage"] = (
        sentiment_summary["count"] / total * 100
    ).round(2)

    return {
        "theme_summary": theme_summary,
        "sub_issue_summary": sub_issue_summary,
        "severity_summary": severity_summary,
        "sentiment_summary": sentiment_summary,
    }
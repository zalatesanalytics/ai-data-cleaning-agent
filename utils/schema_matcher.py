# utils/schema_matcher.py
from collections import defaultdict
from rapidfuzz import fuzz

def normalize_col(col: str) -> str:
    if not isinstance(col, str):
        return ""
    return "".join(ch.lower() for ch in col if ch.isalnum())

def suggest_column_groups(list_of_column_lists, threshold: int = 80):
    """
    Input: list_of_column_lists = [cols_ds1, cols_ds2, ...]
    Output: list of groups, each group is a list of column names that look similar.
    """
    all_cols = []
    for cols in list_of_column_lists:
        all_cols.extend(cols)

    groups = []
    used = set()

    for col in all_cols:
        if col in used:
            continue
        group = [col]
        used.add(col)
        for other in all_cols:
            if other in used:
                continue
            score = fuzz.ratio(normalize_col(col), normalize_col(other))
            if score >= threshold:
                group.append(other)
                used.add(other)
        groups.append(group)

    return groups

def build_canonical_mapping(column_groups):
    """
    For each group of similar columns, pick a canonical name (first one by default).
    Returns: dict original_col -> canonical_col
    """
    mapping = {}
    for group in column_groups:
        canonical = group[0]
        for col in group:
            mapping[col] = canonical
    return mapping

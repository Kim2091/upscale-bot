import re
from difflib import SequenceMatcher

def normalize_model_name(name):
    """Normalize model name for better matching."""
    return re.sub(r'[^a-z0-9]', '', name.lower())

def calculate_similarity(str1, str2):
    """Calculate similarity between two strings."""
    return SequenceMatcher(None, str1, str2).ratio()

def find_closest_models(query, available_models, max_results=5, threshold=60):
    """
    Find the closest matching models using specialized fuzzy string matching.
    
    Args:
    query (str): The search query.
    available_models (list): List of available model names.
    max_results (int): Maximum number of results to return.
    threshold (int): Minimum similarity score to consider a match.
    
    Returns:
    list: List of tuples containing (model_name, similarity_score, match_type).
    """
    normalized_query = normalize_model_name(query)
    matches = []

    for model in available_models:
        normalized_model = normalize_model_name(model)
        
        if normalized_query == normalized_model:
            matches.append((model, 100, "exact"))
        elif normalized_query in normalized_model:
            # Boost substring matches significantly
            score = 95 + (5 * len(normalized_query) / len(normalized_model))
            matches.append((model, round(score), "substring"))
        else:
            similarity = calculate_similarity(normalized_query, normalized_model)
            score = similarity * 100
            if score >= threshold:
                matches.append((model, round(score), "partial"))
    
    # Sort matches by score in descending order
    sorted_matches = sorted(matches, key=lambda x: (-x[1], len(x[0])))
    
    return sorted_matches[:max_results]

def search_models(search_term, available_models, max_results=5, threshold=60):
    """
    Search for models with names similar to the search term.
    
    Args:
    search_term (str): The term to search for.
    available_models (list): List of available model names.
    max_results (int): Maximum number of results to return.
    threshold (int): Minimum similarity score to consider a match.
    
    Returns:
    list: List of tuples containing (model_name, similarity_score, match_type).
    """
    return find_closest_models(search_term, available_models, max_results, threshold)
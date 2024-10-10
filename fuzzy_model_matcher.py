from fuzzywuzzy import process

def find_closest_model(model_name, available_models, threshold=80):
    """
    Find the closest matching models using fuzzy string matching.
    
    Args:
    model_name (str): The name to match against.
    available_models (list): List of available model names.
    threshold (int): Minimum similarity score to consider a match.
    
    Returns:
    list: List of tuples containing (model_name, similarity_score).
    """
    matches = process.extractBests(model_name, available_models, score_cutoff=threshold, limit=3)
    return matches

def search_models(search_term, available_models, threshold=60):
    """
    Search for models with names similar to the search term.
    
    Args:
    search_term (str): The term to search for.
    available_models (list): List of available model names.
    threshold (int): Minimum similarity score to consider a match.
    
    Returns:
    list: List of tuples containing (model_name, similarity_score).
    """
    return find_closest_model(search_term, available_models, threshold)

from fuzzy_model_matcher import search_models
import logging
logger = logging.getLogger('UpscaleBot')

available_models = [
    "4xFireAlpha", "4xForest", "4xFuzzyBox", "4xGround", "4xGuilty", 
    "4xJaypeg90", "4xLADDIER1_282500_G", "4xLSDIRDAT", "4xLady0101", 
    "4xMCWashed", "4xManga109", "4xMeguUp130k", "4xMinecraftAlpha", 
    "4xMinecraftSPSR_60000_G", "1x-Focus", "1x-Focus_Moderate", 
    "1x-GameSmooth_SuperUltraCompact", "1x-PBRify_Height", 
    "1x-PBRify_NormalV3", "1x-PBRify_RoughnessV2", "1xBC1Smooth2", 
    "1xBSVectrix", "1xBSVectrix24000G", "1xBS_Debandizer", 
    "1xBS_DebandizerSPSR", "1xBaldrickVHSFix_180000_G_V0.2",
    "1x_Sayajin_DeJpeg", "DeJpeg", "DeJPEG"
]

# Test cases
test_queries = ["dejpg", "smooth", "minecraft", "4x", "pbrify", "dejpeg", "DeJpeg"]

for query in test_queries:
    logger.info(f"\nSearching for: {query}")
    results = search_models(query, available_models)
    for model, score, match_type in results:
        logger.info(f"  {model} (score: {score}, type: {match_type})")

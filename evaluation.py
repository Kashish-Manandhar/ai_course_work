from typing import Dict, List, Tuple
import numpy as np
from surprise import accuracy
from surprise import SVD
from collections import defaultdict
import math

def evaluate_model(algo: SVD, testset: List[Tuple], k: int = 10, threshold: float = 4.0) -> Dict[str, float]:
    """Enhanced evaluation with additional metrics and error handling"""
    
    predictions = algo.test(testset)
    
    # Basic accuracy metrics
    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)
    
    # Advanced metrics
    p, r, f1 = precision_recall_f1_at_k_improved(predictions, k=k, threshold=threshold)
    ndcg = ndcg_at_k_improved(predictions, k=k, threshold=threshold)
    
    # Additional metrics
    coverage = calculate_coverage(predictions, testset)
    diversity = calculate_diversity(predictions, k=k)
    
    metrics = {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4),
        "ndcg": round(ndcg, 4),
        "coverage": round(coverage, 4),
        "diversity": round(diversity, 4)
    }
    
    print("✅ Enhanced evaluation metrics:", metrics)
    return metrics

def precision_recall_f1_at_k_improved(predictions: List[Tuple], k: int = 10, threshold: float = 4.0) -> Tuple[float, float, float]:
    """Improved precision/recall calculation with better handling of edge cases"""
    
    user_predictions = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        user_predictions[uid].append((iid, est, true_r))

    precisions, recalls = [], []
    
    for uid, user_preds in user_predictions.items():
        if len(user_preds) < k:
            continue
            
        # Sort by estimated rating
        user_preds.sort(key=lambda x: x[1], reverse=True)
        
        # Get top k recommendations
        top_k_items = user_preds[:k]
        top_k_ids = {iid for iid, _, _ in top_k_items}
        
        # Get relevant items (actual rating >= threshold)
        relevant_items = {iid for iid, _, true_r in user_preds if true_r >= threshold}
        
        if not top_k_ids:
            continue
            
        # Calculate precision and recall
        true_positives = len(top_k_ids & relevant_items)
        precision = true_positives / k
        recall = true_positives / len(relevant_items) if relevant_items else 0.0
        
        precisions.append(precision)
        recalls.append(recall)

    if not precisions:
        return 0.0, 0.0, 0.0
        
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    f1 = (2 * avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0.0
    
    return avg_precision, avg_recall, f1

def ndcg_at_k_improved(predictions: List[Tuple], k: int = 10, threshold: float = 4.0) -> float:
    """Enhanced NDCG calculation with better normalization"""
    
    def dcg_at_k(r, k):
        """Calculate DCG@k for a list of relevances"""
        r = np.asfarray(r)[:k]
        if r.size:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        return 0.0

    user_predictions = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        user_predictions[uid].append((iid, est, true_r))

    ndcg_scores = []
    
    for uid, user_preds in user_predictions.items():
        if len(user_preds) < k:
            continue
            
        # Sort by estimated rating
        user_preds.sort(key=lambda x: x[1], reverse=True)
        
        # Get relevance scores for top k
        top_k_relevance = []
        all_relevance = []
        
        for i, (iid, est, true_r) in enumerate(user_preds):
            relevance = 1 if true_r >= threshold else 0
            all_relevance.append(relevance)
            if i < k:
                top_k_relevance.append(relevance)
        
        # Calculate DCG and IDCG
        dcg = dcg_at_k(top_k_relevance, k)
        ideal_relevance = sorted(all_relevance, reverse=True)
        idcg = dcg_at_k(ideal_relevance, k)
        
        if idcg > 0:
            ndcg_scores.append(dcg / idcg)

    return np.mean(ndcg_scores) if ndcg_scores else 0.0

def calculate_coverage(predictions: List[Tuple], testset: List[Tuple]) -> float:
    """Calculate item coverage - fraction of items that can be recommended"""
    # Fix: Correctly access item IDs from the tuple structure
    recommended_items = {pred[1] for pred in predictions if pred[3] >= 3.0}
    total_items = {test[1] for test in testset}
    
    if not total_items:
        return 0.0
        
    return len(recommended_items) / len(total_items)

def calculate_diversity(predictions: List[Tuple], k: int = 10) -> float:
    """Calculate diversity of recommendations across users"""
    user_recommendations = defaultdict(list)
    
    # Fix: Correctly access estimated rating (est) from the tuple
    for uid, iid, _, est, _ in predictions:
        if est >= 3.0:
            user_recommendations[uid].append(iid)
    
    if not user_recommendations:
        return 0.0
    
    # Calculate pairwise diversity
    all_recommendations = []
    for uid, recs in user_recommendations.items():
        top_recs = recs[:k]
        all_recommendations.extend(top_recs)
    
    unique_recommendations = len(set(all_recommendations))
    total_recommendations = len(all_recommendations)
    
    return unique_recommendations / total_recommendations if total_recommendations > 0 else 0.0
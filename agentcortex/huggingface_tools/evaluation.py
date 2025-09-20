"""Model evaluation utilities using Hugging Face evaluate library."""

import logging
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np

try:
    import evaluate
    EVALUATE_AVAILABLE = True
except ImportError:
    EVALUATE_AVAILABLE = False

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Model evaluation using Hugging Face evaluate library.
    
    Supports various metrics for different tasks.
    """
    
    def __init__(self):
        """Initialize model evaluator."""
        if not EVALUATE_AVAILABLE:
            logger.warning("Hugging Face evaluate library not available")
        
        self.loaded_metrics = {}
        logger.info("Model evaluator initialized")
    
    def load_metric(self, metric_name: str, **kwargs) -> Any:
        """
        Load a metric from Hugging Face evaluate.
        
        Args:
            metric_name: Name of the metric to load
            **kwargs: Additional arguments for metric loading
            
        Returns:
            Loaded metric object
        """
        if not EVALUATE_AVAILABLE:
            raise ImportError("Hugging Face evaluate library not available")
        
        if metric_name not in self.loaded_metrics:
            try:
                self.loaded_metrics[metric_name] = evaluate.load(metric_name, **kwargs)
                logger.info(f"Loaded metric: {metric_name}")
            except Exception as e:
                logger.error(f"Failed to load metric {metric_name}: {e}")
                raise
        
        return self.loaded_metrics[metric_name]
    
    def evaluate_classification(
        self,
        predictions: List[Union[int, str]],
        references: List[Union[int, str]],
        metrics: Optional[List[str]] = None,
        average: str = "weighted"
    ) -> Dict[str, float]:
        """
        Evaluate classification predictions.
        
        Args:
            predictions: Predicted labels
            references: True labels
            metrics: List of metrics to compute
            average: Averaging strategy for multi-class metrics
            
        Returns:
            Dictionary of metric scores
        """
        if metrics is None:
            metrics = ["accuracy", "precision", "recall", "f1"]
        
        results = {}
        
        for metric_name in metrics:
            try:
                metric = self.load_metric(metric_name)
                
                if metric_name in ["precision", "recall", "f1"]:
                    score = metric.compute(
                        predictions=predictions,
                        references=references,
                        average=average
                    )
                else:
                    score = metric.compute(
                        predictions=predictions,
                        references=references
                    )
                
                # Handle different return formats
                if isinstance(score, dict):
                    for key, value in score.items():
                        results[f"{metric_name}_{key}"] = value
                else:
                    results[metric_name] = score
                    
            except Exception as e:
                logger.warning(f"Failed to compute {metric_name}: {e}")
                continue
        
        return results
    
    def evaluate_text_generation(
        self,
        predictions: List[str],
        references: List[List[str]],  # Multiple references possible
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Evaluate text generation predictions.
        
        Args:
            predictions: Generated texts
            references: Reference texts (list of lists for multiple references)
            metrics: List of metrics to compute
            
        Returns:
            Dictionary of metric scores
        """
        if metrics is None:
            metrics = ["bleu", "rouge", "meteor"]
        
        results = {}
        
        for metric_name in metrics:
            try:
                metric = self.load_metric(metric_name)
                
                if metric_name == "bleu":
                    score = metric.compute(
                        predictions=predictions,
                        references=references
                    )
                elif metric_name == "rouge":
                    # Convert to single reference format for rouge
                    single_refs = [refs[0] if refs else "" for refs in references]
                    score = metric.compute(
                        predictions=predictions,
                        references=single_refs
                    )
                elif metric_name == "meteor":
                    # METEOR expects single references
                    single_refs = [refs[0] if refs else "" for refs in references]
                    score = metric.compute(
                        predictions=predictions,
                        references=single_refs
                    )
                else:
                    score = metric.compute(
                        predictions=predictions,
                        references=references
                    )
                
                # Handle different return formats
                if isinstance(score, dict):
                    for key, value in score.items():
                        results[f"{metric_name}_{key}"] = value
                else:
                    results[metric_name] = score
                    
            except Exception as e:
                logger.warning(f"Failed to compute {metric_name}: {e}")
                continue
        
        return results
    
    def evaluate_retrieval(
        self,
        retrieved_docs: List[List[str]],  # For each query
        relevant_docs: List[List[str]],   # Relevant docs for each query
        k_values: Optional[List[int]] = None
    ) -> Dict[str, float]:
        """
        Evaluate retrieval performance.
        
        Args:
            retrieved_docs: Retrieved documents for each query
            relevant_docs: Relevant documents for each query
            k_values: Values of k for precision@k and recall@k
            
        Returns:
            Dictionary of metric scores
        """
        if k_values is None:
            k_values = [1, 3, 5, 10]
        
        results = {}
        
        # Compute precision@k and recall@k
        for k in k_values:
            precisions = []
            recalls = []
            
            for retrieved, relevant in zip(retrieved_docs, relevant_docs):
                # Take top-k retrieved docs
                top_k_retrieved = retrieved[:k]
                
                # Compute precision@k
                if top_k_retrieved:
                    relevant_in_topk = len(set(top_k_retrieved) & set(relevant))
                    precision_k = relevant_in_topk / len(top_k_retrieved)
                else:
                    precision_k = 0.0
                
                # Compute recall@k
                if relevant:
                    recall_k = len(set(top_k_retrieved) & set(relevant)) / len(relevant)
                else:
                    recall_k = 0.0
                
                precisions.append(precision_k)
                recalls.append(recall_k)
            
            results[f"precision_at_{k}"] = np.mean(precisions)
            results[f"recall_at_{k}"] = np.mean(recalls)
        
        # Compute Mean Average Precision (MAP)
        map_scores = []
        for retrieved, relevant in zip(retrieved_docs, relevant_docs):
            if not relevant:
                continue
            
            ap_score = self._compute_average_precision(retrieved, relevant)
            map_scores.append(ap_score)
        
        results["map"] = np.mean(map_scores) if map_scores else 0.0
        
        # Compute MRR (Mean Reciprocal Rank)
        mrr_scores = []
        for retrieved, relevant in zip(retrieved_docs, relevant_docs):
            rr = self._compute_reciprocal_rank(retrieved, relevant)
            mrr_scores.append(rr)
        
        results["mrr"] = np.mean(mrr_scores)
        
        return results
    
    def _compute_average_precision(
        self,
        retrieved: List[str],
        relevant: List[str]
    ) -> float:
        """Compute Average Precision for a single query."""
        if not relevant:
            return 0.0
        
        relevant_set = set(relevant)
        precision_sum = 0.0
        relevant_count = 0
        
        for i, doc in enumerate(retrieved):
            if doc in relevant_set:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                precision_sum += precision_at_i
        
        return precision_sum / len(relevant) if relevant else 0.0
    
    def _compute_reciprocal_rank(
        self,
        retrieved: List[str],
        relevant: List[str]
    ) -> float:
        """Compute Reciprocal Rank for a single query."""
        relevant_set = set(relevant)
        
        for i, doc in enumerate(retrieved):
            if doc in relevant_set:
                return 1.0 / (i + 1)
        
        return 0.0
    
    def evaluate_rag_pipeline(
        self,
        questions: List[str],
        generated_answers: List[str],
        reference_answers: List[str],
        retrieved_contexts: List[List[str]],
        relevant_contexts: List[List[str]],
        include_retrieval_metrics: bool = True,
        include_generation_metrics: bool = True
    ) -> Dict[str, float]:
        """
        Comprehensive evaluation of RAG pipeline.
        
        Args:
            questions: Input questions
            generated_answers: Generated answers
            reference_answers: Ground truth answers
            retrieved_contexts: Retrieved context documents
            relevant_contexts: Relevant context documents
            include_retrieval_metrics: Whether to include retrieval evaluation
            include_generation_metrics: Whether to include generation evaluation
            
        Returns:
            Dictionary of all metric scores
        """
        results = {}
        
        # Evaluate retrieval component
        if include_retrieval_metrics and retrieved_contexts and relevant_contexts:
            retrieval_results = self.evaluate_retrieval(
                retrieved_contexts,
                relevant_contexts
            )
            
            for key, value in retrieval_results.items():
                results[f"retrieval_{key}"] = value
        
        # Evaluate generation component
        if include_generation_metrics:
            # Convert single references to list format for generation metrics
            reference_lists = [[ref] for ref in reference_answers]
            
            generation_results = self.evaluate_text_generation(
                generated_answers,
                reference_lists
            )
            
            for key, value in generation_results.items():
                results[f"generation_{key}"] = value
        
        # Additional RAG-specific metrics
        if len(generated_answers) == len(reference_answers):
            # Answer relevance (simple keyword overlap)
            relevance_scores = []
            for gen, ref in zip(generated_answers, reference_answers):
                relevance = self._compute_answer_relevance(gen, ref)
                relevance_scores.append(relevance)
            
            results["answer_relevance"] = np.mean(relevance_scores)
        
        return results
    
    def _compute_answer_relevance(self, generated: str, reference: str) -> float:
        """Compute simple answer relevance based on keyword overlap."""
        gen_words = set(generated.lower().split())
        ref_words = set(reference.lower().split())
        
        if not ref_words:
            return 0.0
        
        overlap = len(gen_words & ref_words)
        return overlap / len(ref_words)
    
    def compare_models(
        self,
        model_results: Dict[str, Dict[str, float]],
        primary_metric: str = "accuracy"
    ) -> Dict[str, Any]:
        """
        Compare multiple models' performance.
        
        Args:
            model_results: Dictionary mapping model names to their results
            primary_metric: Primary metric for ranking
            
        Returns:
            Comparison results with rankings
        """
        if not model_results:
            return {}
        
        # Get all metrics
        all_metrics = set()
        for results in model_results.values():
            all_metrics.update(results.keys())
        
        comparison = {
            "models": list(model_results.keys()),
            "metrics": list(all_metrics),
            "scores": model_results,
            "rankings": {}
        }
        
        # Rank models by each metric
        for metric in all_metrics:
            metric_scores = []
            for model_name in model_results.keys():
                score = model_results[model_name].get(metric, 0.0)
                metric_scores.append((model_name, score))
            
            # Sort by score (descending)
            metric_scores.sort(key=lambda x: x[1], reverse=True)
            comparison["rankings"][metric] = metric_scores
        
        # Overall ranking by primary metric
        if primary_metric in all_metrics:
            comparison["overall_ranking"] = comparison["rankings"][primary_metric]
        
        return comparison
    
    def get_available_metrics(self) -> List[str]:
        """Get list of available metrics."""
        if not EVALUATE_AVAILABLE:
            return []
        
        try:
            return evaluate.list_evaluation_modules()
        except Exception as e:
            logger.warning(f"Could not list available metrics: {e}")
            return []
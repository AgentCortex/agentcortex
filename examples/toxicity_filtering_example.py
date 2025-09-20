#!/usr/bin/env python3
"""
Toxicity Filtering Example

This example demonstrates how to:
1. Detect toxic content in text
2. Filter datasets for toxic content
3. Clean toxic text with different strategies
4. Analyze toxicity statistics
"""

import logging
import sys
from pathlib import Path

# Add the parent directory to the path so we can import agentcortex
sys.path.insert(0, str(Path(__file__).parent.parent))

from agentcortex.toxicity_filter import ToxicityFilter
from agentcortex.utils import setup_logging

# Set up logging
setup_logging(level="INFO")
logger = logging.getLogger(__name__)


def main():
    """Run the toxicity filtering example."""
    logger.info("Starting Toxicity Filtering Example")
    
    # Sample texts with varying levels of toxicity
    sample_texts = [
        "I love machine learning and AI technology!",
        "This is a great tutorial on natural language processing.",
        "The weather is beautiful today.",
        "This is a neutral statement about technology.",
        "I disagree with this approach, but respect your opinion.",
        "This documentation is confusing and poorly written.",
        "HELP ME WITH THIS URGENT PROBLEM!!!",  # Excessive caps
        "aaaahhhhhhhhh noooooooo",  # Repeated characters
        "Your approach is completely wrong and stupid.",  # Mild toxicity
        "Machine learning models can sometimes produce biased results.",
    ]
    
    # Step 1: Initialize toxicity filter
    logger.info("Initializing toxicity filter...")
    toxicity_filter = ToxicityFilter(
        toxicity_threshold=0.7,
        use_rule_based=True
    )
    
    # Step 2: Detect toxicity in individual texts
    logger.info("\nAnalyzing individual texts for toxicity...")
    
    for i, text in enumerate(sample_texts, 1):
        logger.info(f"\nText {i}: {text}")
        
        # Get detailed toxicity analysis
        analysis = toxicity_filter.detect_toxicity(text, return_scores=True)
        
        logger.info(f"  Is toxic: {analysis['is_toxic']}")
        logger.info(f"  Toxicity score: {analysis['toxicity']:.3f}")
        
        if analysis.get('rule_based_flags'):
            logger.info(f"  Rule-based flags: {analysis['rule_based_flags']}")
        
        # Show other toxicity dimensions if available
        for dimension in ['severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack']:
            if dimension in analysis and analysis[dimension] > 0.1:
                logger.info(f"  {dimension}: {analysis[dimension]:.3f}")
    
    # Step 3: Batch detection for efficiency
    logger.info("\nPerforming batch toxicity detection...")
    
    batch_results = toxicity_filter.batch_detect(
        sample_texts,
        batch_size=4,
        show_progress=True
    )
    
    # Count toxic vs non-toxic
    toxic_count = sum(1 for result in batch_results if result['is_toxic'])
    logger.info(f"Batch results: {toxic_count}/{len(sample_texts)} texts flagged as toxic")
    
    # Step 4: Filter dataset
    logger.info("\nFiltering dataset...")
    
    # Remove toxic content
    filtered_texts, analysis_results = toxicity_filter.filter_dataset(
        sample_texts,
        remove_toxic=True,
        clean_toxic=False,
        show_progress=True
    )
    
    logger.info(f"Original texts: {len(sample_texts)}")
    logger.info(f"Filtered texts: {len(filtered_texts)}")
    logger.info(f"Removed: {len(sample_texts) - len(filtered_texts)} toxic texts")
    
    # Step 5: Clean toxic content instead of removing
    logger.info("\nCleaning toxic content...")
    
    cleaned_texts, _ = toxicity_filter.filter_dataset(
        sample_texts,
        remove_toxic=False,
        clean_toxic=True,
        show_progress=True
    )
    
    logger.info("Comparing original vs cleaned texts:")
    for original, cleaned in zip(sample_texts, cleaned_texts):
        if original != cleaned:
            logger.info(f"  Original: {original}")
            logger.info(f"  Cleaned:  {cleaned}")
    
    # Step 6: Test different cleaning strategies
    logger.info("\nTesting different cleaning strategies...")
    
    test_text = "Your approach is completely wrong and stupid."
    
    strategies = ["remove", "replace", "mask"]
    for strategy in strategies:
        cleaned = toxicity_filter.clean_text(
            test_text,
            replacement_strategy=strategy,
            replacement_text="[INAPPROPRIATE]"
        )
        logger.info(f"  {strategy}: {cleaned}")
    
    # Step 7: Get statistics
    logger.info("\nToxicity statistics:")
    
    stats = toxicity_filter.get_statistics(analysis_results)
    
    logger.info(f"  Total texts analyzed: {stats['total_texts']}")
    logger.info(f"  Toxic texts: {stats['toxic_texts']} ({stats['toxic_percentage']:.1f}%)")
    logger.info(f"  Clean texts: {stats['clean_texts']}")
    
    if stats.get('average_scores'):
        logger.info("  Average toxicity scores:")
        for score_type, value in stats['average_scores'].items():
            logger.info(f"    {score_type}: {value:.3f}")
    
    if stats.get('rule_based_flags'):
        logger.info("  Rule-based flags detected:")
        for flag_type, count in stats['rule_based_flags'].items():
            logger.info(f"    {flag_type}: {count}")
    
    # Step 8: Update filters with custom rules
    logger.info("\nUpdating filters with custom rules...")
    
    # Add custom explicit words
    custom_words = ["badword1", "inappropriate_term"]
    toxicity_filter.update_filters(new_explicit_words=custom_words)
    
    # Add custom patterns
    custom_patterns = [
        r'\b(completely\s+wrong)\b',  # Flag "completely wrong"
        r'\b(very\s+bad)\b'           # Flag "very bad"
    ]
    toxicity_filter.update_filters(new_patterns=custom_patterns)
    
    # Test updated filters
    test_texts = [
        "This is completely wrong and inappropriate_term.",
        "That's a very bad approach to the problem.",
        "This is a normal sentence."
    ]
    
    logger.info("Testing updated filters:")
    for text in test_texts:
        is_toxic = toxicity_filter.detect_toxicity(text)
        logger.info(f"  '{text}' -> Toxic: {is_toxic}")
    
    # Step 9: Export configuration
    logger.info("\nExporting filter configuration...")
    
    config = toxicity_filter.export_config()
    logger.info("Current filter configuration:")
    for key, value in config.items():
        if key == 'explicit_words':
            logger.info(f"  {key}: {len(value)} words")
        elif key == 'patterns':
            logger.info(f"  {key}: {len(value)} patterns")
        else:
            logger.info(f"  {key}: {value}")
    
    # Step 10: Demonstrate real-world usage scenarios
    logger.info("\nReal-world usage scenarios...")
    
    # Scenario 1: Content moderation for user comments
    user_comments = [
        "Great article, very informative!",
        "This is completely wrong and the author is stupid.",
        "I have a different perspective on this topic.",
        "HELP HELP HELP URGENT PROBLEM!!!",
        "Thanks for sharing this useful information."
    ]
    
    logger.info("Content moderation for user comments:")
    moderated_comments = []
    
    for comment in user_comments:
        analysis = toxicity_filter.detect_toxicity(comment, return_scores=True)
        
        if analysis['is_toxic']:
            if analysis['toxicity'] > 0.8:
                # High toxicity - remove completely
                action = "REMOVED"
                final_comment = "[Comment removed for policy violation]"
            else:
                # Moderate toxicity - clean
                action = "CLEANED"
                final_comment = toxicity_filter.clean_text(comment, replacement_strategy="replace")
        else:
            action = "APPROVED"
            final_comment = comment
        
        moderated_comments.append(final_comment)
        logger.info(f"  {action}: {comment[:50]}{'...' if len(comment) > 50 else ''}")
    
    # Scenario 2: Training data cleaning
    logger.info("\nTraining data cleaning scenario:")
    
    training_data = sample_texts + user_comments
    clean_training_data, training_analysis = toxicity_filter.filter_dataset(
        training_data,
        remove_toxic=True,
        show_progress=False
    )
    
    logger.info(f"  Original training data: {len(training_data)} samples")
    logger.info(f"  Clean training data: {len(clean_training_data)} samples")
    logger.info(f"  Filtered out: {len(training_data) - len(clean_training_data)} toxic samples")
    
    logger.info("Toxicity Filtering Example completed successfully!")


if __name__ == "__main__":
    main()
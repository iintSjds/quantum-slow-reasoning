"""
Add JSON logging functionality to QuCoNet demo scripts.

This script provides functions to save training results in JSON format
for easy extraction and comparison with CoNet results.
"""

import json
import os
from datetime import datetime
from pathlib import Path


def save_training_results_json(results_dict, output_path, filename=None):
    """
    Save training results to JSON file for easy comparison.
    
    Args:
        results_dict: Dictionary containing training results
        output_path: Directory to save the file
        filename: Optional custom filename (auto-generated if None)
    
    Returns:
        Path to saved JSON file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Generate filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"training_results_{timestamp}.json"
    
    filepath = os.path.join(output_path, filename)
    
    # Save results
    with open(filepath, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"Saved training results to: {filepath}")
    return filepath


def extract_final_metrics_from_trainer(trainer):
    """
    Extract final metrics from QuCoNet trainer.
    
    Args:
        trainer: QuCoNetTrainer instance
    
    Returns:
        Dictionary with final metrics
    """
    metrics = {
        'num_epochs': trainer.current_epoch + 1,
        'best_success_rate': 0.0,
        'best_epoch': 0,
        'final_success_rate': 0.0,
        'final_loss': 0.0,
    }
    
    # Get best metrics
    if hasattr(trainer, 'best_metrics'):
        metrics['best_success_rate'] = trainer.best_metrics.get('success_rate', 0.0)
        metrics['best_epoch'] = trainer.best_metrics.get('epoch', 0)
        metrics['best_avg_success_rate'] = trainer.best_metrics.get('average_success_rate', 0.0)
    
    # Get final metrics from training history
    if hasattr(trainer, 'training_history') and trainer.training_history:
        final_entry = trainer.training_history[-1]
        metrics['final_success_rate'] = final_entry.get('success_rate', 0.0)
        metrics['final_avg_success_rate'] = final_entry.get('average_success_rate', 0.0)
        metrics['final_loss'] = final_entry.get('loss', 0.0)
        
        # Store full training history for detailed analysis
        metrics['training_history'] = trainer.training_history
    
    return metrics


def create_comparison_results_dict(trainer, model_config, qa_pairs, start_time=None, end_time=None):
    """
    Create a comprehensive results dictionary for comparison.
    
    Args:
        trainer: QuCoNetTrainer instance
        model_config: Dictionary with model configuration
        qa_pairs: List of QA pairs used
        start_time: Training start time
        end_time: Training end time
    
    Returns:
        Dictionary with all results
    """
    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'model_type': 'QuCoNet',
            'framework': 'PyTorch'
        },
        'model_config': model_config,
        'qa_pairs': [{'start': q, 'target': a} for q, a in qa_pairs],
        'num_qa_pairs': len(qa_pairs),
    }
    
    # Add timing info if available
    if start_time and end_time:
        results['metadata']['training_duration'] = end_time - start_time
    
    # Add metrics
    results['metrics'] = extract_final_metrics_from_trainer(trainer)
    
    return results


# Example usage in demo scripts:
if __name__ == "__main__":
    # This shows how to use the logging functions in demo scripts
    print("Example usage:")
    print("""
    # At the end of training in real_world_demo_ar.py or rl_sft_loop_demo_ar.py:
    
    from quconet_logging import create_comparison_results_dict, save_training_results_json
    
    # Create results dictionary
    model_config = {
        'N': N,
        'K': K,
        'M': M,
        'batch_size': B,
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
        'loss_type': loss_type
    }
    
    results_dict = create_comparison_results_dict(
        trainer=trainer,
        model_config=model_config,
        qa_pairs=qa_pairs,
        start_time=start_time,
        end_time=end_time
    )
    
    # Save to JSON
    json_path = save_training_results_json(
        results_dict=results_dict,
        output_path='.',
        filename=f'{filename}_results.json'
    )
    """)

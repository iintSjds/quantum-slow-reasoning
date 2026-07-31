"""
QuCoNet Trainer - High-level training interface for quantum walk models

This module provides a comprehensive training framework that handles:
- Configuration management using Hydra and OmegaConf
- Model initialization with various strategies
- Training loop orchestration
- Results tracking and analysis
- Model checkpointing and resuming
"""

import torch
import torch.nn as nn
import json
import os
import time
import math
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from .modules import QuantumCoNet
from .quconet_ar import QuantumCoNetAR


# Configuration structure using OmegaConf for Hydra integration
# The actual configuration will be loaded from YAML files using Hydra

DEFAULT_CONFIG = {
    "experiment_name": "quconet_experiment",
    "output_dir": "results",
    "seed": None,
    "device": "auto",  # auto, cpu, cuda
    "log_level": "INFO",

    "model": {
        "N": 10,
        "K": 2,
        "max_steps": 10,
        "use_ar": False,  # Use AR (autoregressive) mode by default
        "init_strategy": "uniform",
        "loss_type": "standard",
        "device": "auto"
    },

    "training": {
        "num_epochs": 100,
        "batch_size": 16,
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "gradient_clip": None,
        "eval_frequency": 10,
        "save_frequency": 50,
        "log_frequency": 1,
        "early_stopping": {
            "enabled": False,
            "patience": 20,
            "min_delta": 0.001,
            "monitor": "success_rate"
        },
        "model_selection": {
            "metric": "success_rate",
            "mode": "max"
        },
        "dropout": 0.0,
        "mixed_precision": False,
        "distributed": False,
        "save_checkpoints": True
    },

    "data": {
        "graph_type": "synthetic",
        "N": 10,  # Will be set to model.N in actual usage
        "K": 2,   # Will be set to model.K in actual usage
        "graph_seed": None,
        "qa_pairs": None,
        "num_qa_pairs": 20,
        "max_distance": 3,
        "qa_generation_strategy": "uniform",
        "validate_graph": True,
        "ensure_connected": True,
        "augmentation": {
            "enabled": False,
            "flip_probability": 0.1,
            "noise_std": 0.0
        }
    },

    "optimizer": {
        "lr": 0.01,  # Will be set to training.learning_rate in actual usage
        "weight_decay": 0.0  # Will be set to training.weight_decay in actual usage
    },

    "scheduler": {
        "_target_": None
    }
}


class QuCoNetTrainer:
    """
    High-level trainer for QuCoNet quantum walk models using Hydra configuration

    This class provides a comprehensive training interface that handles:
    - Configuration management using Hydra and OmegaConf
    - Model initialization with various strategies
    - Training loop orchestration with progress tracking
    - Results collection and analysis
    - Model checkpointing and resuming
    - Multiple evaluation metrics
    """

    def __init__(self, cfg: Optional[DictConfig] = None, config_path: Optional[str] = None):
        """
        Initialize the trainer with Hydra configuration

        Args:
            cfg: OmegaConf DictConfig from Hydra
            config_path: Path to config directory for manual initialization
        """
        if cfg is not None:
            self.cfg = cfg
        elif config_path is not None:
            # Manual initialization with config directory
            self.cfg = self._load_config_from_dir(config_path)
        else:
            # Use default config
            self.cfg = OmegaConf.create(DEFAULT_CONFIG)

        # Resolve interpolations in config
        OmegaConf.resolve(self.cfg)

        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.device = None
        self.current_epoch = 0
        self.best_metrics = {}
        self.training_history = []
        self.qa_pairs = None
        self.adjacency_list = None

        # Setup training environment
        self._setup_logging()
        self._setup_seed()
        self._setup_device()
        self._setup_output_dirs()

    def _load_config_from_dir(self, config_dir: str) -> DictConfig:
        """Load configuration from Hydra config directory"""
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name="config")
        return cfg

    def _setup_seed(self):
        """Setup random seed for reproducibility"""
        seed = getattr(self.cfg, 'seed', None)
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            print(f"Set random seed: {seed}")

    def _setup_logging(self):
        """Setup logging configuration"""
        import logging
        # Get log level with fallback to INFO if not specified
        log_level = getattr(self.cfg, 'log_level', 'INFO')
        logging.basicConfig(level=getattr(logging, log_level.upper()))
        self.logger = logging.getLogger(__name__)

    def _setup_device(self):
        """Setup computation device"""
        device = getattr(self.cfg, 'device', 'auto')
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.logger.info(f"Using device: {self.device}")

    def _setup_model(self):
        """Initialize the quantum walk model"""
        if self.adjacency_list is None:
            raise ValueError("Adjacency list must be set before initializing model")

        # Determine model type based on use_ar flag
        use_ar = getattr(self.cfg.model, 'use_ar', False)
        model_class = QuantumCoNetAR if use_ar else QuantumCoNet

        # Create model with configuration
        model_config = {
            "N": self.cfg.model.N,
            "K": self.cfg.model.K,
            "adjacency_list": self.adjacency_list,
            "max_steps": self.cfg.model.max_steps,
            "device": self.device
        }

        # Add optional parameters with fallbacks
        if hasattr(self.cfg.model, 'init_scale'):
            model_config["init_scale"] = self.cfg.model.init_scale

        # Add use_ar parameter for AR models
        if use_ar:
            model_config["use_ar"] = True

        self.model = model_class(**model_config)
        self.logger.info(f"Created {model_class.__name__} model with {sum(p.numel() for p in self.model.parameters())} parameters")

    def _setup_optimizer(self):
        """Setup optimizer and learning rate scheduler using Hydra instantiation"""
        # Setup optimizer
        optimizer_cfg = getattr(self.cfg, 'optimizer', None)
        if optimizer_cfg and "_target_" in optimizer_cfg and optimizer_cfg._target_ is not None:
            # Use Hydra instantiation for custom optimizers
            # Create optimizer config with model parameters
            optimizer_config = OmegaConf.to_container(optimizer_cfg, resolve=True)
            optimizer_config['params'] = self.model.parameters()
            self.optimizer = instantiate(optimizer_config)
        else:
            # Default Adam optimizer - handle missing keys gracefully
            lr = self.cfg.training.learning_rate
            weight_decay = getattr(self.cfg.training, 'weight_decay', 0.0)

            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )

        # Setup scheduler if specified
        scheduler_cfg = getattr(self.cfg, 'scheduler', None)
        if scheduler_cfg and scheduler_cfg._target_ is not None:
            scheduler_class = instantiate(scheduler_cfg._target_)
            scheduler_config = OmegaConf.to_container(scheduler_cfg, resolve=True)
            scheduler_config.pop("_target_", None)
            self.scheduler = scheduler_class(self.optimizer, **scheduler_config)
        else:
            self.scheduler = None

        self.logger.info(f"Setup optimizer: {type(self.optimizer).__name__}")
        if self.scheduler is not None:
            self.logger.info(f"Setup scheduler: {type(self.scheduler).__name__}")

    def _setup_output_dirs(self):
        """Create output directories for results and checkpoints"""
        from datetime import datetime
        
        # Get output_dir with fallback to 'results' if not specified
        output_dir = getattr(self.cfg, 'output_dir', 'results')
        
        # Build unique experiment name with parameters and timestamp
        base_name = self.cfg.experiment_name
        N = getattr(self.cfg.model, 'N', 'X')
        K = getattr(self.cfg.model, 'K', 'X')
        M = getattr(self.cfg.model, 'max_steps', 'X')
        B = getattr(self.cfg.training, 'batch_size', getattr(self.cfg.data, 'num_qa_pairs', 'X'))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create unique experiment folder name
        experiment_folder = f"{base_name}_N{N}_K{K}_M{M}_B{B}_{timestamp}"
        self.output_dir = Path(output_dir) / experiment_folder
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.logs_dir = self.output_dir / "logs"

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        
        self.logger.info(f"Output directory: {self.output_dir}")

        # Save configuration
        config_path = self.output_dir / "config.yaml"
        with open(config_path, 'w') as f:
            OmegaConf.save(self.cfg, f)

    def _generate_qa_pairs(self) -> List[tuple]:
        """Generate QA pairs for training"""
        if self.adjacency_list is None:
            raise ValueError("Adjacency list must be set before generating QA pairs")

        N = len(self.adjacency_list)
        qa_pairs = []

        # Generate pairs with different distances
        attempts = 0
        max_attempts = self.cfg.data.num_qa_pairs * 10

        while len(qa_pairs) < self.cfg.data.num_qa_pairs and attempts < max_attempts:
            start = torch.randint(0, N, (1,)).item()
            target = torch.randint(0, N, (1,)).item()

            if start != target:
                # Calculate circular distance for ring graphs
                distance = min(abs(start - target), N - abs(start - target))

                if distance <= self.cfg.data.max_distance and (start, target) not in qa_pairs:
                    qa_pairs.append((start, target))

            attempts += 1

        if len(qa_pairs) < self.cfg.data.num_qa_pairs:
            self.logger.warning(f"Only generated {len(qa_pairs)} QA pairs out of requested {self.cfg.data.num_qa_pairs}")

        self.logger.info(f"Generated {len(qa_pairs)} QA pairs (max distance: {self.cfg.data.max_distance})")
        return qa_pairs

    def _check_early_stopping(self) -> bool:
        """Check if early stopping should be triggered"""
        early_stopping_enabled = getattr(self.cfg.training, 'early_stopping', {}).get('enabled', False)
        if not early_stopping_enabled:
            return False

        patience = getattr(self.cfg.training, 'early_stopping', {}).get('patience', 20)
        if len(self.training_history) < patience:
            return False

        # Check if metric has improved
        monitor = getattr(self.cfg.training, 'early_stopping', {}).get('monitor', 'success_rate')
        recent_metrics = [m[monitor] for m in self.training_history[-patience:]]

        mode = getattr(self.cfg.training, 'model_selection', {}).get('mode', 'max')
        best_recent = max(recent_metrics) if mode == "max" else min(recent_metrics)

        # Check if improvement is significant
        min_delta = getattr(self.cfg.training, 'early_stopping', {}).get('min_delta', 0.001)
        if len(recent_metrics) > 1:
            previous_best = max(recent_metrics[:-1]) if mode == "max" else min(recent_metrics[:-1])
            improvement = abs(best_recent - previous_best)
            return improvement < min_delta

        return False

    def _initialize_model(self) -> QuantumCoNet:
        """Initialize model with specified strategy"""
        # This will be implemented with various initialization methods
        pass

    def train(self, qa_pairs: Optional[List[tuple]] = None) -> Dict[str, Any]:
        """
        Main training function

        Args:
            qa_pairs: Optional list of QA pairs to train on. If None, generates from config

        Returns:
            Dictionary containing training results and metrics
        """
        start_time = time.time()

        # Setup QA pairs
        if qa_pairs is not None:
            self.qa_pairs = qa_pairs
        else:
            self.qa_pairs = self._generate_qa_pairs()

        # Initialize model and optimizer if not already set up
        if self.model is None:
            self._setup_model()
        if self.optimizer is None:
            self._setup_optimizer()

        self.logger.info(f"Starting training with {len(self.qa_pairs)} QA pairs")
        self.logger.info(f"Training for {self.cfg.training.num_epochs} epochs")

        # Training loop
        self.training_history = []
        self.best_metrics = {"success_rate": 0.0, "epoch": 0}

        for epoch in range(self.cfg.training.num_epochs):
            self.current_epoch = epoch

            # Train for one epoch
            epoch_metrics = self.train_epoch(self.qa_pairs)

            # Evaluate if needed
            eval_frequency = getattr(self.cfg.training, 'eval_frequency', 10)
            if epoch % eval_frequency == 0:
                eval_metrics = self.evaluate(self.qa_pairs)
                epoch_metrics.update({f"eval_{k}": v for k, v in eval_metrics.items()})

            # Update best metrics
            # Use average_success_rate if available (batch training), otherwise use success_rate
            current_success = epoch_metrics.get("average_success_rate", epoch_metrics.get("success_rate", 0))
            if current_success > self.best_metrics.get("average_success_rate", self.best_metrics.get("success_rate", 0)):
                # Store both for compatibility
                if "average_success_rate" in epoch_metrics:
                    self.best_metrics["average_success_rate"] = epoch_metrics["average_success_rate"]
                self.best_metrics["success_rate"] = epoch_metrics.get("success_rate", current_success)
                self.best_metrics["epoch"] = epoch
                save_checkpoints = getattr(self.cfg.training, 'save_checkpoints', True)
                if save_checkpoints:
                    self.save_checkpoint(epoch, is_best=True)

            # Save checkpoint if needed
            save_frequency = getattr(self.cfg.training, 'save_frequency', 50)
            save_checkpoints = getattr(self.cfg.training, 'save_checkpoints', True)
            if save_checkpoints and epoch % save_frequency == 0:
                self.save_checkpoint(epoch)

            # Log progress
            log_frequency = getattr(self.cfg.training, 'log_frequency', 1)
            if epoch % log_frequency == 0:
                success_rate = epoch_metrics['success_rate']
                # Handle both tensor and float formats
                if hasattr(success_rate, 'item'):
                    success_rate = success_rate.item()
                self.logger.info(f"Epoch {epoch:3d}: success_rate={success_rate:.4f}, "
                                 f"loss={epoch_metrics['loss']:.6f}")

            # Store history
            epoch_metrics["epoch"] = epoch
            self.training_history.append(epoch_metrics)

            # Early stopping check
            if self._check_early_stopping():
                self.logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        # Final evaluation
        final_metrics = self.evaluate(self.qa_pairs)

        training_time = time.time() - start_time

        results = {
            "config": OmegaConf.to_container(self.cfg, resolve=True),
            "training_history": self.training_history,
            "best_metrics": self.best_metrics,
            "final_metrics": final_metrics,
            "training_time": training_time,
            "total_epochs": len(self.training_history)
        }

        self.logger.info(f"Training completed in {training_time:.2f} seconds")
        self.logger.info(f"Best success rate: {self.best_metrics['success_rate']:.4f} at epoch {self.best_metrics['epoch']}")

        return results

    def train_epoch(self, qa_pairs: List[tuple]) -> Dict[str, float]:
        """
        Train for one epoch

        Args:
            qa_pairs: List of (start_node, target_node) pairs

        Returns:
            Dictionary of metrics for this epoch
        """
        self.model.train()

        # Create batches
        batch_size = min(self.cfg.training.batch_size, len(qa_pairs))
        num_batches = (len(qa_pairs) + batch_size - 1) // batch_size

        total_loss = 0.0
        total_success_rate = 0.0
        all_individual_rates = []  # Collect all individual rates across batches
        all_avg_steps = []  # Collect avg_steps_to_success across batches

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(qa_pairs))
            batch_qa_pairs = qa_pairs[start_idx:end_idx]

            # Extract start and target nodes
            start_nodes = [qa[0] for qa in batch_qa_pairs]
            target_nodes = [qa[1] for qa in batch_qa_pairs]

            # Forward pass
            loss_type = getattr(self.cfg.model, 'loss_type', 'standard')
            if getattr(self.model, 'markov_grover', False) and loss_type == 'grover':
                # cheap which-path-exact forward (position-Markov DP, no N*K^M state)
                loss, metrics = self.model.grover_loss_markov(
                    start_nodes, target_nodes, return_metrics=True)
            else:
                # Generate initial states
                init_strategy = getattr(self.cfg.model, 'init_strategy', 'unique')
                if init_strategy == "unique":
                    initial_psi = self.model.generate_batched_initial_states(start_nodes, target_nodes, initial_coin_state=init_strategy)
                else:
                    initial_psi = self.model.generate_batched_initial_states(start_nodes, initial_coin_state=init_strategy)
                loss, metrics = self.model(initial_psi, target_nodes, loss_type=loss_type, return_metrics=True)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Skip update if gradients contain NaN/Inf (saturation protection)
            has_bad_grad = False
            for p in self.model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    has_bad_grad = True
                    break

            if has_bad_grad:
                self.optimizer.zero_grad()  # discard bad gradients
            else:
                # Gradient clipping if specified
                gradient_clip = getattr(self.cfg.training, 'gradient_clip', None)
                if gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
                self.optimizer.step()

            # Accumulate metrics
            total_loss += loss.item()
            total_success_rate += metrics['average_success_rate']
            # Collect avg_steps_to_success if it exists in metrics
            if 'avg_steps_to_success' in metrics:
                all_avg_steps.append(float(metrics['avg_steps_to_success']))

            # ALWAYS collect individual rates from metrics
            if 'individual_success_rates' in metrics:
                all_individual_rates.extend(metrics['individual_success_rates'])

        # Average metrics over batches
        avg_loss = total_loss / num_batches
        avg_success_rate = total_success_rate / num_batches

        # Compute average of avg_steps_to_success, filtering out inf values (no success cases)
        avg_success_steps = float('inf')  # Default to inf if no success in any batch
        finite_steps = [step for step in all_avg_steps if math.isfinite(step)]
        if finite_steps:
            avg_success_steps = sum(finite_steps) / len(finite_steps)

        # Build return dict with both average and individual rates
        result = {
            "loss": avg_loss,
            "success_rate": avg_success_rate,
            "num_batches": num_batches
        }

        # Add individual rates if we collected them (which we always do for batch training)
        if all_individual_rates:
            result["individual_success_rates"] = all_individual_rates
            result["average_success_rate"] = avg_success_rate

        # Add avg_steps_to_success if we have finite values
        if math.isfinite(avg_success_steps):
            result["avg_steps_to_success"] = avg_success_steps

        return result

    def evaluate(self, qa_pairs: List[tuple]) -> Dict[str, float]:
        """
        Evaluate model on given QA pairs

        Args:
            qa_pairs: List of (start_node, target_node) pairs

        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()

        with torch.no_grad():
            # Create batches
            batch_size = min(self.cfg.training.batch_size, len(qa_pairs))
            num_batches = (len(qa_pairs) + batch_size - 1) // batch_size

            total_success_rate = 0.0
            individual_probs_list = []

            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(qa_pairs))
                batch_qa_pairs = qa_pairs[start_idx:end_idx]

                # Extract start and target nodes
                start_nodes = [qa[0] for qa in batch_qa_pairs]
                target_nodes = [qa[1] for qa in batch_qa_pairs]

                # Success probabilities (cheap which-path-exact path when enabled)
                if getattr(self.model, 'markov_grover', False):
                    p = self.model.markov_success_prob(start_nodes, target_nodes)
                    total_success_rate += p.mean().item()
                    individual_probs_list.extend(p.detach().cpu().tolist())
                else:
                    init_strategy = getattr(self.cfg.model, 'init_strategy', 'unique')
                    if init_strategy=="unique":
                        initial_psi = self.model.generate_batched_initial_states(start_nodes, end_nodes=target_nodes, initial_coin_state=init_strategy)
                    else:
                        initial_psi = self.model.generate_batched_initial_states(start_nodes, initial_coin_state=init_strategy)
                    trajectory = self.model.generate_trajectory(initial_psi, target_nodes)
                    total_success_rate += trajectory['total_success_prob'].mean().item()
                    individual_probs_list.extend(trajectory['total_success_prob'].tolist())

            # Average metrics
            avg_success_rate = total_success_rate / num_batches
            individual_probs = torch.tensor(individual_probs_list)

            return {
                "success_rate": avg_success_rate,
                "individual_success_rates": individual_probs.tolist(),
                "mean_individual_success": individual_probs.mean().item(),
                "std_individual_success": individual_probs.std().item(),
                "min_individual_success": individual_probs.min().item(),
                "max_individual_success": individual_probs.max().item()
            }

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save model checkpoint

        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_metrics": self.best_metrics,
            "training_history": self.training_history,
            "config": OmegaConf.to_container(self.cfg, resolve=True)
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        # Save regular checkpoint
        checkpoint_path = self.checkpoints_dir / f"epoch_{epoch:04d}.pt"
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoints_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model: {best_path}")

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load model checkpoint

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Epoch number that was loaded
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_metrics = checkpoint["best_metrics"]
        self.training_history = checkpoint["training_history"]

        if "scheduler_state_dict" in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        loaded_epoch = checkpoint["epoch"]
        self.current_epoch = loaded_epoch + 1

        self.logger.info(f"Loaded checkpoint from epoch {loaded_epoch}")
        return loaded_epoch

    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training progress and results"""
        if not self.training_history:
            return {"status": "not_started"}

        final_metrics = self.training_history[-1] if self.training_history else {}

        return {
            "status": "completed",
            "total_epochs": len(self.training_history),
            "best_success_rate": self.best_metrics.get("success_rate", 0.0),
            "best_epoch": self.best_metrics.get("epoch", 0),
            "final_success_rate": final_metrics.get("success_rate", 0.0),
            "final_loss": final_metrics.get("loss", 0.0),
            "training_time": getattr(self, "_training_time", 0.0),
            "config_summary": {
                "N": self.cfg.model.N,
                "K": self.cfg.model.K,
                "max_steps": self.cfg.model.max_steps,
                "num_epochs": self.cfg.training.num_epochs,
                "learning_rate": self.cfg.training.learning_rate,
                "batch_size": self.cfg.training.batch_size,
                "loss_type": self.cfg.model.loss_type
            }
        }

    def plot_training_curves(self, save_path: Optional[str] = None):
        """
        Plot training curves with individual QA pair success rates

        Args:
            save_path: Optional path to save plot. If None, displays plot
        """
        if not self.training_history:
            self.logger.warning("No training history available for plotting")
            return

        try:
            import matplotlib.pyplot as plt

            epochs = [m["epoch"] for m in self.training_history]
            losses = [m["loss"] for m in self.training_history]

            # Check if we have individual success rates
            has_individual = 'individual_success_rates' in self.training_history[0]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            if has_individual:
                # Extract individual rates for each QA pair
                num_qa_pairs = len(self.training_history[0]['individual_success_rates'])
                individual_rates = {i: [] for i in range(num_qa_pairs)}

                for data in self.training_history:
                    for i, rate in enumerate(data['individual_success_rates']):
                        if i < num_qa_pairs:
                            individual_rates[i].append(rate)

                avg_rates = [data.get('average_success_rate', 0.0) for data in self.training_history]

                # Plot individual QA pair success rates
                colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
                for i in range(num_qa_pairs):
                    ax1.plot(epochs, individual_rates[i], '-', linewidth=2,
                            label=f'QA {i}', alpha=0.8, color=colors[i % len(colors)])

                # Plot average success rate
                ax1.plot(epochs, avg_rates, 'k--', linewidth=3, label='Average', alpha=0.7)

                # Mark best average
                best_avg_rate = self.best_metrics.get("average_success_rate", self.best_metrics.get("success_rate", 0.0))
                ax1.axhline(y=best_avg_rate, color='r', linestyle=':',
                           label=f'Best Avg: {best_avg_rate:.4f}', alpha=0.8)
            else:
                # Fallback to old format (single success rate)
                success_rates = [m["success_rate"] for m in self.training_history]
                ax1.plot(epochs, success_rates, 'b-', linewidth=2, label='Success Rate')
                ax1.axhline(y=self.best_metrics["success_rate"], color='r', linestyle='--',
                           label=f'Best: {self.best_metrics["success_rate"]:.4f}')

            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Success Rate')
            ax1.set_title('Training Success Rates')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Loss plot
            ax2.plot(epochs, losses, 'g-', linewidth=2, label='Loss')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.set_title('Training Loss')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                self.logger.info(f"Training curves saved to: {save_path}")
            else:
                plt.show()

        except ImportError:
            self.logger.warning("Matplotlib not available - skipping training curves plot")

    def export_results(self, format: str = "json") -> str:
        """
        Export training results in specified format

        Args:
            format: Export format ("json", "csv", "pickle")

        Returns:
            Path to exported file
        """
        if not self.training_history:
            raise ValueError("No training results available for export")

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_filename = f"training_results_{timestamp}"

        if format == "json":
            export_path = self.output_dir / f"{base_filename}.json"
            results = {
                "config": OmegaConf.to_container(self.cfg, resolve=True),
                "training_history": self.training_history,
                "best_metrics": self.best_metrics,
                "summary": self.get_training_summary()
            }
            with open(export_path, 'w') as f:
                json.dump(results, f, indent=2)

        elif format == "csv":
            import csv
            export_path = self.output_dir / f"{base_filename}.csv"
            with open(export_path, 'w', newline='') as f:
                if self.training_history:
                    writer = csv.DictWriter(f, fieldnames=self.training_history[0].keys())
                    writer.writeheader()
                    writer.writerows(self.training_history)

        elif format == "pickle":
            import pickle
            export_path = self.output_dir / f"{base_filename}.pkl"
            results = {
                "config": self.cfg,
                "training_history": self.training_history,
                "best_metrics": self.best_metrics,
                "model_state": self.model.state_dict() if self.model else None
            }
            with open(export_path, 'wb') as f:
                pickle.dump(results, f)

        else:
            raise ValueError(f"Unsupported export format: {format}")

        self.logger.info(f"Results exported to: {export_path}")
        return str(export_path)

    def analyze_model(self) -> Dict[str, Any]:
        """
        Analyze trained model characteristics

        Returns:
            Dictionary containing model analysis results
        """
        if self.model is None:
            return {"error": "Model not initialized"}

        # Analyze coin operators
        coin_params = []
        for name, param in self.model.named_parameters():
            if 'coin' in name.lower():
                coin_params.append({
                    "name": name,
                    "shape": list(param.shape),
                    "mean": param.mean().item(),
                    "std": param.std().item(),
                    "min": param.min().item(),
                    "max": param.max().item()
                })

        return {
            "total_parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            "coin_operator_stats": coin_params,
            "model_config": {
                "N": self.cfg.model.N,
                "K": self.cfg.model.K,
                "max_steps": self.cfg.model.max_steps
            }
        }


def create_default_config(**kwargs) -> DictConfig:
    """
    Create a default configuration with optional overrides

    Args:
        **kwargs: Configuration parameters to override

    Returns:
        OmegaConf DictConfig with default values and overrides applied
    """
    config = OmegaConf.create(DEFAULT_CONFIG)

    # Apply overrides
    for key, value in kwargs.items():
        OmegaConf.set_struct(config, False)  # Allow modifications
        if "." in key:
            # Handle nested keys like "model.N"
            OmegaConf.update(config, key, value)
        else:
            # Handle top-level keys or nested dict updates
            if key in config:
                if isinstance(value, dict) and isinstance(config[key], dict):
                    # For nested dict updates, update individual keys to preserve structure
                    for subkey, subvalue in value.items():
                        config[key][subkey] = subvalue
                else:
                    config[key] = value
            else:
                raise ValueError(f"Unknown config parameter: {key}")

    # Don't resolve interpolations for default config to avoid issues
    return config


def load_config_from_file(config_path: str) -> DictConfig:
    """
    Load configuration from YAML file

    Args:
        config_path: Path to YAML configuration file

    Returns:
        OmegaConf DictConfig loaded from file
    """
    return OmegaConf.load(config_path)


def save_config_to_file(config: DictConfig, save_path: str):
    """
    Save configuration to YAML file

    Args:
        config: Configuration to save
        save_path: Path to save YAML file
    """
    OmegaConf.save(config, save_path)


# Example usage function
def example_usage():
    """Example of how to use the QuCoNetTrainer with OmegaConf"""

    # Create configuration using OmegaConf
    cfg = create_default_config(
        model={"N": 20, "K": 3, "max_steps": 15},
        training={"num_epochs": 100, "learning_rate": 0.01, "batch_size": 32},
        experiment_name="lazy_walk_example"
    )

    # Create trainer
    trainer = QuCoNetTrainer(cfg)

    # Set up a simple graph (1D chain)
    def create_1d_chain_adjacency(N):
        adjacency_list = []
        for i in range(N):
            neighbors = [(i-1) % N, (i+1) % N]
            adjacency_list.append(neighbors)
        return adjacency_list

    trainer.adjacency_list = create_1d_chain_adjacency(cfg.model.N)

    # Train model
    results = trainer.train()

    # Get summary
    summary = trainer.get_training_summary()
    print(f"Training completed! Best success rate: {summary['best_success_rate']:.4f}")

    # Plot training curves
    trainer.plot_training_curves()

    # Export results
    export_path = trainer.export_results("json")
    print(f"Results exported to: {export_path}")


if __name__ == "__main__":
    example_usage()
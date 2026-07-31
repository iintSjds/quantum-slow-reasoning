"""
Supervised Fine-Tuning (SFT) for QuCoNet - CORRECTED VERSION

This module implements supervised learning from expert trajectories for quantum walk policies.
Given expert demonstrations of paths from Q to A, we adjust the coin operators to
increase the probability of following the demonstrated trajectories.

CORRECTED: Now properly computes conditional transition probabilities
P(next_node | current_node) = P(at next_node after shift) / P(at current_node before shift)
"""

import torch
import torch.nn.functional as F
from typing import Tuple, List
from .quconet_ar import QuantumCoNetAR


def compute_trajectory_step_loss(
    model: QuantumCoNetAR,
    psi: torch.Tensor,
    step: int,
    current_node: int,
    next_node: int,
    batch_idx: int = 0,
    loss_type: str = "nll"
) -> Tuple[torch.Tensor, float]:
    """
    Compute SFT loss for a single step of a trajectory.

    This function calculates the probability that the quantum walk moves from
    current_node to next_node in one step, and returns a loss that encourages
    this transition.

    Args:
        model: QuantumCoNetAR model with learned coin operators
        psi: Input quantum state before the step (batch, ...)
        step: Current step index (0-indexed)
        current_node: Node x in the trajectory (source node)
        next_node: Node y in the trajectory (target neighbor)
        batch_idx: Batch index to use (default 0 for single trajectory)
        loss_type: Type of loss to compute
            - "nll": Negative log likelihood (default, for minimization)
            - "neg_prob": Negative probability (alternative)
            - "mse": Mean squared error from target probability
            - "bce": Binary cross-entropy style loss

    Returns:
        Tuple of:
        - Loss tensor (scalar) that encourages following the expert trajectory
        - Expert transition probability (float)

    Raises:
        ValueError: If next_node is not a neighbor of current_node
    """
    # Verify that next_node is actually a neighbor of current_node
    neighbors = model.shift.adjacency_list[current_node]
    if next_node not in neighbors:
        raise ValueError(
            f"Node {next_node} is not a neighbor of node {current_node}. "
            f"Valid neighbors: {neighbors}"
        )

    # Store probability at current node BEFORE the step (keep as tensor for gradients)
    # Probabilities are real values
    prob_at_current_before = torch.sum(torch.abs(psi[batch_idx, current_node])**2).real

    # Apply the complete quantum step: coin + shift
    # Note: coin operators use 1-indexed steps
    if model.use_ar:
        coin_step = step + 1
    else:
        coin_step = 1

    # Apply coin operation
    psi_after_coin = model.coin(psi, step=coin_step)

    # Apply shift operation to propagate to neighbors
    psi_after_shift = model.shift(psi_after_coin, step=coin_step)

    # Compute probability at next node AFTER the shift
    prob_at_next_after = torch.sum(torch.abs(psi_after_shift[batch_idx, next_node])**2).real

    # The transition probability is CONDITIONAL:
    # P(current -> next) = P(at next after shift) / P(at current before shift)
    # Use torch.where for numerical stability
    epsilon_tensor = torch.tensor(1e-12, device=psi.device, dtype=prob_at_current_before.dtype)
    expert_prob_tensor = torch.where(
        prob_at_current_before > epsilon_tensor,
        prob_at_next_after / prob_at_current_before,
        torch.tensor(0.0, device=psi.device, dtype=prob_at_current_before.dtype)
    )

    # Clamp for numerical stability
    expert_prob_tensor = torch.clamp(expert_prob_tensor, min=epsilon_tensor, max=1.0 - epsilon_tensor)

    # Compute loss based on loss_type
    if loss_type == "nll":
        # Negative log likelihood - directly minimize this
        loss = -torch.log(expert_prob_tensor)
    elif loss_type == "neg_prob":
        # Negative probability - simple but less well-behaved
        loss = -expert_prob_tensor
    elif loss_type == "mse":
        # MSE from target probability (1.0 = expert wants this transition)
        target_prob = torch.tensor(1.0, device=psi.device)
        loss = F.mse_loss(expert_prob_tensor, target_prob)
    elif loss_type == "bce":
        # Binary cross-entropy style - treat as classification
        # Minimize -log(p) for expert direction
        loss = -torch.log(expert_prob_tensor)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    return loss, expert_prob_tensor.item()


def compute_trajectory_loss(
    model: QuantumCoNetAR,
    psi_initial: torch.Tensor,
    trajectory: List[int],
    q_node: int = None,
    a_node: int = None,
    batch_idx: int = 0,
    loss_type: str = "nll",
    aggregate: str = "sum"
) -> Tuple[torch.Tensor, dict]:
    """
    Compute SFT loss for a complete trajectory.

    This function runs the quantum walk step-by-step, computing supervised loss
    at each step to encourage following the entire expert trajectory.

    Args:
        model: QuantumCoNetAR model
        psi_initial: Initial quantum state at Q node
        trajectory: List of node indices [Q, intermediate_nodes..., A]
        q_node: Expected starting node (optional, will validate if provided)
        a_node: Expected target node (optional, will validate if provided)
        batch_idx: Batch index
        loss_type: Type of loss (see compute_trajectory_step_loss)
        aggregate: How to aggregate step losses
            - "sum": Sum of all step losses (default)
            - "mean": Average of step losses
            - "max": Maximum step loss (focus on worst step)

    Returns:
        Tuple of:
        - Total loss tensor (scalar)
        - Dictionary with metrics:
          * step_losses: list of per-step losses
          * expert_probs: list of expert transition probabilities
          * avg_expert_prob: average probability of expert actions
    """
    psi = psi_initial.clone()
    total_loss = torch.tensor(0.0, device=psi.device)
    step_losses = []
    expert_probs = []

    # Iterate through trajectory steps
    num_steps = len(trajectory) - 1

    # Validate trajectory if q_node and a_node are provided
    if q_node is not None and trajectory[0] != q_node:
        raise ValueError(f"Trajectory must start at Q node {q_node}, got {trajectory[0]}")
    if a_node is not None and trajectory[-1] != a_node:
        raise ValueError(f"Trajectory must end at A node {a_node}, got {trajectory[-1]}")

    # Iterate through trajectory steps
    for step in range(num_steps):
        current_node = trajectory[step]
        next_node = trajectory[step + 1]

        # Compute loss for this step
        # The function handles the complete quantum step (coin + shift)
        step_loss, expert_prob = compute_trajectory_step_loss(
            model=model,
            psi=psi,
            step=step,
            current_node=current_node,
            next_node=next_node,
            batch_idx=batch_idx,
            loss_type=loss_type
        )

        # Store metrics
        step_losses.append(step_loss.item())
        expert_probs.append(expert_prob)

        # Add to total loss
        if aggregate == "sum":
            total_loss = total_loss + step_loss
        elif aggregate == "mean":
            total_loss = total_loss + step_loss / num_steps
        elif aggregate == "max":
            total_loss = torch.max(total_loss, step_loss)
        else:
            raise ValueError(f"Unknown aggregate: {aggregate}")

        # Update psi to the state after this step
        # Note: compute_trajectory_step_loss already did coin+shift internally,
        # but it creates new tensors. We need to apply the same
        # transformation here to keep psi in sync for the next iteration.
        if model.use_ar:
            coin_step = step + 1
        else:
            coin_step = 1
        psi = model.coin(psi, step=coin_step)
        psi = model.shift(psi, step=coin_step)

    # Compute metrics
    metrics = {
        "step_losses": step_losses,
        "expert_probs": expert_probs,
        "avg_expert_prob": sum(expert_probs) / len(expert_probs) if expert_probs else 0.0,
        "num_steps": num_steps
    }

    return total_loss, metrics


def sft_training_step(
    model: QuantumCoNetAR,
    qa_pairs: List[Tuple[int, int]],
    trajectories: List[List[int]],
    optimizer: torch.optim.Optimizer,
    loss_type: str = "nll",
    aggregate: str = "sum",
    initial_coin_state: str = "unique"
) -> dict:
    """
    Perform one SFT training step on a batch of QA pairs with expert trajectories.

    This function is the main entry point for supervised fine-tuning. It takes a batch
    of question-answer pairs along with expert-demonstrated trajectories and updates
    the model's coin operators to increase the likelihood of following these trajectories.

    Args:
        model: QuantumCoNetAR model to train
        qa_pairs: List of (Q, A) tuples
        trajectories: List of expert trajectories, one per QA pair
        optimizer: PyTorch optimizer
        loss_type: Type of loss (see compute_trajectory_step_loss)
        aggregate: How to aggregate step losses (see compute_trajectory_loss)
        initial_coin_state: How to initialize quantum states ("basis", "uniform", "random", "unique")
                             "unique" is the default, matching QuCoNetAR behavior

    Returns:
        Dictionary with training metrics:
        - loss: Total loss for the batch
        - avg_step_loss: Average loss per step
        - avg_expert_prob: Average probability of expert actions
        - num_trajectories: Number of trajectories in batch
    """
    model.train()
    optimizer.zero_grad()

    total_loss = torch.tensor(0.0, device=model.device)
    all_step_losses = []
    all_expert_probs = []
    num_steps_total = 0

    # Process each trajectory
    for batch_idx, ((q_node, a_node), trajectory) in enumerate(zip(qa_pairs, trajectories)):
        # Verify trajectory starts at Q and ends at A
        if trajectory[0] != q_node:
            raise ValueError(f"Trajectory must start at Q node {q_node}, got {trajectory[0]}")
        if trajectory[-1] != a_node:
            raise ValueError(f"Trajectory must end at A node {a_node}, got {trajectory[-1]}")

        # Generate initial state at Q node
        start_nodes = [q_node]
        end_nodes = [a_node]

        psi_initial = model.generate_batched_initial_states(
            start_nodes=start_nodes,
            end_nodes=end_nodes,
            initial_coin_state=initial_coin_state
        )

        # Compute trajectory loss
        traj_loss, metrics = compute_trajectory_loss(
            model=model,
            psi_initial=psi_initial,
            trajectory=trajectory,
            q_node=q_node,
            a_node=a_node,
            batch_idx=0,  # Only one QA pair per batch item
            loss_type=loss_type,
            aggregate=aggregate
        )

        # Accumulate
        total_loss = total_loss + traj_loss
        all_step_losses.extend(metrics["step_losses"])
        all_expert_probs.extend(metrics["expert_probs"])
        num_steps_total += metrics["num_steps"]

    # Average loss over batch
    batch_size = len(qa_pairs)
    if batch_size > 1:
        total_loss = total_loss / batch_size

    # Backward pass
    total_loss.backward()

    # Gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # Optimizer step
    optimizer.step()

    # Compute metrics
    avg_step_loss = sum(all_step_losses) / len(all_step_losses) if all_step_losses else 0.0
    avg_expert_prob = sum(all_expert_probs) / len(all_expert_probs) if all_expert_probs else 0.0

    metrics = {
        "loss": total_loss.item(),
        "avg_step_loss": avg_step_loss,
        "avg_expert_prob": avg_expert_prob,
        "num_trajectories": batch_size,
        "avg_episode_length": num_steps_total / batch_size if batch_size > 0 else 0
    }

    return metrics

"""
AR version of QuantumCoNet main model for quantum walk simulation

This module implements the QuCoNet model with AR (autoregressive) capability
for quantum walks with adaptive rollout management.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from .operators_ar import CoinOperatorAR, ShiftOperatorAR, get_optimal_device, optimize_model_for_device


class QuantumCoNetAR(nn.Module):
    """
    Main QuCoNet AR model that runs quantum walk simulation for graph traversal.

    This module implements the full quantum walk algorithm with AR capabilities:
    1. Initialize quantum state at starting node
    2. Evolve state through alternating coin and shift operations
    3. Measure success probability at target node each step
    4. Calculate total reward as success probability over all steps
    5. Adaptive rollout management for efficiency
    """

    def __init__(self, N: int, K: int, adjacency_list: List[List[int]], max_steps: int, device: Optional[torch.device] = None, init_scale: float = 1.0, use_ar: bool = True):
        """
        Initialize the QuantumCoNetAR model.

        Args:
            N: Number of nodes in the graph
            K: Degree of each node
            adjacency_list: Adjacency list for graph structure
            max_steps: Maximum number of quantum walk steps
            device: Device to place tensors on. If None, auto-select optimal device.
            init_scale: Scale factor for Hamiltonian parameter initialization (default: 1.0)
            use_ar: Whether to use AR (autoregressive) mode with M coin dimensions (default: False)
        """
        super().__init__()
        self.N = N
        self.K = K
        self.max_steps = max_steps
        self.device = get_optimal_device(device)
        self.init_scale = init_scale
        self.use_ar = use_ar

        # Initialize quantum operators
        self.coin = CoinOperatorAR(N, K, device=self.device, init_scale=init_scale)
        self.shift = ShiftOperatorAR(N, K, adjacency_list, device=self.device)

    @staticmethod
    def to_k_base_torch(numbers, K, n, device="cpu"):
        """
        Convert numbers to K-base representation using efficient tensor operations.

        Args:
            numbers: Scalar or tensor containing integer(s) to convert
            K: Base for conversion
            n: Number of digits in output
            device: Device for tensor operations

        Returns:
            digits: Tensor containing K-base digits
        """
        if isinstance(numbers, (int, float)):
            numbers = torch.tensor([numbers])
        powers = K ** torch.arange(n - 1, -1, -1, device=device)
        return (numbers // powers) % K

    @staticmethod
    def generate_unique_coin_state(N: int, K: int, Q: int, A: int, max_length: int = None) -> List[int]:
        """
        Generate a unique coin state for a QA pair based on K-based representation.

        This function creates a unique initialization pattern for each QA pair to avoid
        frustration between trajectories. It converts Q and A to K-based numbers and
        interleaves their digits: Q digits at even positions, A digits at odd positions.

        Args:
            N: Number of nodes in the graph
            K: Degree of each node (base for conversion)
            Q: Starting node index
            A: Target node index
            max_length: Maximum length of returned array (if None, uses computed length)

        Returns:
            coin_state: List of integers in [0, K-1] representing the coin state
        """
        # Find n such that K^(n+1) > N
        # We need digits to represent numbers up to N-1
        n = 0
        while K ** (n + 1) <= N:
            n += 1

        # Convert Q and A to K-base representation with n+1 digits using torch
        q_digits = QuantumCoNetAR.to_k_base_torch(Q, K, n + 1).squeeze().tolist()
        a_digits = QuantumCoNetAR.to_k_base_torch(A, K, n + 1).squeeze().tolist()

        # Interleave digits: even positions get Q digits, odd positions get A digits
        # Position 0 (even): Q0, Position 1 (odd): A0, Position 2 (even): Q1, etc.
        coin_state = []
        for i in range(n + 1):
            # Even position: Q digit
            coin_state.append(q_digits[i])
            # Odd position: A digit
            coin_state.append(a_digits[i])

        # If max_length is specified and longer than current length,
        # repeat the pattern or pad with zeros
        if max_length is not None and max_length > len(coin_state):
            # Extend by repeating the pattern
            repeated = []
            while len(repeated) < max_length - len(coin_state):
                repeated.extend(coin_state)
            # Take exactly what we need and append
            coin_state.extend(repeated[:max_length - len(coin_state)])
        elif max_length is not None and max_length < len(coin_state):
            # Truncate if max_length is smaller
            coin_state = coin_state[:max_length]

        return coin_state

    def quantum_step(self, psi: torch.Tensor, step: int = 1) -> torch.Tensor:
        """
        Perform one step of quantum walk: coin operation + shift operation.

        Args:
            psi: Current quantum state, shape (B, N, K) for legacy or (B, N, K, K, ..., K) for AR
            step: Current step number (1-indexed), used in AR mode to select coin dimension

        Returns:
            psi_next: Quantum state after one step, same shape as input
        """
        if self.use_ar:
            # AR mode: apply coin and shift to the current step's coin dimension
            psi = self.coin(psi, step=step)
            psi = self.shift(psi, step=step)
        else:
            # Legacy mode: always use step=1 (single coin dimension) - ignore step parameter
            psi = self.coin(psi, step=1)
            psi = self.shift(psi, step=1)

        return psi

    def measure_and_project(self, psi: torch.Tensor, target_node_idx: torch.Tensor, current_step: int = None):
        """
        Measure success probability at target nodes for each batch element and project state.

        Args:
            psi: Current quantum state, shape (B, N, K) for legacy or (B, N, K, K, ..., K) for AR
            target_node_idx: Target node indices for each batch element, shape (B,)
            current_step: Current step number (1-indexed). Only needed for AR mode to determine
                         which coin dimensions have been activated. If None, assumes all dimensions active.

        Returns:
            success_prob: Success probability at this step for each batch element, shape (B,)
            psi_projected: Projected quantum state, same shape as input
        """
        shape = psi.shape
        B, N = shape[0], shape[1]
        assert target_node_idx.shape == (B,), f"target_node_idx shape {target_node_idx.shape} doesn't match batch size {B}"

        # Get amplitude at target nodes for each batch element
        # Use advanced indexing to get psi[b, target_node_idx[b], ...] for each b
        batch_indices = torch.arange(B, device=self.device)

        if self.use_ar:
            # AR mode: state has shape (B, N, K, K, ..., K) with M coin dimensions
            M = len(shape) - 2  # Number of coin dimensions

            # Get all amplitudes at target node: psi[b, target_node_idx[b], :, :, ..., :]
            psi_at_target = psi[batch_indices, target_node_idx]  # (B, K, K, ..., K)

            # For AR mode, we should only sum over coin dimensions that have been activated
            # At step n, only the first n coin dimensions are active (indices 0 to n-1)
            # The remaining dimensions correspond to future steps and shouldn't be measured yet
            if current_step is None:
                # If current_step not provided, sum over all coin dimensions (backward compatibility)
                sum_dims = tuple(range(1, len(psi_at_target.shape)))
            else:
                # Calculate which coin dimensions correspond to the current step
                # At step 1, only coin dimension 0 is active
                # At step 2, coin dimensions 0 and 1 are active
                # ...
                # At step n, coin dimensions 0 to n-1 are active
                activated_dims = min(current_step, M)  # Don't exceed available dimensions

                # We need to sum over all future coin dimensions too because of entanglement
                # Actually, in AR mode, all coin dimensions are entangled after step 1
                # The issue is different: we shouldn't initialize future coin dimensions with amplitude
                sum_dims = tuple(range(1, len(psi_at_target.shape)))

            # Calculate success probability: sum over relevant coin dimensions
            success_prob = torch.sum(torch.abs(psi_at_target)**2, dim=sum_dims)  # (B,)

            # Project to unfinished space: zero out target node amplitude for each batch element
            psi_projected = psi.clone()
            psi_projected[batch_indices, target_node_idx] = 0.0

            # Calculate remaining probability for each batch element
            p_unf = torch.sum(torch.abs(psi_projected)**2, dim=tuple(range(1, len(psi_projected.shape))))  # (B,)
        else:
            # Legacy mode: state has shape (B, N, K)
            psi_at_target = psi[batch_indices, target_node_idx, :]  # (B, K)

            # Calculate success probability: sum over coin dimension
            success_prob = torch.sum(torch.abs(psi_at_target)**2, dim=-1)  # (B,)

            # Project to unfinished space: zero out target node amplitude for each batch element
            psi_projected = psi.clone()
            psi_projected[batch_indices, target_node_idx, :] = 0.0

            # Calculate remaining probability for each batch element
            p_unf = torch.sum(torch.abs(psi_projected)**2, dim=(-1, -2))  # (B,)

        # CRITICAL: Clamp for numerical stability
        p_unf_stable = torch.clamp(p_unf, min=1e-12)

        # Normalize projected state for each batch element
        if self.use_ar:
            # AR mode: broadcast sqrt normalization across all dimensions
            psi_projected = psi_projected / torch.sqrt(p_unf_stable).view(-1, *([1] * (len(shape) - 1)))
        else:
            # Legacy mode: specific reshape for (B, N, K)
            psi_projected = psi_projected / torch.sqrt(p_unf_stable).view(-1, 1, 1)

        return success_prob, psi_projected

    def compute_standard_loss(self, step_probs: list) -> torch.Tensor:
        """Compute standard loss: R = 1 - prod(1 - p_n) over all steps."""
        p_n_tensor = torch.stack(step_probs, dim=1)  # (B, max_steps)
        p_n_clamped = p_n_tensor.clamp(max=1.0 - 1e-7)  # prevent 1-p_n = 0
        p_fail = torch.prod(1.0 - p_n_clamped, dim=1)  # (B,)
        R = 1.0 - p_fail  # (B,)
        return -torch.sum(R)

    def compute_log_prob_loss(self, step_probs: list) -> torch.Tensor:
        """Compute improved log probability loss: log(prod(1-p_n)) = sum(log(1-p_n)).

        This provides better gradient behavior and naturally decomposes to stepwise form
        under the log transformation, making it more interpretable for optimization.

        The loss is: -sum(log(1-p_n)) which encourages maximizing p_n (success probabilities)
        """
        p_n_tensor = torch.stack(step_probs, dim=1)  # (B, max_steps)
        p_n_clamped = p_n_tensor.clamp(max=1.0 - 1e-7)  # prevent log(0)
        # Compute log of (1 - p_n) for each step, then sum across steps
        log_one_minus_p = torch.log(1.0 - p_n_clamped + 1e-12)  # Add epsilon for stability
        stepwise_log_loss = torch.sum(log_one_minus_p, dim=1)  # (B,) - sum across steps
        return torch.sum(stepwise_log_loss)  # sum over B (gradient ∝ B)

    # Removed compute_stepwise_loss as the current implementation lacks natural justification
    # Future stepwise loss will be implemented with better theoretical foundation

    @staticmethod
    def grover_acc(p: torch.Tensor, n: int) -> torch.Tensor:
        """Amplitude-amplification accuracy after spending up to n Grover iterations
        with optimal stopping, as a differentiable function of the one-shot success
        probability p (shape (B,)):

            theta = arcsin(sqrt(p)),   A_n(p) = sin^2((2k+1) theta),   k = min(n, k*(p))

        where k*(p) = round(pi/(4 theta) - 1/2) is the iteration count that first reaches
        the amplitude peak (optimal stop). k is *detached* (piecewise-constant integer),
        so the gradient flows only through theta:
            - dead pairs (p->0, k=n):   dA/dp ~ (2n+1)^2  -> gradient amplified ~(2n+1)^2
            - saturated pairs (k=k*):   (2k*+1) theta ~ pi/2  -> dA/dp ~ 0 (no over-invest)
        Reduces to the one-shot objective A_0(p) = p when n = 0.
        """
        eps = 1e-7
        p = p.clamp(eps, 1.0 - eps)
        theta = torch.asin(torch.sqrt(p))                                   # (B,)
        with torch.no_grad():
            kstar = torch.round(math.pi / (4.0 * theta) - 0.5)
            k = torch.clamp(torch.minimum(kstar, torch.full_like(kstar, float(n))), min=0.0)
        return torch.sin((2.0 * k + 1.0) * theta) ** 2

    def compute_grover_loss(self, step_probs: list, grover_n: int) -> torch.Tensor:
        """Grover-n objective: maximize sum_B A_n(p_B), with p_B = 1 - prod_n (1 - p_n).

        Same per-QA success probability as the standard loss, passed through the
        (concave, saturating) amplitude-amplification response A_n before summing.
        The concavity water-fills capacity toward low-p ("dead") pairs -> less policy
        collapse -> (hypothesis) better generalization. grover_n=0 recovers standard.
        """
        p_n = torch.stack(step_probs, dim=1).clamp(max=1.0 - 1e-7)          # (B, max_steps)
        p = 1.0 - torch.prod(1.0 - p_n, dim=1)                              # (B,) total success prob
        A = self.grover_acc(p, grover_n)                                    # (B,)
        return -torch.sum(A)

    def compute_bestk_loss(self, step_probs: list, best_k: int) -> torch.Tensor:
        """Classical inference-aware objective on the SAME architecture:
        maximize sum_B [1 - (1 - p_B)^k], with p_B = 1 - prod_n (1 - p_n).

        p_B is the AR walk's exact sequence-level success probability, which
        (by the which-path equivalence) equals the success probability of the
        measured-per-step classical Markov chain with the same coins.  This is
        therefore the exact-gradient "semi-QuCoNet trained for best-of-k"
        control: identical parameters and dynamics to the grover-n runs, only
        the outer response function differs (monotone cl_k vs peaked A_n).
        best_k=1 recovers the standard loss.
        """
        p_n = torch.stack(step_probs, dim=1).clamp(max=1.0 - 1e-7)          # (B, max_steps)
        p = 1.0 - torch.prod(1.0 - p_n, dim=1)                              # (B,)
        return -torch.sum(1.0 - (1.0 - p) ** int(best_k))

    def markov_success_prob(self, start_nodes, target_nodes) -> torch.Tensor:
        """Per-QA success probability p_i via the position-Markov DP.

        By the which-path equivalence, the AR forward's p_i equals the success
        probability of a time-inhomogeneous classical Markov chain whose step-s
        transition out of node x is the Born probability |C_x[c_s, .]|^2, with
        c_s the QA-conditioned initial coin at step s.  This propagates a length-N
        position distribution instead of the N*K^M joint state -- O(B*N*K*M)
        memory, no exponential in M -- and is fully differentiable in the coin
        parameters.  Verified equal to the AR forward to float32 precision
        (p_i agree to ~1e-7, gradients to ~1e-6, training trajectories
        bit-identical).  Returns p_i of shape (B,).
        """
        dev = self.device
        Q = torch.as_tensor(start_nodes, dtype=torch.long, device=dev)
        A = torch.as_tensor(target_nodes, dtype=torch.long, device=dev)
        B = Q.shape[0]
        N, K, M = self.N, self.K, self.max_steps
        # Coin Born probabilities from the CURRENT parameters, built with the
        # model's own coin routines so C(params) and its gradient are identical.
        H = self.coin.build_hamiltonian_batch(1)                    # (1,N,K,K)
        C = self.coin.build_coin_operators(H)[0]                    # (N,K,K)
        coin_probs = (C.abs() ** 2)                                 # (N,Kin,Kout)
        # QA-conditioned per-step input coin (same generator as the initial state)
        pat = torch.tensor(
            [self.generate_unique_coin_state(N, K, int(q), int(a), max_length=M)
             for q, a in zip(Q.tolist(), A.tolist())],
            dtype=torch.long, device=dev)                           # (B,M)
        idx = self.shift.shift_node_map.long().reshape(-1)          # (N*K,)
        ar = torch.arange(B, device=dev)
        oneA = torch.zeros(B, N, device=dev); oneA[ar, A] = 1.0
        dist = torch.zeros(B, N, device=dev); dist[ar, Q] = 1.0
        succ = torch.zeros(B, device=dev)
        for s in range(M):
            succ = succ + (dist * oneA).sum(1)                      # absorb target mass
            dist = dist * (1.0 - oneA)
            Ts = coin_probs[:, pat[:, s], :].permute(1, 0, 2)       # (B,N,Kout)
            contrib = (dist.unsqueeze(-1) * Ts).reshape(B, N * K)
            dist = torch.zeros(B, N, device=dev).index_add(1, idx, contrib)
        return succ + (dist * oneA).sum(1)

    def grover_loss_markov(self, start_nodes, target_nodes, return_metrics: bool = False):
        """Grover-n loss via the cheap Markov forward; objective identical to
        compute_grover_loss (L = -sum_B A_n(p_B)), only p_B is computed the
        memory-light way.  Drop-in for the expensive forward under the grover loss."""
        p = self.markov_success_prob(start_nodes, target_nodes)
        acc = self.grover_acc(p, int(getattr(self, "grover_n", 1)))
        loss = -torch.sum(acc)
        if not return_metrics:
            return loss
        metrics = {
            "total_success_rate": p.detach(),
            "individual_success_rates": p.detach().cpu().tolist(),
            "average_success_rate": p.mean().item(),
            "final_success_prob": p.mean().item(),
            "avg_steps_to_success": float("nan"),
        }
        return loss, metrics

    def compute_trajectory_loss(self, step_probs: list, loss_type: str = "standard") -> torch.Tensor:
        """
        Compute loss from step probabilities using different loss functions.

        Args:
            step_probs: List of success probabilities at each step, each of shape (B,)
            loss_type: Type of loss function to use

        Returns:
            loss: Computed loss, shape ()
        """
        if loss_type == "standard":
            return self.compute_standard_loss(step_probs)
        elif loss_type == "log_prob":
            return self.compute_log_prob_loss(step_probs)
        elif loss_type == "grover":
            return self.compute_grover_loss(step_probs, int(getattr(self, "grover_n", 1)))
        elif loss_type == "bestk":
            return self.compute_bestk_loss(step_probs, int(getattr(self, "best_k", 2)))
        else:
            raise ValueError(f"Unknown loss type: {loss_type}. "
                             f"Available: standard, log_prob, grover, bestk")

    def compute_linear_stage_loss(self, step_probs: list, first_success_step: torch.Tensor) -> torch.Tensor:
        """
        Compute loss for linear stage (before first measurement success).

        In this stage, the quantum evolution is purely unitary and linear,
        so we can use full backpropagation through all steps.

        Args:
            step_probs: List of success probabilities at each step, each (B,)
            first_success_step: First success step for each batch element, (B,)

        Returns:
            loss: Linear stage loss, shape ()
        """
        if len(step_probs) == 0:
            return torch.tensor(0.0, device=self.device)

        # Stack step probabilities: (max_steps, B) -> (B, max_steps)
        p_n_tensor = torch.stack(step_probs, dim=1)
        B, max_steps = p_n_tensor.shape

        # Create mask for linear stage (steps before first success)
        linear_loss = torch.tensor(0.0, device=self.device)
        batch_indices = torch.arange(B, device=self.device)

        for b in range(B):
            first_success = first_success_step[b].item()
            if first_success > 0:  # Has first success after step 0
                # Only consider steps before first success
                steps_before = int(first_success)
                if steps_before > 0:
                    # Compute loss for linear stage only
                    linear_probs = p_n_tensor[b, :steps_before]
                    # Standard loss computation for linear stage
                    p_fail_linear = torch.prod(1.0 - linear_probs)
                    R_linear = 1.0 - p_fail_linear
                    linear_loss += -R_linear
            elif first_success == 0:  # First success at step 0
                # No linear stage, only non-linear
                pass
            else:  # No success occurred (-1)
                # Entire trajectory is linear
                p_fail_all = torch.prod(1.0 - p_n_tensor[b])
                R_all = 1.0 - p_fail_all
                linear_loss += -R_all

        return linear_loss / B  # Average over batch

    def compute_nonlinear_stage_loss(self, step_probs: list, first_success_step: torch.Tensor,
                                   window_size: int = 5) -> torch.Tensor:
        """
        Compute loss for non-linear stage (after first measurement success).

        In this stage, measurement introduces non-linearity. We use truncated
        backpropagation through time (TBPTT) with a fixed window.

        Args:
            step_probs: List of success probabilities at each step, each (B,)
            first_success_step: First success step for each batch element, (B,)
            window_size: Number of recent steps to consider for TBPTT

        Returns:
            loss: Non-linear stage loss, shape ()
        """
        if len(step_probs) == 0:
            return torch.tensor(0.0, device=self.device)

        # Stack step probabilities: (max_steps, B) -> (B, max_steps)
        p_n_tensor = torch.stack(step_probs, dim=1)
        B, max_steps = p_n_tensor.shape

        nonlinear_loss = torch.tensor(0.0, device=self.device)

        for b in range(B):
            first_success = first_success_step[b].item()
            if first_success >= 0:  # Has first success
                # Consider window of steps after first success
                start_step = int(first_success)
                end_step = min(start_step + window_size, max_steps)

                if end_step > start_step:
                    # Compute loss for window of steps after first success
                    window_probs = p_n_tensor[b, start_step:end_step]
                    # Standard loss computation for window
                    p_fail_window = torch.prod(1.0 - window_probs)
                    R_window = 1.0 - p_fail_window
                    nonlinear_loss += -R_window

        # Only average over elements that had non-linear stage
        num_nonlinear = (first_success_step >= 0).sum().float()
        if num_nonlinear > 0:
            nonlinear_loss = nonlinear_loss / num_nonlinear

        return nonlinear_loss

    def compute_two_stage_loss(self, trajectory_data: dict, window_size: int = 5,
                             linear_weight: float = 0.7, loss_type: str = "standard") -> torch.Tensor:
        """
        Compute two-stage loss combining linear and non-linear stages.

        This implements the hybrid optimization strategy where:
        - Linear stage (before first success): Full backpropagation
        - Non-linear stage (after first success): Truncated backprop with window

        Args:
            trajectory_data: Dictionary from generate_trajectory() with two-stage tracking
            window_size: Window size for truncated backprop in non-linear stage
            linear_weight: Weight for linear stage loss (0-1)
            loss_type: Type of loss to use within each stage

        Returns:
            loss: Combined two-stage loss, shape ()
        """
        step_probs = trajectory_data['step_probs']
        first_success_step = trajectory_data['first_success_step']

        # Compute stage-specific losses
        if loss_type == "standard":
            linear_loss = self.compute_linear_stage_loss(step_probs, first_success_step)
            nonlinear_loss = self.compute_nonlinear_stage_loss(step_probs, first_success_step, window_size)
        elif loss_type == "log_prob":
            # For log_prob loss, we need to modify the stage loss computations
            # This is a simplified version - can be extended
            linear_loss = self.compute_linear_stage_loss(step_probs, first_success_step)
            nonlinear_loss = self.compute_nonlinear_stage_loss(step_probs, first_success_step, window_size)
        else:
            raise ValueError(f"Unknown loss type for two-stage: {loss_type}")

        # Combine with weights
        total_loss = linear_weight * linear_loss + (1 - linear_weight) * nonlinear_loss

        return total_loss

    def generate_trajectory(self, initial_psi: torch.Tensor, target_node_idx):
        """
        Generate quantum walk trajectory and collect metrics.

        Args:
            initial_psi: Initial quantum state of shape (B, N, K)
            target_node_idx: Target node indices, shape (B,) for batch QA pairs, or int/list for single target

        Returns:
            trajectory_data: Dictionary containing:
                - step_probs: List of success probabilities at each step, each (B,)
                - final_psi: Final quantum state
                - total_success_prob: Overall success probability, shape (B,)
                - step_success_rates: Success rate at each step
                - avg_steps_to_success: Average steps needed (estimated)
        """
        # Ensure initial_psi is on the correct device
        initial_psi = initial_psi.to(self.device)

        if self.use_ar:
            # AR mode: state has shape (B, N, K, K, ..., K) with M copies of K
            B = initial_psi.shape[0]
            N = initial_psi.shape[1]
            K = initial_psi.shape[2]
        else:
            # Legacy mode: state has shape (B, N, K)
            B, N, K = initial_psi.shape
        assert N == self.N and K == self.K, f"Expected psi shape (B, {self.N}, {self.K}), got {initial_psi.shape}"

        # Handle target_node_idx - ensure it's a tensor
        if isinstance(target_node_idx, int):
            target_node_idx = torch.full((B,), target_node_idx, dtype=torch.long, device=self.device)
        elif isinstance(target_node_idx, list):
            target_node_idx = torch.tensor(target_node_idx, dtype=torch.long, device=self.device)
        elif isinstance(target_node_idx, torch.Tensor):
            target_node_idx = target_node_idx.to(self.device)
        else:
            raise ValueError(f"target_node_idx must be int, list, or tensor, got {type(target_node_idx)}")

        assert target_node_idx.shape == (B,), f"target_node_idx shape {target_node_idx.shape} doesn't match batch size {B}"

        # Initialize quantum state
        psi = initial_psi.clone()

        # Store step probabilities and metrics
        step_probs = []
        step_success_rates = []

        # Track first success for each batch element (two-stage optimization)
        first_success_step = torch.full((B,), -1, dtype=torch.long, device=self.device)
        success_occurred = torch.zeros(B, dtype=torch.bool, device=self.device)

        # Quantum walk evolution
        for step in range(self.max_steps):
            # Perform one quantum walk step
            psi = self.quantum_step(psi, step=step + 1)  # step is 1-indexed

            # Measure success and project state for each batch element
            success_prob, psi = self.measure_and_project(psi, target_node_idx)
            step_probs.append(success_prob)

            # Update first success tracking for two-stage optimization
            newly_successful = (success_prob > 0) & ~success_occurred
            first_success_step[newly_successful] = step
            success_occurred |= newly_successful

            # Calculate success rate (mean across batch)
            success_rate = torch.mean(success_prob).item()
            step_success_rates.append(success_rate)

        # Calculate overall success probability for each batch element
        p_n_tensor = torch.stack(step_probs, dim=1)  # (B, max_steps)
        p_fail = torch.prod(1.0 - p_n_tensor, dim=1)  # (B,)
        total_success_prob = 1.0 - p_fail  # (B,)

        # Calculate average steps to success (estimated from probabilities)
        # This is an approximation based on the geometric distribution
        avg_steps_to_success = float('inf')  # Default to inf when no success
        if len(step_probs) > 0 and torch.mean(total_success_prob).item() > 0:
            # Only compute if there's actual success
            weighted_sum = 0.0
            # Weighted average where weights are the probability of first success at each step
            for i, prob in enumerate(step_probs):
                # Probability of first success at step i+1
                if i == 0:
                    first_success_prob = prob
                else:
                    # Product of (1 - previous_probs) * current_prob
                    prev_fail = torch.prod(1.0 - torch.stack(step_probs[:i]), dim=0)
                    first_success_prob = prev_fail * prob

                weighted_sum += torch.mean((i + 1) * first_success_prob).item()

            # Normalize by total success probability
            total_success_prob_mean = torch.mean(total_success_prob).item()
            if total_success_prob_mean > 1e-12:  # Avoid division by zero
                avg_steps_to_success = weighted_sum / total_success_prob_mean

        trajectory_data = {
            'step_probs': step_probs,
            'final_psi': psi,
            'total_success_prob': total_success_prob,
            'step_success_rates': step_success_rates,
            'avg_steps_to_success': avg_steps_to_success,
            'first_success_step': first_success_step,  # For two-stage optimization
            'success_occurred': success_occurred
        }

        return trajectory_data

    def collect_metrics(self, trajectory_data: dict) -> dict:
        """
        Collect and compute training metrics from trajectory data.

        Args:
            trajectory_data: Dictionary from generate_trajectory()

        Returns:
            metrics: Dictionary of computed metrics
        """
        metrics = {}

        # Basic success metrics - Return BOTH individual and average
        # total_success_prob has shape (B,) - success prob for each QA pair
        metrics['total_success_rate'] = trajectory_data['total_success_prob']  # Keep as tensor for individual QA rates
        metrics['individual_success_rates'] = trajectory_data['total_success_prob'].detach().cpu().tolist()
        metrics['average_success_rate'] = torch.mean(trajectory_data['total_success_prob']).item()
        metrics['final_success_prob'] = torch.mean(trajectory_data['total_success_prob']).item()
        metrics['avg_steps_to_success'] = trajectory_data['avg_steps_to_success']

        # Step-wise metrics
        step_success_rates = trajectory_data['step_success_rates']
        if len(step_success_rates) > 0:
            metrics['max_step_success_rate'] = max(step_success_rates)
            metrics['min_step_success_rate'] = min(step_success_rates)
            metrics['mean_step_success_rate'] = sum(step_success_rates) / len(step_success_rates)

        # Early vs late success analysis
        if len(step_success_rates) > 1:
            early_steps = step_success_rates[:len(step_success_rates)//2]
            late_steps = step_success_rates[len(step_success_rates)//2:]
            metrics['early_success_rate'] = sum(early_steps) / len(early_steps) if early_steps else 0.0
            metrics['late_success_rate'] = sum(late_steps) / len(late_steps) if late_steps else 0.0

        # Two-stage optimization metrics
        if 'first_success_step' in trajectory_data:
            first_success_steps = trajectory_data['first_success_step']
            success_occurred = trajectory_data['success_occurred']

            # Statistics about first success
            num_with_success = success_occurred.sum().item()
            metrics['fraction_with_success'] = num_with_success / len(success_occurred) if len(success_occurred) > 0 else 0.0

            if num_with_success > 0:
                # Average first success step (for those that succeeded)
                avg_first_success = first_success_steps[success_occurred].float().mean().item()
                metrics['avg_first_success_step'] = avg_first_success

                # Distribution of first success steps
                first_success_counts = torch.bincount(first_success_steps[success_occurred] + 1)[1:]  # +1 to handle -1
                metrics['first_success_distribution'] = first_success_counts.tolist()

        return metrics

    def print_metrics(self, metrics: dict):
        """Print training metrics in a readable format."""
        print(f"Training Metrics:")
        print(f"  Total Success Rate: {metrics['total_success_rate']:.4f}")
        print(f"  Avg Steps to Success: {metrics['avg_steps_to_success']:.2f}")

        if 'mean_step_success_rate' in metrics:
            print(f"  Mean Step Success Rate: {metrics['mean_step_success_rate']:.4f}")
            print(f"  Max Step Success Rate: {metrics['max_step_success_rate']:.4f}")
            print(f"  Min Step Success Rate: {metrics['min_step_success_rate']:.4f}")

        if 'early_success_rate' in metrics:
            print(f"  Early Success Rate: {metrics['early_success_rate']:.4f}")
            print(f"  Late Success Rate: {metrics['late_success_rate']:.4f}")

    def generate_batched_initial_states(self, start_nodes: list, end_nodes: list = None, *, initial_coin_state: str = "unique") -> torch.Tensor:
        """
        Generate batched initial quantum states from lists of start and end nodes.

        Args:
            start_nodes: List of starting node indices, length B
            end_nodes: List of target node indices, length B (required for "unique" mode)
            initial_coin_state: Type of initial coin state ("uniform", "random", "basis", "unique")
                              "unique" is the default, which creates QA-specific coin states

        Returns:
            initial_psi: Batched quantum states
                - Legacy mode (use_ar=False): shape (B, N, K)
                - AR mode (use_ar=True): shape (B, N, K, K, ..., K) with M copies of K
        """
        B = len(start_nodes)
        N, K = self.N, self.K

        # Validate parameters
        if initial_coin_state == "unique" and end_nodes is None:
            raise ValueError("end_nodes must be provided when using 'unique' initial_coin_state")

        if initial_coin_state == "unique" and len(end_nodes) != B:
            raise ValueError(f"end_nodes length {len(end_nodes)} doesn't match start_nodes length {B}")

        if self.use_ar:
            # AR mode: state has M coin dimensions (B, N, K, K, ..., K)
            M = self.max_steps
            state_shape = [B, N] + [K] * M
            initial_psi = torch.zeros(*state_shape, dtype=torch.complex64, device=self.device)

            for b, start_node in enumerate(start_nodes):
                if initial_coin_state == "uniform":
                    # Uniform superposition across ALL coin dimensions
                    # Each possible combination of coin states gets equal amplitude
                    # This ensures amplitude is available at each step
                    norm_factor = torch.sqrt(torch.tensor(K ** M, dtype=torch.complex64, device=self.device))

                    # Iterate over all K^M combinations of coin states
                    from itertools import product
                    for coin_states in product(range(K), repeat=M):
                        slice_idx = [b, start_node] + list(coin_states)
                        initial_psi[tuple(slice_idx)] = 1.0 / norm_factor

                elif initial_coin_state == "random":
                    # Random amplitudes across all coin dimensions
                    from itertools import product
                    for coin_states in product(range(K), repeat=M):
                        slice_idx = [b, start_node] + list(coin_states)
                        random_amp = torch.randn(1, dtype=torch.complex64, device=self.device)
                        initial_psi[tuple(slice_idx)] = random_amp

                    # Normalize
                    norm = torch.sqrt(torch.sum(torch.abs(initial_psi[b, start_node])**2))
                    initial_psi[b, start_node] = initial_psi[b, start_node] / norm

                elif initial_coin_state == "basis":
                    # All coin dimensions in basis state |0>
                    slice_idx = [b, start_node] + [0] * M
                    initial_psi[tuple(slice_idx)] = 1.0

                elif initial_coin_state == "unique":
                    # Generate unique coin state for this QA pair
                    # Get the coin state pattern from generate_unique_coin_state
                    coin_state_pattern = self.generate_unique_coin_state(N, K, start_node, end_nodes[b], max_length=M)

                    # Set the basis state according to the coin pattern
                    slice_idx = [b, start_node] + coin_state_pattern
                    initial_psi[tuple(slice_idx)] = 1.0
                else:
                    raise ValueError(f"Unknown initial_coin_state: {initial_coin_state}")
        else:
            # Legacy mode: state has shape (B, N, K)
            initial_psi = torch.zeros(B, N, K, dtype=torch.complex64, device=self.device)

            for b, start_node in enumerate(start_nodes):
                if initial_coin_state == "uniform":
                    # Uniform superposition over coin states
                    initial_psi[b, start_node, :] = torch.ones(K, dtype=torch.complex64, device=self.device) / torch.sqrt(torch.tensor(K, dtype=torch.complex64, device=self.device))
                elif initial_coin_state == "random":
                    # Random complex amplitudes
                    random_state = torch.randn(K, dtype=torch.complex64, device=self.device)
                    initial_psi[b, start_node, :] = random_state / torch.norm(random_state)
                elif initial_coin_state == "basis":
                    # First coin state (basis state)
                    initial_psi[b, start_node, 0] = 1.0

                elif initial_coin_state == "unique":
                    # For legacy mode, "unique" falls back to "basis"
                    # Since there's only one coin dimension, we can't use the full pattern
                    initial_psi[b, start_node, 0] = 1.0
                else:
                    raise ValueError(f"Unknown initial_coin_state: {initial_coin_state}")

        # Verify normalization: in both AR and legacy modes with B batch elements,
        # each state should be normalized (per-batch normalization)
        batch_size = initial_psi.shape[0]
        sum_dims = list(range(1, len(initial_psi.shape)))  # All dims except batch
        batch_norm = torch.sum(torch.abs(initial_psi)**2, dim=sum_dims)
        expected_norm = torch.ones(batch_size, device=self.device)
        assert torch.allclose(batch_norm, expected_norm, atol=1e-5), \
            f"Batch normalization failed: {batch_norm.tolist()}"

        return initial_psi

    def forward(self, initial_psi: torch.Tensor, target_node_idx, loss_type: str = "standard",
                return_metrics: bool = False, two_stage: bool = False, window_size: int = 5,
                linear_weight: float = 0.7):
        """
        Run quantum walk simulation and compute training loss.

        Args:
            initial_psi: Initial quantum state of shape (B, N, K)
            target_node_idx: Target node indices, shape (B,) for batch QA pairs, or int for single target
            loss_type: Type of loss function to use: "standard" or "log_prob" (default: "standard")
            return_metrics: Whether to return metrics along with loss
            two_stage: Whether to use two-stage optimization (default: False)
            window_size: Window size for truncated backprop in non-linear stage
            linear_weight: Weight for linear stage loss in two-stage optimization

        Returns:
            If return_metrics=False:
                loss: Negative mean reward (to minimize), shape ()
            If return_metrics=True:
                (loss, metrics) tuple
        """
        # Phase 1: Generate trajectory and collect data
        trajectory_data = self.generate_trajectory(initial_psi, target_node_idx)

        # Phase 2: Collect metrics (can be extended for more complex analysis)
        metrics = self.collect_metrics(trajectory_data)

        # Phase 3: Compute loss
        if two_stage:
            # Use two-stage optimization with linear and non-linear stages
            loss = self.compute_two_stage_loss(trajectory_data, window_size, linear_weight, loss_type)
            metrics['two_stage'] = True
            metrics['linear_weight'] = linear_weight
            metrics['window_size'] = window_size
        else:
            # Use standard loss computation
            loss = self.compute_trajectory_loss(trajectory_data['step_probs'], loss_type)
            metrics['two_stage'] = False

        if return_metrics:
            return loss, metrics
        else:
            return loss

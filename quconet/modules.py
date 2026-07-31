"""
Core quantum modules for QuCoNet - Quantum Concept Network

This module implements the fundamental components of a discrete-time quantum walk on graphs:
1. CoinOperator: Learnable node-specific coin operations
2. ShiftOperator: Unitary permutation based on graph structure
3. QuantumCoNet: Main model that runs the quantum walk simulation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


def optimize_model_for_device(model: nn.Module, device: torch.device) -> nn.Module:
    """
    Optimize model for the target device using available extensions.

    Args:
        model: PyTorch model to optimize
        device: Target device

    Returns:
        Optimized model
    """
    if device.type == 'cuda':
        # Use standard PyTorch CUDA optimizations
        model = model.to(device)
    else:
        # CPU device - no special optimization needed
        model = model.to(device)

    return model


def get_optimal_device(device: Optional[torch.device] = None) -> torch.device:
    """
    Automatically select the optimal available device.

    Priority order: CUDA → CPU

    Args:
        device: Optional specific device to use. If None, auto-select optimal device.

    Returns:
        torch.device: The selected device
    """
    if device is not None:
        return device

    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


class CoinOperator(nn.Module):
    """
    Applies learnable, node-specific coin operations for quantum walks.

    The coin operator Cx = exp(i*Hx) where Hx is a learnable real symmetric
    Hamiltonian matrix. This ensures Cx is unitary while allowing gradient-based
    optimization of the coin parameters.
    """

    def __init__(self, N: int, K: int, device: Optional[torch.device] = None, init_scale: float = None):
        """
        Initialize the CoinOperator.

        Args:
            N: Number of nodes in the graph
            K: Degree of each node (coin dimension)
            device: Device to place tensors on. If None, auto-select optimal device.
            init_scale: Optional scale factor. If None, uses optimized spectral initialization (π/K)
        """
        super().__init__()
        self.N = N
        self.K = K
        self.device = get_optimal_device(device)

        # Number of unique parameters for symmetric K×K matrix: K*(K+1)//2
        self.num_params_per_node = K * (K + 1) // 2

        # Learnable parameters for all nodes: (N, K*(K+1)//2)
        # These represent the upper triangle of symmetric Hamiltonians
        # Use uniform spectral initialization: U(-π/K, π/K) for good mixing
        # This ensures the spectral radius is matched to the Hilbert space dimension
        if init_scale is None:
            # Default: Optimized spectral initialization
            spectral_bound = torch.pi / self.K
            self.hamiltonian_params = nn.Parameter(
                torch.empty(N, self.num_params_per_node, device=self.device)
            )
            nn.init.uniform_(self.hamiltonian_params, -spectral_bound, spectral_bound)
        else:
            # Custom initialization with user-specified scale
            self.hamiltonian_params = nn.Parameter(
                torch.empty(N, self.num_params_per_node, device=self.device)
            )
            nn.init.uniform_(self.hamiltonian_params, -init_scale, init_scale)

        # Pre-compute upper triangle indices for efficient matrix construction
        triu_indices = torch.triu_indices(K, K, offset=0, device=self.device)
        self.register_buffer('triu_indices', triu_indices)

    def build_hamiltonian_batch(self, batch_size: int) -> torch.Tensor:
        """
        Build batch of symmetric Hamiltonian matrices from learnable parameters.

        Args:
            batch_size: Batch size for the output tensors

        Returns:
            H_batch: Hamiltonian matrices of shape (B, N, K, K)
        """
        # Create empty Hamiltonian tensors: (B, N, K, K)
        H_batch = torch.zeros(
            batch_size, self.N, self.K, self.K,
            dtype=self.hamiltonian_params.dtype,
            device=self.device
        )

        # Scatter parameters into upper triangle for each node
        # hamiltonian_params: (N, num_params)
        for node_idx in range(self.N):
            node_params = self.hamiltonian_params[node_idx]  # (num_params,)
            # Fill upper triangle
            H_batch[:, node_idx, self.triu_indices[0], self.triu_indices[1]] = node_params

        # Make symmetric: H = H + H^T - diag(H) (to avoid double-counting diagonal)
        H_batch = H_batch + H_batch.transpose(-1, -2)

        # Subtract diagonal elements that were double-counted
        diag_mask = torch.eye(self.K, dtype=torch.bool, device=self.device)
        H_batch[:, :, diag_mask] = H_batch[:, :, diag_mask] / 2

        return H_batch

    def build_coin_operators(self, H_batch: torch.Tensor) -> torch.Tensor:
        """
        Build unitary coin operators C = exp(i*H) from Hamiltonian matrices.

        Uses torch.matrix_exp for numerical stability and better backward pass behavior.

        Args:
            H_batch: Hamiltonian matrices of shape (B, N, K, K)

        Returns:
            C_batch: Unitary coin operators of shape (B, N, K, K)
        """
        # Ensure H_batch is on the correct device and has correct dtype
        H_batch = H_batch.to(self.device)

        # Convert to complex for matrix exponential
        H_batch_complex = H_batch.to(torch.complex64)

        # Build coin operators: C = exp(i*H)
        C_batch = torch.matrix_exp(1j * H_batch_complex)  # (B, N, K, K)

        return C_batch

    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        """
        Apply coin operation to quantum state.

        Args:
            psi: Input quantum state of shape (B, N, K)

        Returns:
            psi_prime: Quantum state after coin operation, shape (B, N, K)
        """
        # Ensure psi is on the correct device
        psi = psi.to(self.device)

        B, N, K = psi.shape
        assert N == self.N and K == self.K, f"Expected psi shape (B, {self.N}, {self.K}), got {psi.shape}"

        # Build Hamiltonian batch: (B, N, K, K)
        H_batch = self.build_hamiltonian_batch(B)

        # Build coin operators using matrix exponential
        C_batch = self.build_coin_operators(H_batch)  # (B, N, K, K)

        # Reshape psi for batch matrix multiplication: (B, N, K) -> (B*N, K, 1)
        psi_reshaped = psi.reshape(B * N, K, 1)

        # Reshape C_batch for batch multiplication: (B, N, K, K) -> (B*N, K, K)
        C_batch_reshaped = C_batch.reshape(B * N, K, K)

        # Apply coin operations: |psi'> = C @ |psi>
        psi_prime_reshaped = torch.bmm(C_batch_reshaped, psi_reshaped)  # (B*N, K, 1)

        # Reshape back: (B*N, K, 1) -> (B, N, K)
        psi_prime = psi_prime_reshaped.reshape(B, N, K)

        return psi_prime


class ShiftOperator(nn.Module):
    """
    Applies the unitary shift (permutation) operation for quantum walks.

    The shift operator maps |x,i⟩ → |y,j⟩ where y is the i-th neighbor of x,
    and j is the coin index that maps back to x. This creates a unitary
    permutation that preserves quantum amplitudes.
    """

    def __init__(self, N: int, K: int, adjacency_list: List[List[int]], device: Optional[torch.device] = None):
        """
        Initialize the ShiftOperator.

        Args:
            N: Number of nodes in the graph
            K: Degree of each node
            adjacency_list: Adjacency list where adj_list[x][i] gives i-th neighbor of x
            device: Device to place tensors on. If None, auto-select optimal device.
        """
        super().__init__()
        self.N = N
        self.K = K
        self.device = get_optimal_device(device)

        # Build shift maps for unitary permutation
        shift_node_map = torch.zeros(N, K, dtype=torch.long, device=self.device)
        shift_coin_map = torch.zeros(N, K, dtype=torch.long, device=self.device)

        for node_x in range(N):
            neighbors = adjacency_list[node_x]
            assert len(neighbors) == K, f"Node {node_x} has {len(neighbors)} neighbors, expected {K}"

            for coin_i, node_y in enumerate(neighbors):
                # Validate that node_y is within bounds
                if node_y < 0 or node_y >= N:
                    raise ValueError(f"Node {node_y} (neighbor of {node_x}) is out of bounds [0, {N-1}]")

                # Find the coin index j that maps back from y to x
                neighbors_of_y = adjacency_list[node_y]
                try:
                    coin_j = neighbors_of_y.index(node_x)
                except ValueError:
                    raise ValueError(f"Node {node_y} (neighbor of {node_x}) doesn't have {node_x} as neighbor")

                shift_node_map[node_x, coin_i] = node_y
                shift_coin_map[node_x, coin_i] = coin_j

        # Register as buffers (non-learnable parameters)
        self.register_buffer('shift_node_map', shift_node_map)
        self.register_buffer('shift_coin_map', shift_coin_map)

    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        """
        Apply shift operation to quantum state.

        Args:
            psi: Input quantum state of shape (B, N, K)

        Returns:
            psi_prime: Quantum state after shift operation, shape (B, N, K)
        """
        # Ensure psi is on the correct device
        psi = psi.to(self.device)

        B, N, K = psi.shape
        assert N == self.N and K == self.K, f"Expected psi shape (B, {self.N}, {self.K}), got {psi.shape}"

        # Flatten psi: (B, N, K) -> (B*N*K,)
        psi_flat = psi.reshape(B * N * K)

        # Create output tensor
        psi_prime_flat = torch.zeros_like(psi_flat)

        # Compute destination indices for the permutation
        # For each |x,i⟩ state, find its destination |y,j⟩
        for x in range(N):
            for i in range(K):
                src_idx = x * K + i  # Flat index for |x,i⟩

                # Get destination: |y,j⟩
                y = self.shift_node_map[x, i].item()
                j = self.shift_coin_map[x, i].item()
                dst_idx = y * K + j  # Flat index for |y,j⟩

                # Copy amplitude (unitary permutation)
                for b in range(B):
                    src_flat_idx = b * N * K + src_idx
                    dst_flat_idx = b * N * K + dst_idx
                    psi_prime_flat[dst_flat_idx] = psi_flat[src_flat_idx]

        # Reshape back: (B*N*K,) -> (B, N, K)
        psi_prime = psi_prime_flat.reshape(B, N, K)

        return psi_prime


class QuantumCoNet(nn.Module):
    """
    Main QuCoNet model that runs quantum walk simulation for graph traversal.

    This module implements the full quantum walk algorithm:
    1. Initialize quantum state at starting node
    2. Evolve state through alternating coin and shift operations
    3. Measure success probability at target node each step
    4. Calculate total reward as success probability over all steps
    """

    def __init__(self, N: int, K: int, adjacency_list: List[List[int]], max_steps: int, device: Optional[torch.device] = None, init_scale: float = 1.0):
        """
        Initialize the QuantumCoNet model.

        Args:
            N: Number of nodes in the graph
            K: Degree of each node
            adjacency_list: Adjacency list for graph structure
            max_steps: Maximum number of quantum walk steps
            device: Device to place tensors on. If None, auto-select optimal device.
            init_scale: Scale factor for Hamiltonian parameter initialization (default: 1.0)
        """
        super().__init__()
        self.N = N
        self.K = K
        self.max_steps = max_steps
        self.device = get_optimal_device(device)
        self.init_scale = init_scale

        # Initialize quantum operators
        self.coin = CoinOperator(N, K, device=self.device, init_scale=init_scale)
        self.shift = ShiftOperator(N, K, adjacency_list, device=self.device)

    def quantum_step(self, psi: torch.Tensor, step: int = 1) -> torch.Tensor:
        """
        Perform one step of quantum walk: coin operation + shift operation.

        Args:
            psi: Current quantum state of shape (B, N, K)
            step: Step number (ignored in legacy mode, for API compatibility)

        Returns:
            psi_next: Quantum state after one step, shape (B, N, K)
        """
        # Apply coin operation
        psi = self.coin(psi)

        # Apply shift operation
        psi = self.shift(psi)

        return psi

    def measure_and_project(self, psi: torch.Tensor, target_node_idx: torch.Tensor):
        """
        Measure success probability at target nodes for each batch element and project state.

        Args:
            psi: Current quantum state of shape (B, N, K)
            target_node_idx: Target node indices for each batch element, shape (B,)

        Returns:
            success_prob: Success probability at this step for each batch element, shape (B,)
            psi_projected: Projected quantum state, shape (B, N, K)
        """
        B, N, K = psi.shape
        assert target_node_idx.shape == (B,), f"target_node_idx shape {target_node_idx.shape} doesn't match batch size {B}"

        # Get amplitude at target nodes for each batch element
        # Use advanced indexing to get psi[b, target_node_idx[b], :] for each b
        batch_indices = torch.arange(B, device=self.device)
        psi_at_target = psi[batch_indices, target_node_idx, :]  # (B, K)

        # Calculate success probability: p_n = sum(|amplitude|^2 over coin dimension)
        success_prob = torch.sum(torch.abs(psi_at_target)**2, dim=-1)  # (B,)

        # Project to unfinished space: zero out target node amplitude for each batch element
        psi_projected = psi.clone()
        psi_projected[batch_indices, target_node_idx, :] = 0.0

        # Calculate remaining probability for each batch element
        p_unf = torch.sum(torch.abs(psi_projected)**2, dim=(-1, -2))  # (B,)

        # CRITICAL: Clamp for numerical stability
        p_unf_stable = torch.clamp(p_unf, min=1e-12)

        # Normalize projected state for each batch element
        psi_projected = psi_projected / torch.sqrt(p_unf_stable).view(-1, 1, 1)

        return success_prob, psi_projected

    def compute_standard_loss(self, step_probs: list) -> torch.Tensor:
        """Compute standard loss: R = 1 - prod(1 - p_n) over all steps."""
        p_n_tensor = torch.stack(step_probs, dim=1)  # (B, max_steps)
        p_fail = torch.prod(1.0 - p_n_tensor, dim=1)  # (B,)
        R = 1.0 - p_fail  # (B,)
        return -torch.mean(R)

    def compute_log_prob_loss(self, step_probs: list) -> torch.Tensor:
        """Compute improved log probability loss: log(prod(1-p_n)) = sum(log(1-p_n)).

        This provides better gradient behavior and naturally decomposes to stepwise form
        under the log transformation, making it more interpretable for optimization.

        The loss is: -sum(log(1-p_n)) which encourages maximizing p_n (success probabilities)
        """
        p_n_tensor = torch.stack(step_probs, dim=1)  # (B, max_steps)
        # Compute log of (1 - p_n) for each step, then sum across steps
        log_one_minus_p = torch.log(1.0 - p_n_tensor + 1e-12)  # Add epsilon for stability
        stepwise_log_loss = torch.sum(log_one_minus_p, dim=1)  # (B,) - sum across steps
        return torch.mean(stepwise_log_loss)  # Positive value for gradient ascent (we want to minimize this)

    # Removed compute_stepwise_loss as the current implementation lacks natural justification
    # Future stepwise loss will be implemented with better theoretical foundation

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
        else:
            raise ValueError(f"Unknown loss type: {loss_type}. Available: standard, log_prob")

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
            psi = self.quantum_step(psi)

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
        avg_steps_to_success = 0.0
        if len(step_probs) > 0:
            # Weighted average where weights are the probability of first success at each step
            for i, prob in enumerate(step_probs):
                # Probability of first success at step i+1
                if i == 0:
                    first_success_prob = prob
                else:
                    # Product of (1 - previous_probs) * current_prob
                    prev_fail = torch.prod(1.0 - torch.stack(step_probs[:i]), dim=0)
                    first_success_prob = prev_fail * prob

                avg_steps_to_success += torch.mean((i + 1) * first_success_prob).item()

        trajectory_data = {
            'step_probs': step_probs,
            'final_psi': psi,
            'total_success_prob': total_success_prob,
            'step_success_rates': step_success_rates,
            'avg_steps_to_success': avg_steps_to_success / (torch.mean(total_success_prob).item() + 1e-100),
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

    def generate_batched_initial_states(self, start_nodes: list, end_nodes: list = None, *, initial_coin_state: str = "uniform") -> torch.Tensor:
        """
        Generate batched initial quantum states from lists of start and end nodes.

        Args:
            start_nodes: List of starting node indices, length B
            end_nodes: List of target node indices, length B (ignored in legacy mode)
            initial_coin_state: Type of initial coin state ("uniform", "random", "basis")
                              Note: "unique" mode not supported in legacy mode, falls back to "basis"

        Returns:
            initial_psi: Batched quantum states of shape (B, N, K)
        """
        B = len(start_nodes)
        N, K = self.N, self.K

        # Create initial quantum states
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
                # Legacy mode doesn't support unique initialization, fall back to basis
                initial_psi[b, start_node, 0] = 1.0
            else:
                raise ValueError(f"Unknown initial_coin_state: {initial_coin_state}")

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
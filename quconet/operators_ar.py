"""
AR version of quantum operators for QuCoNet - CoinOperator and ShiftOperator

This module contains:
1. CoinOperator: Learnable node-specific coin operations with AR implementation
2. ShiftOperator: Unitary permutation based on graph structure with AR support
"""

import torch
import torch.nn as nn
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


class CoinOperatorAR(nn.Module):
    """
    Applies learnable, node-specific coin operations for quantum walks.

    The coin operator Cx = exp(i*Hx) where Hx is a learnable real symmetric
    Hamiltonian matrix. This ensures Cx is unitary while allowing gradient-based
    optimization of the coin parameters.
    """

    def __init__(self, N: int, K: int, device: Optional[torch.device] = None, init_scale: Optional[float] = None):
        """
        Initialize the CoinOperator.

        Args:
            N: Number of nodes in the graph
            K: Degree of each node (coin dimension)
            device: Device to place tensors on. If None, auto-select optimal device.
            init_scale: Optional scale factor for random perturbations around flat coin.
                       If None, uses π/K. Small values give near-flat initialization.
        """
        super().__init__()
        self.N = N
        self.K = K
        self.device = get_optimal_device(device)

        # Number of unique parameters for symmetric K×K matrix: K*(K+1)//2
        self.num_params_per_node = K * (K + 1) // 2

        # Compute base Hamiltonian for flat coin initialization
        # This ensures C = exp(i*H) has all elements with similar magnitude ~1/sqrt(K)
        # instead of being close to identity (which kills 2 out of 3 routes)
        base_hamiltonian = self._compute_flat_coin_hamiltonian(K)  # (K, K)
        
        # Extract upper triangle of base Hamiltonian as starting point
        triu_indices = torch.triu_indices(K, K, offset=0, device=self.device)
        # Move base_hamiltonian to target device before indexing
        base_hamiltonian = base_hamiltonian.to(self.device)
        base_params = base_hamiltonian[triu_indices[0], triu_indices[1]]  # (num_params_per_node,)
        
        # Initialize learnable parameters as base + small random perturbations
        # When init_scale is small, we get something close to flat coin
        if init_scale is None:
            init_scale = torch.pi / self.K
            
        self.hamiltonian_params = nn.Parameter(
            torch.empty(N, self.num_params_per_node, device=self.device)
        )
        
        # Expand base_params to (N, num_params_per_node) and add small perturbations
        base_params_expanded = base_params.unsqueeze(0).expand(N, -1)  # (N, num_params_per_node)
        perturbations = torch.empty_like(self.hamiltonian_params)
        nn.init.uniform_(perturbations, -init_scale, init_scale)
        self.hamiltonian_params.data = base_params_expanded + perturbations

        # Pre-compute upper triangle indices for efficient matrix construction
        self.register_buffer('triu_indices', triu_indices)

    @staticmethod
    def _compute_flat_coin_hamiltonian(K: int) -> torch.Tensor:
        """
        Compute the Hamiltonian H that generates a flat coin operator C = exp(iH).
        
        A flat coin has all elements with similar magnitude ~1/sqrt(K), enabling
        equal superposition across all routes. This is achieved by using the
        DFT matrix as the target unitary and extracting its Hermitian logarithm.
        
        Args:
            K: Coin dimension
            
        Returns:
            H: Real symmetric Hamiltonian matrix of shape (K, K)
        """
        # Generate DFT matrix: W[m,n] = exp(-2j*pi*m*n/K) / sqrt(K)
        # This is a flat unitary with all elements having magnitude 1/sqrt(K)
        m, n = torch.meshgrid(torch.arange(K), torch.arange(K), indexing='ij')
        omega = torch.exp(torch.tensor(-2j * torch.pi / K, dtype=torch.complex64))
        W = (omega ** (m * n)) / torch.sqrt(torch.tensor(K, dtype=torch.complex64))
        
        # Compute H such that W = exp(i*H) by taking matrix logarithm
        # H = -i * logm(W), then extract real symmetric part
        # For numerical stability, use eigendecomposition: W = V*diag(eig)*V^H
        # Then logm(W) = V*diag(log(eig))*V^H
        eigvals, eigvecs = torch.linalg.eig(W)  # Use eig since DFT is not Hermitian
        log_eigvals = torch.log(eigvals)
        # Create diagonal matrix with complex dtype
        diag_matrix = torch.zeros((K, K), dtype=torch.complex64, device=log_eigvals.device)
        diag_matrix[torch.arange(K), torch.arange(K)] = log_eigvals.to(torch.complex64)
        log_W = eigvecs @ diag_matrix @ torch.linalg.inv(eigvecs)
        
        # H = log_W / 1j, then take real part and symmetrize
        H_complex = log_W / 1j
        H = H_complex.real
        H = (H + H.T) / 2  # Ensure strict symmetry
        
        return H

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

    def forward_legacy(self, psi: torch.Tensor, step: int = 1) -> torch.Tensor:
        """
        Apply coin operation to quantum state in AR mode (legacy implementation).

        In AR mode, psi can have shape (B, N, K) or (B, N, K, K, ..., K) with M copies of K dimension.
        For step i, we apply the coin operator to the i-th K dimension.

        This is the original implementation with permute/reshape operations. For memory efficiency,
        use forward() which calls forward_einsum() without these operations.

        Args:
            psi: Input quantum state. Can be:
                 - Legacy shape: (B, N, K)
                 - AR shape: (B, N, K, K, ..., K) with M copies of K
            step: Which coin state to act on (1-indexed, default: 1)

        Returns:
            psi_prime: Quantum state after coin operation, same shape as input
        """
        # Ensure psi is on the correct device
        psi = psi.to(self.device)

        # Get shape and validate
        shape = psi.shape
        B, N = shape[0], shape[1]

        # Calculate M (number of coin dimensions)
        coin_dims = shape[2:]
        M = len(coin_dims)

        # Validate that all coin dimensions have size K
        for dim_size in coin_dims:
            assert dim_size == self.K, f"All coin dimensions should have size {self.K}, got {dim_size}"

        # Validate step index
        step_idx = step - 1  # Convert to 0-indexed
        assert 0 <= step_idx < M, f"Step {step} is out of range for {M} coin dimensions"

        # Build coin operators
        H_batch = self.build_hamiltonian_batch(B)
        C_batch = self.build_coin_operators(H_batch)  # (B, N, K, K)

        # Reshape C_batch: (B, N, K, K) -> (B*N, K, K)
        C_batch_reshaped = C_batch.reshape(B * N, self.K, self.K)

        # Permute dimensions to move the target K dimension to the end
        # Original: (B, N, K, K, ..., K)
        # Target position is step_idx in dimensions [2:]
        permute_dims = list(range(len(shape)))
        # Remove the target dimension and append it at the end
        target_dim = 2 + step_idx
        permute_dims.pop(target_dim)
        permute_dims.append(target_dim)

        psi_permuted = psi.permute(permute_dims)
        # Now shape: (B, N, K, ..., K, K) where the last K is the acted-on dimension

        # Calculate the size of the flattened dimension for all other coin states
        # Product of all K dimensions except the target one
        other_K_product = self.K ** (M - 1)

        # Reshape: (B, N, K, ..., K, K) -> (B * N * other_K_product, K, 1)
        psi_reshaped = psi_permuted.reshape(B * N * other_K_product, self.K, 1)

        # Apply coin operations: |psi'> = C @ |psi>
        # Need to handle batching: C_batch_reshaped is (B*N, K, K)
        # psi_reshaped is (B*N*other_K_product, K, 1)

        # For each batch and node, we need to apply the same C to multiple states
        # Reshape psi to make batching work: (B*N, other_K_product, K, 1)
        psi_grouped = psi_reshaped.reshape(B * N, other_K_product, self.K, 1)

        # Apply C to each group: (B*N, other_K_product, K, 1) with (B*N, K, K)
        # We can use einsum for clarity
        psi_applied = torch.einsum('bkl,bxlc->bxkc', C_batch_reshaped, psi_grouped)

        # Reshape back: (B*N, other_K_product, K, 1) -> (B*N*other_K_product, K, 1)
        psi_result = psi_applied.reshape(B * N * other_K_product, self.K, 1)

        # Reshape back to permuted shape: (B, N, K, ..., K, K)
        psi_permuted_result = psi_result.reshape(*psi_permuted.shape)

        # Inverse permutation: move the last dimension back to its original position
        inverse_permute_dims = [0] * len(shape)
        for i, dim in enumerate(permute_dims):
            inverse_permute_dims[dim] = i

        psi_prime = psi_permuted_result.permute(inverse_permute_dims)

        return psi_prime

    def forward(self, psi: torch.Tensor, step: int = 1) -> torch.Tensor:
        """
        Apply coin operation to quantum state in AR mode (optimized implementation).

        This method uses the optimized forward_einsum implementation without permute/reshape operations
        for better memory efficiency. For backward compatibility, the legacy implementation is available
        as forward_legacy().

        Args:
            psi: Input quantum state. Can be:
                 - Legacy shape: (B, N, K)
                 - AR shape: (B, N, K, K, ..., K) with M copies of K
            step: Which coin state to act on (1-indexed, default: 1)

        Returns:
            psi_prime: Quantum state after coin operation, same shape as input
        """
        return self.forward_einsum(psi, step)

    def forward_einsum(self, psi: torch.Tensor, step: int = 1) -> torch.Tensor:
        """
        Optimized in-place coin operation using einsum directly without permute/reshape.

        This version uses torch.einsum to apply the coin operator directly to the
        desired dimension without creating intermediate permuted/reshaped tensors.

        Args:
            psi: Input quantum state (modified in-place)
            step: Which coin state to act on (1-indexed)

        Returns:
            psi: Modified quantum state (same tensor, modified in-place)
        """
        # Build coin operators (same computation as forward)
        B, N = psi.shape[0], psi.shape[1]
        M = len(psi.shape) - 2  # Number of coin dimensions
        step_idx = step - 1

        # Validate
        assert 0 <= step_idx < M, f"Step {step} is out of range for {M} coin dimensions"
        assert all(psi.shape[i+2] == self.K for i in range(M)), f"All coin dimensions should have size {self.K}"

        H_batch = self.build_hamiltonian_batch(B)
        C_batch = self.build_coin_operators(H_batch)  # (B, N, K, K)

        # Use einsum directly on the original tensor
        # We need to construct an einsum equation that targets the specific dimension

        # Build einsum string dynamically
        # For psi with shape (B, N, K, K, ..., K) and step_idx,
        # We want: output[b,n,...,i,...] = sum(j) C[b,n,i,j] * psi[b,n,...,j,...]
        # where i,j replace the dimension at step_idx

        # Get letters for dimensions
        # We'll use: b for batch, n for node, then letters for K dimensions
        chars = [chr(ord('a') + i) for i in range(26)]  # a, b, c, d, e, f, g, ...

        # C indices: b (batch), n (node), i (output coin), j (input coin)
        C_idx = [chars[0], chars[1], chars[2], chars[3]]  # b, n, i, j

        # Psi indices: b, n, plus M K dimensions
        psi_idx = [C_idx[0], C_idx[1]]  # Start with b, n
        output_idx = [C_idx[0], C_idx[1]]  # Start with b, n

        # dim_chars will store the letters for each K dimension
        dim_chars = []
        char_idx = 4  # Start after b, n, i, j (which use chars 0-3)

        for d in range(M):
            dim_char = chars[char_idx]
            dim_chars.append(dim_char)
            char_idx += 1

            if d == step_idx:
                # This is the dimension we're contracting
                # Input side uses j (from C), output side uses i (from C)
                psi_idx.append(C_idx[3])  # j (input coin index)
                output_idx.append(C_idx[2])  # i (output coin index)
            else:
                # Other dimensions just pass through
                psi_idx.append(dim_char)
                output_idx.append(dim_char)

        # Build einsum string: "C_indices,psi_indices->output_indices"
        einsum_str = f"{''.join(C_idx)},{''.join(psi_idx)}->{''.join(output_idx)}"

        # Apply coin operation using einsum directly on psi
        psi_prime = torch.einsum(einsum_str, C_batch, psi)

        return psi_prime


class ShiftOperatorAR(nn.Module):
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

        # Store adjacency list for use in forward pass
        self.adjacency_list = adjacency_list

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

    def forward(self, psi: torch.Tensor, step: int = 1) -> torch.Tensor:
        """
        Apply shift operation to quantum state using optimized torch operations.

        The shift maps |x, i⟩ → |y, j⟩ where y is the i-th neighbor of x,
        and j is the coin index that maps back to x.

        Args:
            psi: Input quantum state (B, N, K) for legacy or (B, N, K, K, ..., K) for AR
            step: Which coin dimension to use for shifting (1-indexed, default: 1)

        Returns:
            psi_prime: Quantum state after shift operation, same shape as input
        """
        # Ensure psi is on the correct device
        psi = psi.to(self.device)

        # Get shape
        shape = psi.shape
        B, N = shape[0], shape[1]
        M = len(shape) - 2  # Number of coin dimensions
        step_idx = step - 1  # 0-indexed

        # For legacy mode (M=1): (B, N, K) - use simple implementation
        if M == 1:
            return self._forward_legacy(psi)

        # For AR mode (M > 1): use optimized torch implementation
        # Shape: (B, N, K, K, ..., K) - use optimized implementation
        return self._forward_ar_optimized(psi, step)

    def _forward_legacy(self, psi):
        """Forward for legacy mode (B, N, K) - simple for-loop implementation"""
        B, N, K = psi.shape

        # Create output
        psi_prime = torch.zeros_like(psi)

        # Direct for-loop over batch, nodes, and coin states
        for b in range(B):
            for node_x in range(N):
                for coin_i in range(K):
                    amplitude = psi[b, node_x, coin_i]

                    if abs(amplitude) > 0:
                        # Get destination using shift maps
                        node_y = self.shift_node_map[node_x, coin_i].item()
                        coin_j = self.shift_coin_map[node_x, coin_i].item()

                        # Assign to destination
                        psi_prime[b, node_y, coin_j] = amplitude

        return psi_prime

    def _forward_ar_optimized(self, psi, step):
        """Optimized forward for AR mode using torch operations"""
        shape = psi.shape
        B, N = shape[0], shape[1]
        M = len(shape) - 2
        step_idx = step - 1

        # Step 1: Move N and the target K dimension to the last two slots
        # Current: (B, N, K1, K2, ..., Kstep, ..., KM)
        # Target:  (B, K1, ..., KM (excl Kstep), N, Kstep)
        dims = list(range(psi.dim()))
        # Remove N (dim 1) and Kstep (dim step+2) from their positions
        rest_dims = [d for d in dims if d not in (1, step_idx + 2)]
        # Create new order: rest_dims + [N, Kstep]
        permute_order = [dims[0]] + rest_dims[1:] + [1, step_idx + 2]
        psi_permuted = psi.permute(*permute_order)

        # Step 2: Get the shift maps as indexing tensors
        # shift_node_map gives destination node for each (node, coin)
        # shift_coin_map gives destination coin for each (node, coin)
        target_N = self.shift_node_map  # Shape: (N, K)
        target_K = self.shift_coin_map  # Shape: (N, K)

        # Step 3: Apply shift using advanced indexing
        # This performs the permutation: |x, i⟩ → |y, j⟩ where y = target_N[x, i], j = target_K[x, i]
        shifted_psi = psi_permuted[..., target_N, target_K]

        # Step 4: Permute back to original shape
        # Move N back to position 1 and Kstep back to position step+2
        res = shifted_psi.movedim(-2, 1)  # Move N from -2 to 1
        res = res.movedim(-1, step_idx + 2)  # Move Kstep from -1 to step+2

        return res


def create_undirected_regular_graph(N, K):
    """Create a proper undirected K-regular graph"""
    adjacency_list = [[] for _ in range(N)]
    half_k = K // 2
    for i in range(N):
        for j in range(1, half_k + 1):
            neighbor = (i + j) % N
            adjacency_list[i].append(neighbor)
            adjacency_list[neighbor].append(i)
    if K % 2 == 1:
        if N % 2 == 0:
            for i in range(N // 2):
                opposite = (i + N // 2) % N
                adjacency_list[i].append(opposite)
                adjacency_list[opposite].append(i)
        else:
            for i in range(N):
                opposite = (i + N // 2 + 1) % N
                if opposite not in adjacency_list[i]:
                    adjacency_list[i].append(opposite)
                    adjacency_list[opposite].append(i)
    for i in range(N):
        adjacency_list[i] = sorted(list(set(adjacency_list[i])))
        if len(adjacency_list[i]) != K:
            raise ValueError(f"Node {i} has {len(adjacency_list[i])} neighbors, expected {K}")
    return adjacency_list



def create_undirected_regular_graph(N, K):
    """Create a proper undirected K-regular graph"""
    adjacency_list = [[] for _ in range(N)]
    half_k = K // 2
    for i in range(N):
        for j in range(1, half_k + 1):
            neighbor = (i + j) % N
            adjacency_list[i].append(neighbor)
            adjacency_list[neighbor].append(i)
    if K % 2 == 1:
        if N % 2 == 0:
            for i in range(N // 2):
                opposite = (i + N // 2) % N
                adjacency_list[i].append(opposite)
                adjacency_list[opposite].append(i)
        else:
            for i in range(N):
                opposite = (i + N // 2 + 1) % N
                if opposite not in adjacency_list[i]:
                    adjacency_list[i].append(opposite)
                    adjacency_list[opposite].append(i)
    for i in range(N):
        adjacency_list[i] = sorted(list(set(adjacency_list[i])))
        if len(adjacency_list[i]) != K:
            raise ValueError(f"Node {i} has {len(adjacency_list[i])} neighbors, expected {K}")
    return adjacency_list


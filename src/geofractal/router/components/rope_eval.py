import torch
import matplotlib.pyplot as plt
import seaborn as sns
from geofractal.router.components.rope_component import CantorRoPE


def visualize_cantor_topology():
    """
    Visualizes the raw topology of the CantorRoPE embedding space.

    If the architecture is correct, the self-attention matrix should NOT
    show linear distance decay (a diagonal band).

    It should show a fractal block structure, proving that the system
    organizes information by Branch Path, not by Array Index.
    """
    seq_len = 10240  # Sufficient length to see recursive structure
    head_dim = 64
    levels = 5

    print("Generating Topological Manifold...")

    # 1. Initialize CantorRoPE in pure path_encoding mode
    # We turn off 'hybrid' to see the naked topology of the architecture
    rope = CantorRoPE(
        'topology_test',
        head_dim=head_dim,
        levels=levels,
        mode='path_encoding',  # PURE FRACTAL MODE
        max_seq_len=seq_len
    )

    # 2. Generate the Rotational Embeddings
    # We don't need Q/K vectors here. The topology is defined by the
    # dot product of the Rotary Embeddings themselves.
    # The cosine similarity of the positions defines the "geometry" of the space.
    cos, sin = rope.embed(seq_len)

    # Construct the complex form for easier correlation analysis
    # pos_vectors[i] = cos[i] + j*sin[i]
    # We treat the embedding dimension as a set of frequencies.
    # We want to see: How similar is the rotation at i to the rotation at j?

    # Flatten the head dim to treat the whole rotation state as a single vector
    pos_state = torch.cat([cos, sin], dim=-1)  # (L, 2D)

    # Normalize to visualize pure directional alignment
    pos_state = torch.nn.functional.normalize(pos_state, p=2, dim=-1)

    # 3. Compute the Manifold Similarity Matrix
    # S[i, j] = 1.0 means i and j are topologically identical
    similarity_matrix = torch.matmul(pos_state, pos_state.T)  # (L, L)

    # 4. Visualization
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        similarity_matrix.detach().numpy(),
        cmap="magma",
        xticklabels=False,
        yticklabels=False,
        square=True,
        cbar_kws={'label': 'Rotational Alignment'}
    )

    plt.title(f"CantorRoPE Topology (L={seq_len}, Levels={levels})")
    plt.xlabel("Array Index j")
    plt.ylabel("Array Index i")

    print("\nInterpreting the visual output:")
    print("1. Standard RoPE would show a thin diagonal yellow line (Locality).")
    print("2. CantorRoPE should show recursive square blocks.")
    print("   - Large blocks = Coarse branch alignment (Level 1)")
    print("   - Sub-blocks   = Finer branch alignment (Level 2+)")
    print("3. Off-diagonal 'hits' represent Topological Aliasing.")

    plt.show()


if __name__ == "__main__":
    visualize_cantor_topology()
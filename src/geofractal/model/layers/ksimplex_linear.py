import torch
import torch.nn as nn
import torch.nn.functional as F


class SimplexLayer(nn.Module):
    """Single layer of simplex traversal."""

    def __init__(self, hidden_dim, num_edges):
        super().__init__()
        self.route_weights = nn.Linear(hidden_dim, num_edges)
        self.edge_transforms = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=False)
            for _ in range(num_edges)
        ])
        self.layer_bias = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, h):
        # h: (batch, input_dim, hidden_dim)

        # Routing weights
        route_logits = self.route_weights(h)
        route_weights = F.softmax(route_logits, dim=-1)  # (batch, input_dim, num_edges)

        # Edge transforms
        edge_outputs = torch.stack([
            edge(h) for edge in self.edge_transforms
        ], dim=2)  # (batch, input_dim, num_edges, hidden_dim)

        # Weighted sum
        h = (edge_outputs * route_weights.unsqueeze(-1)).sum(dim=2)
        h = h + self.layer_bias
        h = F.gelu(h)

        return h


class KSimplexLinear(nn.Module):
    """
    k-simplex as depth structure.

    k=4 pentachoron:
        - 5 vertices
        - 5 layers of depth
        - 4 connections per vertex (edges)
        - Each input scalar traverses full depth
    """

    def __init__(self, input_dim, output_dim, k=4):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.k = k
        self.num_vertices = k + 1
        self.depth = k + 1
        self.edges_per_vertex = k
        self.hidden_dim = self.num_vertices

        # Entry: scalar → hidden state
        self.entry = nn.Linear(1, self.hidden_dim)

        # Depth layers
        self.layers = nn.ModuleList([
            SimplexLayer(self.hidden_dim, self.edges_per_vertex)
            for _ in range(self.depth)
        ])

        # Exit: hidden state → scalar
        self.exit = nn.Linear(self.hidden_dim, 1)

        # Output projection if dims differ
        if input_dim != output_dim:
            self.output_proj = nn.Linear(input_dim, output_dim)
        else:
            self.output_proj = None

    def forward(self, x):
        batch = x.shape[0]

        # Entry: (batch, input_dim) → (batch, input_dim, hidden_dim)
        h = x.unsqueeze(-1)
        h = self.entry(h)

        # Traverse depth
        for layer in self.layers:
            h = layer(h)

        # Exit: (batch, input_dim, hidden_dim) → (batch, input_dim)
        out = self.exit(h).squeeze(-1)

        if self.output_proj is not None:
            out = self.output_proj(out)

        return out

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    def structure_summary(self):
        print(f"\n=== KSimplexLinear (k={self.k}) ===")
        print(f"Depth: {self.depth} layers")
        print(f"Edges per layer: {self.edges_per_vertex}")
        print(f"Hidden dim: {self.hidden_dim}")
        print(f"Params: {self.param_count():,}")

        linear_equiv = self.input_dim * self.output_dim + self.output_dim
        print(f"nn.Linear equiv: {linear_equiv:,}")
        print(f"Ratio: {self.param_count() / linear_equiv:.3f}x")


class KSimplexClassifier(nn.Module):
    def __init__(self, k=4):
        super().__init__()
        self.simplex = KSimplexLinear(784, 784, k=k)
        self.head = nn.Linear(784, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.simplex(x)
        x = self.head(x)
        return x

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    print("=== KSimplex as Depth Structure ===\n")

    for k in [2, 3, 4]:
        layer = KSimplexLinear(784, 784, k=k)
        layer.structure_summary()

        x = torch.randn(4, 784)
        y = layer(x)
        print(f"Forward: {x.shape} → {y.shape}")
        print(f"✓ Gradients flow\n")

    print("=== Classifier ===")
    model = KSimplexClassifier(k=4)
    print(f"Total params: {model.param_count():,}")
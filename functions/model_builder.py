import torch
from torch import nn


# 1. Create a class caledd PatchEmbedding
class PatchEmbedding(nn.Module):
    """Turns a 2D input image into a 1D sequence learnable embedding vector.

    Args:
      in_channels(int): Number of color channels for the input images. Defaults set to 3
      patch_size(int): Size of patches to convert input image into. Defaults set to 16
      embedding_dim(int): Size of embedding to turn image into. Defaults to 768
    """

    # 2. Initialize the layer with appropriate hyperparameters
    def __init__(
        self, in_channels: int = 3, patch_size: int = 16, embedding_dim: int = 768
    ):
        super().__init__()

        self.patch_size = patch_size

        # 3. Create a layer to turn an image into embedded patches
        self.patcher = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )

        # 4. Create a layer to flatten feature map outputs of Conv2d
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)

    # 5. Define a forward method to define the forward computation steps
    def forward(self, x):
        # Create assertion to check that inputs are the correct shape
        image_resolution = x.shape[-1]
        assert (
            image_resolution % self.patch_size == 0
        ), f"Input image size must be divisible by patch_size, image shape: {image_resolution}, patch_size = {self.patch_size}"

        # Perform the forward pass and permute into the correct shape
        return self.flatten(self.patcher(x)).permute(0, 2, 1)


class MultiHeadSelfAttentionBlock(nn.Module):
    """Creates a multi-head self-attention block ("MSA block" for short)."""

    def __init__(
        self,
        num_heads: int = 12,  # Heads from Table 1 for ViT-Base
        embedding_dim: int = 768,  # Hidden size D (embedding dimension) from Table 1 for ViT-Base
        attn_dropout: int = 0,
    ):
        super().__init__()
        # Create the norm layer (LN)
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        # Create multihead attention (MSA) layer
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )  # is the batch first? (batch, seq, feature) -> (batch, number_of_patches, embedding_dimension)

    def forward(self, x):
        norm_x = self.layer_norm(x)
        attention_output, _ = self.multihead_attention(
            query=norm_x, key=norm_x, value=norm_x, need_weights=False
        )
        return attention_output + x


# Create a class that inherits nn.Module
class MLPBlock(nn.Module):
    """Creates a layer normalized multilayer perceptron block (MLP Block) for short"""

    # Initialize the class with hyperparameters from Table 1 and Table 3
    def __init__(
        self, embedding_dim: int = 768, mlp_size: int = 3072, dropout: int = 0.1
    ):
        super().__init__()

        # Create the norm layer (LN)
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        # Create the MLP
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=mlp_size),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_size, out_features=embedding_dim),
            nn.Dropout(p=dropout),
        )

    # Create the forward method to pass data through layers
    def forward(self, x):
        return self.mlp(self.layer_norm(x)) + x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 768,  # Hidden size D from Table 1, 768 for ViT-Base
        num_heads: int = 12,  # from table 1
        mlp_size: int = 3072,  # from table 1
        mlp_dropout: int = 0.1,  # from table 3
        attn_dropout: int = 0,
    ):
        super().__init__()

        # Create MSA block (Equation 2)
        self.msa_block = MultiHeadSelfAttentionBlock(
            num_heads=num_heads, embedding_dim=embedding_dim, attn_dropout=attn_dropout
        )

        # Create MLP block (Equation 3)
        self.mlp_block = MLPBlock(
            embedding_dim=embedding_dim, mlp_size=mlp_size, dropout=mlp_dropout
        )

    def forward(self, x):
        return self.mlp_block(self.msa_block(x))


# Create a ViT class
class ViT(nn.Module):
    def __init__(
        self,
        img_size: int = 224,  # Table 3 from the ViT paper
        in_channels: int = 3,
        patch_size: int = 16,
        num_transformer_layer: int = 12,  # Table 1 "Layers" for ViT-Base
        embedding_dim: int = 768,  # Hidden size D from Table 1 for ViT-Base
        mlp_size: int = 3072,  # Table 1
        num_heads: int = 12,  # Table 1
        attn_dropout: int = 0,
        mlp_dropout: int = 0.1,
        embedding_dropout: int = 0.1,  # Dropout for patch and position embeddings
        num_classes: int = 1000,
    ):  # Number of classes in the classification problem
        super().__init__()

        # Make an assertion for image size (must be compatible with patch size)
        assert (
            img_size % patch_size == 0
        ), f"Image size must be divisible by patch size, image: {img_size}, patch_size: {patch_size}"

        # Calculate the number of patches
        self.number_of_patches = int(
            (img_size * img_size) / patch_size**2
        )  # Formula from the ViT paper

        # Create learnable class token embedding (needs to go at front of sequence of patch embeddings)
        self.class_embedding = nn.Parameter(
            data=torch.randn(1, 1, embedding_dim), requires_grad=True
        )

        # Create learnable position embedding
        self.position_embedding = nn.Parameter(
            data=torch.randn(1, self.number_of_patches + 1, embedding_dim),
            requires_grad=True,
        )

        # Create embedding dropout value
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        # Create patch embedding layer
        self.patch_embedding_layer = PatchEmbedding(
            in_channels=in_channels, patch_size=patch_size, embedding_dim=embedding_dim
        )

        # Create the transformer encoder block
        self.transformer_encoder = nn.Sequential(
            *[
                TransformerEncoder(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_size=mlp_size,
                    mlp_dropout=mlp_dropout,
                )
                for _ in range(num_transformer_layer)
            ]
        )

        # Create classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim, out_features=num_classes),
        )

    def forward(self, x):
        # Get the batch_size
        batch_size = x.shape[0]

        # Create class token embedding and expand it to match the batch size (Equation 1)
        class_token = self.class_embedding.expand(
            batch_size, -1, -1
        )  # "-1" means to infer the dimension

        # Create the patch embedding (Equation 1)
        x = self.patch_embedding_layer(x)

        # Concat class token embedding and patch embedding (Equation 1)
        x = torch.cat(
            (class_token, x), dim=1
        )  # (batch_size, number_of_patches, embedding_dim)

        # Add position embedding to class token and patch_embedding
        x = self.position_embedding + x

        # Apply dropout
        x = self.embedding_dropout(x)

        # Pass the position and patch embedding to the transformer encoder (Equation 2 & 3)
        x = self.transformer_encoder(x)

        # Put 0th index logit through the classifier (Equation 4)
        x = self.classifier(x[:, 0])

        return x

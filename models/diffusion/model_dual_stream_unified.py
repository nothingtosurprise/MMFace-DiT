# model_dual_stream_unified.py
"""
Unified Dual-Stream Diffusion Transformer (DiT) with separate image and text streams.
The two streams process information in parallel and interact through shared attention.
This unified version incorporates a Modality Embedder to handle different types of
conditioning latents (e.g., masks or sketches) dynamically during the forward pass.
"""
import math
import torch
import torch.nn as nn
from einops import rearrange
import torch.utils.checkpoint
from utils.rope_embedding import EmbedND
from utils.dual_stream_processor import DualStreamAttnProcessor

# ------------------------------ #
#     Helper Functions           #
# ------------------------------ #
def modulate(x, shift, scale):
    """Apply spatial modulation: x * (1 + scale) + shift"""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    Create 2D sin/cos positional embeddings.
    Args:
        embed_dim (int): Embedding dimension.
        grid_size (int): The grid height and width.
        cls_token (bool): Whether to include a cls token.
        extra_tokens (int): Number of extra tokens (e.g., cls).
    Returns:
        torch.Tensor: Positional embeddings. Shape: [grid_size*grid_size (+ extra_tokens), embed_dim]
    """
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_w, grid_h, indexing='ij')  # here w goes first
    grid = torch.stack(grid, dim=0)  # [2, grid_size, grid_size]
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = torch.cat([torch.zeros([1, embed_dim]), pos_embed], dim=0)
    if extra_tokens > 0:
        # Add extra embeddings (e.g., for timestep/class)
        pos_embed = torch.cat([torch.zeros([extra_tokens, embed_dim]), pos_embed], dim=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """Generate 2D sin/cos embeddings from a grid."""
    assert embed_dim % 2 == 0
    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    emb = torch.cat([emb_h, emb_w], dim=1) # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)
    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum('m,d->md', pos, omega)  # (M, D/2), outer product
    emb_sin = torch.sin(out) # (M, D/2)
    emb_cos = torch.cos(out) # (M, D/2)
    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb

# ------------------------------ #
#     Core Modules               #
# ------------------------------ #
class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding """
    def __init__(self, img_size, patch_size, in_chans, embed_dim, bias=True):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})."
        x = self.proj(x) # [B, embed_dim, grid_size, grid_size]
        # Flatten grid spatial dimensions and move channels to second dim
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        return x

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq.to(self.mlp[0].weight.dtype))
        return t_emb

class CaptionEmbedder(nn.Module):
    """
    Embeds text caption/pooled embeddings into vector representations.
    Handles both pooled (for adaLN-Zero) and sequence (for cross-attn) embeddings.
    """
    def __init__(self, in_channels, hidden_size, uncond_prob=0.1):
        super().__init__()
        self.proj = nn.Linear(in_channels, hidden_size)
        self.null_feature_pooled = nn.Parameter(torch.zeros(in_channels))
        self.uncond_prob = uncond_prob

    def token_drop(self, caption, force_drop_ids=None):
        """
        Drops caption to simulate condition dropout for classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(caption.shape[0], device=caption.device) < self.uncond_prob
        else:
            drop_ids = force_drop_ids == 1
        caption = torch.where(drop_ids[:, None], self.null_feature_pooled, caption)
        return caption

    def forward(self, caption, train, force_drop_ids=None):
        use_dropout = self.uncond_prob > 0 and train
        if (force_drop_ids is None) and use_dropout:
            caption = self.token_drop(caption, None)
        elif force_drop_ids is not None:
            caption = self.token_drop(caption, force_drop_ids)
        embeddings = self.proj(caption)
        return embeddings

class DualStreamAttention(nn.Module):
    """A replacement for nn.MultiheadAttention that uses our custom processor."""
    def __init__(self, hidden_size, num_heads, single=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.heads = num_heads
        self.single = single
        # Linear layers for queries, keys, and values
        self.to_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_out = nn.Linear(hidden_size, hidden_size, bias=False)
        if not single:
            # Additional layers for text stream
            self.to_q_t = nn.Linear(hidden_size, hidden_size, bias=False)
            self.to_k_t = nn.Linear(hidden_size, hidden_size, bias=False)
            self.to_v_t = nn.Linear(hidden_size, hidden_size, bias=False)
            self.to_out_t = nn.Linear(hidden_size, hidden_size, bias=False)
        # Set the processor
        self.processor = None

    def set_processor(self, processor: "DualStreamAttnProcessor"):
        self.processor = processor

    def forward(self, image_tokens, text_tokens=None, image_tokens_masks=None, rope=None):
        if self.processor is None:
            raise ValueError("Attention processor not set")
        return self.processor(self, image_tokens, image_tokens_masks, text_tokens, rope)

class DualStreamDiTBlock(nn.Module):
    """
    A DiT block with dual streams (image and text) that can attend to each other.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attention_type='full', sparse_attn_config=None, **block_kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        # LayerNorm for image and text streams
        self.norm_i = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_t = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # Shared attention mechanism
        self.attention_type = attention_type
        if self.attention_type == 'sparse':
            raise NotImplementedError("Sparse attention not implemented")
        else: # 'full' attention with RoPE
            self.attn = DualStreamAttention(hidden_size, num_heads)
        # MLP for image and text streams
        self.norm_mlp_i = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_mlp_t = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp_i = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            approx_gelu(),
            nn.Linear(mlp_hidden_dim, hidden_size),
        )
        self.mlp_t = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            approx_gelu(),
            nn.Linear(mlp_hidden_dim, hidden_size),
        )
        # adaLN modulation that outputs 12 * hidden_size parameters for both streams
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 12 * hidden_size, bias=True)
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, image_tokens, text_tokens, c_global=None, rope=None):
        # Extract modulation parameters for both streams
        shift_msa_i, scale_msa_i, gate_msa_i, shift_mlp_i, scale_mlp_i, gate_mlp_i, \
        shift_msa_t, scale_msa_t, gate_msa_t, shift_mlp_t, scale_mlp_t, gate_mlp_t = \
            self.adaLN_modulation(c_global).chunk(12, dim=1)
        # --- Attention Step: Normalize -> Modulate -> Operate ---
        # 1. Normalize image and text tokens independently
        norm_image_tokens = self.norm_i(image_tokens)
        norm_text_tokens = self.norm_t(text_tokens)
        # 2. Modulate the normalized tokens
        modulated_image_tokens = modulate(norm_image_tokens, shift_msa_i, scale_msa_i)
        modulated_text_tokens = modulate(norm_text_tokens, shift_msa_t, scale_msa_t)
        # 3. Apply shared attention mechanism with RoPE
        if self.attention_type == 'sparse':
            raise NotImplementedError("Sparse attention not implemented for RoPE")
        else: # full attention with RoPE
            image_attn_out, text_attn_out = self.attn(
                modulated_image_tokens, 
                modulated_text_tokens, 
                rope=rope
            )
        # 4. Apply gating and residual connection
        image_tokens = image_tokens + gate_msa_i.unsqueeze(1) * image_attn_out
        text_tokens = text_tokens + gate_msa_t.unsqueeze(1) * text_attn_out
        # --- MLP Step: Normalize -> Modulate -> Operate ---
        # 1. Normalize the outputs from attention
        norm_mlp_image_tokens = self.norm_mlp_i(image_tokens)
        norm_mlp_text_tokens = self.norm_mlp_t(text_tokens)
        # 2. Modulate the normalized tokens for MLP
        modulated_mlp_image_tokens = modulate(norm_mlp_image_tokens, shift_mlp_i, scale_mlp_i)
        modulated_mlp_text_tokens = modulate(norm_mlp_text_tokens, shift_mlp_t, scale_mlp_t)
        # 3. Apply MLPs independently
        image_mlp_out = self.mlp_i(modulated_mlp_image_tokens)
        text_mlp_out = self.mlp_t(modulated_mlp_text_tokens)
        # 4. Apply gating and residual connection
        image_tokens = image_tokens + gate_mlp_i.unsqueeze(1) * image_mlp_out
        text_tokens = text_tokens + gate_mlp_t.unsqueeze(1) * text_mlp_out
        return image_tokens, text_tokens

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        # Initialize the last linear layer to zero
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        # x: [B, N, D]
        # c: [B, D]
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

# ------------------------------ #
#     Main DiT Model             #
# ------------------------------ #
class UnifiedDualStreamDiT(nn.Module):
    """
    Unified Dual-Stream Diffusion Transformer with separate image and text processing streams.
    Incorporates a Modality Embedder to dynamically handle different conditioning types
    (e.g., mask or sketch) specified by the 'modality' input during the forward pass.
    """
    def __init__(self, 
                 input_size=32, # Latent resolution (e.g., 256/8 = 32)
                 patch_size=2,
                 in_channels=8, # 4 (image latents) + 4 (conditioning latents: mask OR sketch)
                 hidden_size=1152,
                 depth=28,
                 num_heads=16,
                 mlp_ratio=4.0,
                 text_embed_dim=1024, # From CLIP Text Encoder
                 learn_sigma=False,
                 caption_dropout_prob=0.1, # Probability of dropping text condition during training
                 # --- Attention configuration ---
                 attention_type='full',
                 sparse_attn_config=None
                 ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels # Should be 8 for concat method (4 img + 4 conditioning)
        self.out_channels = (in_channels // 2) * 2 if learn_sigma else (in_channels // 2)  # 8 or 4
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.input_size = input_size
        self.use_cross_attn = False  # Not used in dual-stream architecture
        self.caption_dropout_prob = caption_dropout_prob
        # Gradient checkpointing support
        self.gradient_checkpointing = False

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        # Caption embedder for pooled text embeddings (used in adaLN-Zero)
        self.c_embedder = CaptionEmbedder(text_embed_dim, hidden_size, uncond_prob=caption_dropout_prob)
        # CLIP sequence embedding projection
        # Project CLIP embeddings to the same dimension as the model's hidden size
        self.clip_seq_proj = nn.Linear(text_embed_dim, hidden_size)
        
        # --- Modality Embedder ---
        # Embedding layer for modality type (0: mask, 1: sketch)
        self.modality_embedder = nn.Embedding(2, hidden_size)

        # RoPE positional embedding
        head_dim = hidden_size // num_heads
        self.pe_embedder = EmbedND(theta=10000, axes_dim=[head_dim // 2, head_dim // 2])  # For 2D grid positions

        # Dual-stream blocks
        self.blocks = nn.ModuleList([
            DualStreamDiTBlock(
                hidden_size, num_heads, mlp_ratio=mlp_ratio,
                attention_type=attention_type, sparse_attn_config=sparse_attn_config
            ) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)
        # Initialize timestep embedding MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        # Initialize caption embedding MLP
        nn.init.normal_(self.c_embedder.proj.weight, std=0.02)
        nn.init.constant_(self.c_embedder.proj.bias, 0)
        # Initialize CLIP sequence projection
        nn.init.normal_(self.clip_seq_proj.weight, std=0.02)
        nn.init.constant_(self.clip_seq_proj.bias, 0)
        
        # --- Initialize Modality Embedder ---
        # Standard normal initialization for the embedding weights
        nn.init.normal_(self.modality_embedder.weight, std=0.02)

        # Initialize attention processors
        for block in self.blocks:
            if hasattr(block.attn, 'set_processor'):
                block.attn.set_processor(DualStreamAttnProcessor())
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def gradient_checkpointing_enable(self):
        """
        Enable gradient checkpointing for the model.
        """
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        """
        Disable gradient checkpointing for the model.
        """
        self.gradient_checkpointing = False

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, C, H, W)
        """
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, text_pooled_embeddings, modality, clip_seq_embeddings=None):
        """
        Forward pass for Unified Dual-Stream DiT with dynamic modality conditioning.

        Args:
            x (torch.Tensor): Concatenated noisy image latents and conditioning latents.
                              Shape: [B, 8, Hl, Wl] (e.g., B, 8, 32, 32)
                              The conditioning latents are either mask or sketch latents.
            t (torch.Tensor): Timestep tensor. Shape: [B,]
            text_pooled_embeddings (torch.Tensor): Pooled text embeddings. Shape: [B, text_embed_dim]
            modality (torch.Tensor): Modality identifier tensor. Shape: [B,]
                                     Values: 0 for mask, 1 for sketch.
            clip_seq_embeddings (torch.Tensor, optional): CLIP sequence text embeddings. Shape: [B, seq_len, text_embed_dim]
            
        Returns:
            torch.Tensor: Predicted noise/target. Shape: [B, out_channels, Hl, Wl]
        """
        H = W = self.input_size
        batch_size = x.shape[0]
        
        # x is the concatenated input [image_latents, conditioning_latents (mask OR sketch)]
        image_tokens = self.x_embedder(x)  # Patchify: [B, N, D]
        t_emb = self.t_embedder(t)         # Time embedding: [B, D]
        
        # --- Modality Embedding Lookup ---
        # Look up the embedding vector for each sample's modality
        modality_emb = self.modality_embedder(modality) # [B, D]

        # Process CLIP text embeddings for the text stream
        if clip_seq_embeddings is not None:
            # Project CLIP sequence embeddings to the model's hidden size
            text_tokens = self.clip_seq_proj(clip_seq_embeddings)
        else:
            # If no CLIP embeddings provided, create dummy text tokens
            text_tokens = torch.zeros(image_tokens.shape[0], 1, image_tokens.shape[2], device=image_tokens.device)

        # --- Global conditioning with Modality Embedder ---
        # Global conditioning for adaLN-Zero (time + CLIP pooled text + modality)
        c_global = self.c_embedder(text_pooled_embeddings, train=self.training) + t_emb + modality_emb # [B, D]

        # Generate position IDs for RoPE
        # Calculate actual sequence lengths
        image_seq_len = image_tokens.shape[1]
        text_seq_len = text_tokens.shape[1]
        total_seq_len = image_seq_len + text_seq_len

        # For image tokens, we need to generate 2D position IDs
        grid_size = int(image_seq_len ** 0.5)  # Assuming square grid
        if grid_size * grid_size == image_seq_len:  # Perfect square
            pos_ids_img = torch.stack(torch.meshgrid(torch.arange(grid_size), torch.arange(grid_size), indexing='ij'), dim=-1)
            pos_ids_img = pos_ids_img.reshape(-1, 2).expand(batch_size, -1, -1).to(x.device)
        else:
            # Fallback for non-square grids
            pos_ids_img = torch.zeros((batch_size, image_seq_len, 2), device=x.device, dtype=torch.long)

        # For text tokens, we use 1D position IDs
        pos_ids_text = torch.zeros((batch_size, text_seq_len, 2), device=x.device, dtype=torch.long)

        # Combine image and text position IDs
        pos_ids = torch.cat([pos_ids_img, pos_ids_text], dim=1)
        # Generate RoPE embeddings
        rope = self.pe_embedder(pos_ids)

        # Apply transformer blocks
        for block in self.blocks:
            if self.training and self.gradient_checkpointing:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                image_tokens, text_tokens = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    image_tokens, text_tokens, c_global, rope,
                    use_reentrant=False
                )
            else:
                image_tokens, text_tokens = block(image_tokens, text_tokens, c_global=c_global, rope=rope) # [B, N, D]

        # Final layer - only use image tokens for output
        image_tokens = self.final_layer(image_tokens, c_global)                # [B, N, p*p*out_channels]
        image_tokens = self.unpatchify(image_tokens)                           # [B, out_channels, Hl, Wl]
        return image_tokens

    def forward_with_cfg(self, x, t, text_pooled_embeddings, modality, clip_seq_embeddings=None, cfg_scale=1.0):
        """
        Forward pass with classifier-free guidance.
        Assumes text embeddings and modality have been duplicated for null condition.
        """
        # Split inputs for conditional and unconditional passes
        half = x.shape[0] // 2
        x_cond, x_uncond = x[:half], x[half:]
        t_cond, t_uncond = t[:half], t[half:]
        pooled_cond, pooled_uncond = text_pooled_embeddings[:half], text_pooled_embeddings[half:]
        modality_cond, modality_uncond = modality[:half], modality[half:]
        clip_seq_cond, clip_seq_uncond = (clip_seq_embeddings[:half], clip_seq_embeddings[half:]) if clip_seq_embeddings is not None else (None, None)
        
        # Run conditional and unconditional passes
        out_cond = self.forward(x_cond, t_cond, pooled_cond, modality_cond, clip_seq_cond)
        out_uncond = self.forward(x_uncond, t_uncond, pooled_uncond, modality_uncond, clip_seq_uncond)
        # Apply CFG
        return out_uncond + cfg_scale * (out_cond - out_uncond)

# ------------------------------ #
#     Test Script                #
# ------------------------------ #
if __name__ == "__main__":
    # --- Test Script for Unified Dual-Stream DiT Model ---
    print("--- Testing Unified Dual-Stream DiT Model ---")
    # Define parameters
    batch_size = 2
    latent_height = 32 # Latent height/width (e.g., 256px image / 8)
    latent_width = 32
    latent_channels = 8 # 4 (image) + 4 (conditioning: mask OR sketch)
    text_embed_dim = 1024 # CLIP text embedding dimension
    hidden_size = 1152 # DiT-XL
    depth = 28
    num_heads = 16
    clip_seq_len = 77 # CLIP token sequence length (example)

    # Create dummy inputs
    noisy_latents = torch.randn(batch_size, latent_channels, latent_height, latent_width) # Concatenated input
    timesteps = torch.randint(0, 1000, (batch_size,))
    text_pooled_embs = torch.randn(batch_size, text_embed_dim) # Pooled embeddings
    clip_seq_embs = torch.randn(batch_size, clip_seq_len, text_embed_dim) # CLIP sequence embeddings
    # --- Create dummy modality tensor ---
    # 0 for mask, 1 for sketch. Here we randomly choose for testing.
    modality_tensor = torch.randint(0, 2, (batch_size,)) # [B,]

    # Instantiate the Unified Dual-Stream DiT model
    unified_dit_model = UnifiedDualStreamDiT(
        input_size=latent_height, # Assumes H=W
        patch_size=2,
        in_channels=latent_channels, # 8 channels
        hidden_size=hidden_size,
        depth=depth,
        num_heads=num_heads,
        text_embed_dim=text_embed_dim,
        learn_sigma=False
    )
    print(f"Unified Dual-Stream DiT Model instantiated:")
    print(f" Input Latents (Concatenated) Shape: {noisy_latents.shape}") # [B, 8, Hl, Wl]
    print(f" Timesteps Shape: {timesteps.shape}")
    print(f" Text Pooled Emb Shape: {text_pooled_embs.shape}")
    print(f" CLIP Seq Emb Shape: {clip_seq_embs.shape}")
    print(f" Modality Tensor Shape: {modality_tensor.shape}") # [B,]
    print(f" Model Output Channels: {unified_dit_model.out_channels}")

    # Run forward pass
    with torch.no_grad():
        output = unified_dit_model(
            noisy_latents, 
            timesteps, 
            text_pooled_embs, 
            modality_tensor, # Pass the modality tensor
            clip_seq_embeddings=clip_seq_embs
        )
    print(f"Output Shape: {output.shape}")
    print("Unified Dual-Stream DiT forward pass successful!")

    # --- Test forward_with_cfg ---
    print("\n--- Testing forward_with_cfg ---")
    # Duplicate inputs for CFG
    noisy_latents_cfg = torch.cat([noisy_latents, noisy_latents], dim=0)
    timesteps_cfg = torch.cat([timesteps, timesteps], dim=0)
    text_pooled_embs_cfg = torch.cat([text_pooled_embs, text_pooled_embs], dim=0)
    clip_seq_embs_cfg = torch.cat([clip_seq_embs, clip_seq_embs], dim=0) if clip_seq_embs is not None else None
    modality_tensor_cfg = torch.cat([modality_tensor, modality_tensor], dim=0) # Duplicate modality

    with torch.no_grad():
        output_cfg = unified_dit_model.forward_with_cfg(
            noisy_latents_cfg,
            timesteps_cfg,
            text_pooled_embs_cfg,
            modality_tensor_cfg, # Pass the duplicated modality tensor
            clip_seq_embeddings=clip_seq_embs_cfg,
            cfg_scale=3.0
        )
    print(f"Output Shape (CFG): {output_cfg.shape}")
    print("Unified Dual-Stream DiT forward_with_cfg pass successful!")
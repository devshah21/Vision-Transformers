import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    # splitting images into patches and then embedding
    
    # img size refers to size of the image (square)
    # patch size refers to the size of the path (sqaure)
    # in_chans refers to the number of input channels
    # embed_dim refers to the embedding dimensions
    def __init__(self, img_size, patch_size, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size//patch_size) ** 2
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
    def forward(self, x):
        # x is a torch.Tensor with the Shape (n_samples, in_chans, img_size, img_size)
        # it returns a ch.tensor with the shape (n_samples, n_patches, embed_dim)
        x = self.proj(
            x
        )
        x = x.flatten(2)
        x = x.transpose(1,2)
        return x

import torch
import torch.nn as nn

class Attention(nn.Module):
    
    # dim refers tot he input and out dimension of per token features
    # n_heads is the num of attention heads
    # qvk_bias is a boolean and if it's true, then we include the qvk projections
    # atten_p is the dropout prob which is applied to the qvk tensors
    # proj_p is also a dropout prob, but it is applied to the output tensor
    
    def __init__(self, dim, n_heads=12, qvk_bias = True, atten_p=0, proj_p =0):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.qvk = nn.Linear(dim, dim*3, bias=qvk_bias) #linear mapping; takes in token embedding and generates qvk
        self.attention_drop = nn.Dropout(atten_p) 
        self.proj = nn.Linear(dim, dim) # linear mapping that takes concatenated heads and maps them into a new space
        self.proj_drop = nn.Dropout(proj_p)
    
    def forward(self, x):
    # Get the shape of the input tensor
        n_samples, n_tokens, dim = x.shape

        # Check if the dimension of the input tensor matches self.dim
        if dim != self.dim:
            raise ValueError("Input tensor dimension doesn't match self.dim")

        # Apply the qvk transformation to the input tensor
        qvk = self.qvk(x)

        # Reshape qvk to have 3 dimensions: (n_samples, n_tokens, 3, self.n_heads, self.head_dim)
        qvk = qvk.reshape(n_samples, n_tokens, 3, self.n_heads, self.head_dim)

        # Permute the dimensions of the qvk tensor
        qvk = qvk.permute(2, 0, 3, 1, 4)

        # Separate q, k, and v from the reshaped qvk tensor
        q, k, v = qvk[0], qvk[1], qvk[2]

        # Transpose the k tensor
        k_t = k.transpose(-2, -1)

        # Compute the dot product of q and transposed k, scaled by self.scale
        dotp = (q @ k_t) * self.scale

        # Apply softmax to the dot product to compute attention scores
        attn = dotp.softmax(dim=-1)

        # Apply dropout to the attention scores
        attn = self.attention_drop(attn)

        # Compute the weighted average of values (v) using the attention scores
        weightedavg = attn @ v

        # Transpose the dimensions of the weighted average tensor
        weightedavg = weightedavg.transpose(1, 2)

        # Flatten the weighted average tensor
        weightedavg = weightedavg.flatten(2)

        # Apply linear projection to the flattened tensor
        x = self.proj(weightedavg)

        # Apply dropout to the projected tensor
        x = self.proj_drop(x)

        # Return the final output tensor
        return x
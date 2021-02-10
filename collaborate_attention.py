import torch
import torch.nn as nn
import tensorly as tl
import copy
import timm
import einops
import tqdm
from pathlib import Path
import argparse
import models

tl.set_backend("pytorch")
from tensorly.decomposition import parafac


class CollaborativeAttention(nn.Module):
    def __init__(self, dim, all_key_dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.all_key_dim = all_key_dim or dim

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, self.all_key_dim, bias=False)
        self.k = nn.Linear(dim, self.all_key_dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=True)
        # self.qkv = nn.Linear(dim, 2 * self.all_key_dim + dim, bias=qkv_bias)
        # just a mixing matrix of dimension (num_heads, all_key_dim)
        self.mixing = nn.Linear(all_key_dim, num_heads, bias=False)
        self.content_bias = nn.Linear(dim, num_heads, bias=False)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        # qkv = self.qkv(x)
        # qk = qkv[..., :2 * self.all_key_dim].reshape(B, N, 2, self.all_key_dim).permute(2, 0, 1, 3)
        # q_shared, k = qk[0], qk[1]
        # v = qkv[..., 2 * self.all_key_dim:].reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q_shared = self.q(x)
        k = self.k(x)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # use mixing matrix to obtain query per head
        q = torch.einsum("bnd,hd->bhnd", q_shared, self.mixing.weight)
        # unsqueeze head dimension for the shared keys
        k = k.unsqueeze(1)

        content_bias = self.content_bias(x)
        broadcast_content_bias = einops.rearrange(content_bias, "b n h -> b h () n")

        attn = (q @ k.transpose(-2, -1) + broadcast_content_bias) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @classmethod
    def from_attention_layer(cls, original_layer, all_key_dim, tol=1e-6, reparametrize=True):
        # Retrive configuration of the original layer
        device = next(original_layer.parameters()).device
        dim = original_layer.proj.weight.shape[0]
        num_heads = original_layer.num_heads
        qkv_bias = original_layer.qkv.bias is not None

        # Create collaborative layer
        layer = CollaborativeAttention(dim, all_key_dim, num_heads=num_heads, qkv_bias=qkv_bias)
        layer = layer.to(device)

        if not reparametrize:
            return layer

        # Copy untouched layers
        layer.attn_drop = copy.deepcopy(original_layer.attn_drop)
        layer.proj = copy.deepcopy(original_layer.proj)
        layer.proj_drop = copy.deepcopy(original_layer.proj_drop)

        # Copy value weight and bias
        layer.v.weight.data.copy_(original_layer.qkv.weight[-dim:, :])
        layer.v.bias.data.copy_(original_layer.qkv.bias[-dim:])

        # Tensor decomposition to get shared projections and mixing
        WQ_per_head = original_layer.qkv.weight[:dim, :].view([num_heads, -1, dim])
        WK_per_head = original_layer.qkv.weight[dim : 2 * dim, :].view([num_heads, -1, dim])
        WQWKT_per_head = torch.einsum("hdq,hdk->qhk", WQ_per_head, WK_per_head)

        _, factors = parafac(WQWKT_per_head.detach(), all_key_dim, init="random", tol=tol)
        WQ_shared, mixing, WK_shared = factors
        layer.k.weight.data.copy_(WK_shared.transpose(0, 1))
        layer.q.weight.data.copy_(WQ_shared.transpose(0, 1))
        layer.mixing.weight.data.copy_(mixing)

        # Content bias reparametrization
        bq_per_head = original_layer.qkv.bias[:dim].reshape([num_heads, -1])
        content_bias = bq_per_head.unsqueeze(1) @ WK_per_head
        content_bias = content_bias.squeeze(1)
        layer.content_bias.weight.data.copy_(content_bias)

        return layer

class FlexibleKeyDimensionAttention(nn.Module):
    def __init__(self, dim, all_key_dim, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        print(f"Creates FlexibleKeyDimensionAttention with all_key_dim={all_key_dim}")
        self.dim = dim
        self.all_key_dim = all_key_dim or dim

        self.num_heads = num_heads
        head_dim = all_key_dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, self.all_key_dim, bias=qkv_bias)
        self.k = nn.Linear(dim, self.all_key_dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape

        q = self.q(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def swap(model, compressed_key_dim, reparametrize=True):
    num_layers = len(model.blocks)
    _range = tqdm.trange if reparametrize else range
    for i in _range(num_layers):
        model.blocks[i].attn = CollaborativeAttention.from_attention_layer(
            model.blocks[i].attn, compressed_key_dim, reparametrize=reparametrize
        )


def get_args_parser():
    parser = argparse.ArgumentParser("Reparametrization script for collaborative attention")
    # fmt: off
    parser.add_argument("model", type=str, help="Model to load")
    parser.add_argument("shared_key_query_dim", type=int, help="New shared dimension")
    parser.add_argument("--output_dir", default="./models", help="Directory where to save the reparametrized models")
    # fmt: on

    return parser


def main(args):
    print(f"Creating model: {args.model}")
    model = timm.create_model(
        args.model,
        pretrained=True,
        num_classes=1000,
        drop_rate=0.1,
        drop_path_rate=0.1,
        drop_block_rate=None,
    )
    model = model.to("cuda")

    print("Reparametrizing model with collaborative attention...")
    swap(model, args.shared_key_query_dim)

    print("Saving...")
    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / f"{args.model}_collab{args.shared_key_query_dim}.pth"
    torch.save({"model": model.state_dict()}, checkpoint_path)
    print(f"Reparametrized model saved in '{checkpoint_path}'.")


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

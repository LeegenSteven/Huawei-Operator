import torch

D_NOPE, D_ROPE, D_V = 512, 64, 512

def make_custom_inputs(ax, device):
    B, S1, S2, N1, K = ax['B'], ax['S1'], ax['S2'], ax['N1'], ax['K']
    dtype = torch.float16 if ax['dtype_code'] == 1 else torch.bfloat16
    g = torch.Generator(device='cpu')
    g.manual_seed(1234 + B + S1 + S2 + N1)
    query = torch.randn(B, S1, N1, D_NOPE, dtype=dtype, generator=g).to(device)
    key = torch.randn(B, S2, 1, D_NOPE, dtype=dtype, generator=g).to(device)
    value = torch.randn(B, S2, 1, D_V, dtype=dtype, generator=g).to(device)
    query_rope = torch.randn(B, S1, N1, D_ROPE, dtype=dtype, generator=g).to(device)
    key_rope = torch.randn(B, S2, 1, D_ROPE, dtype=dtype, generator=g).to(device)
    out = torch.full((B, S1, 1, K), -1, dtype=torch.int32)
    k_keep = min(K, S2)
    for b in range(B):
        for s in range(S1):
            perm = torch.randperm(S2, generator=g)[:k_keep]
            out[b, s, 0, :k_keep] = perm.to(torch.int32)
    sparse_indices = out.to(device)
    scale_value = 1.0 / (D_NOPE + D_ROPE) ** 0.5
    return {'query': query.contiguous(), 'key': key.contiguous(),
            'value': value.contiguous(), 'sparse_indices': sparse_indices.contiguous(),
            'scale_value': scale_value, 'query_rope': query_rope.contiguous(),
            'key_rope': key_rope.contiguous()}

def run(query, key, value, sparse_indices, scale_value, query_rope, key_rope):
    q = torch.cat([query.float(), query_rope.float()], dim=-1)
    k = torch.cat([key.float(), key_rope.float()], dim=-1)
    v = value.float()
    idx = sparse_indices.squeeze(2)
    B, S1, N1, Dqk = q.shape
    _, S2, _, Dv = v.shape
    valid = (idx >= 0) & (idx < S2)
    idx_safe = idx.clamp(0, S2 - 1).to(torch.int64)
    k_flat = k.squeeze(2)
    v_flat = v.squeeze(2)
    k_sel = torch.gather(k_flat.unsqueeze(1).expand(-1, S1, -1, -1), 2, idx_safe.unsqueeze(-1).expand(-1, -1, -1, Dqk))
    v_sel = torch.gather(v_flat.unsqueeze(1).expand(-1, S1, -1, -1), 2, idx_safe.unsqueeze(-1).expand(-1, -1, -1, Dv))
    scores = torch.einsum("bsnd,bskd->bsnk", q, k_sel) * scale_value
    scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    attn = torch.nan_to_num(attn, nan=0.0)
    out = torch.einsum("bsnk,bskd->bsnd", attn, v_sel)
    return out.to(dtype=query.dtype)

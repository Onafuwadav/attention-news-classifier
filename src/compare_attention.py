# Check my reduced MultiheadSelfAttention does match PyTorch MultiheadAttention
import torch
from attention import MultiheadSelfAttention_reduced

torch.manual_seed(0)

batch, seq_len, embedDim, noHeads = 4, 15, 32, 4
x = torch.randn(batch, seq_len, embedDim)

pad_mask = torch.zeros(batch, seq_len, dtype=torch.bool)
pad_mask[:2, -5:] = True

ours = MultiheadSelfAttention_reduced(embedDim, embedDim, noHeads)
ref = torch.nn.MultiheadAttention(embedDim, noHeads, batch_first=True)

with torch.no_grad():
    ref.in_proj_weight.copy_(torch.cat([ours.query.weight, ours.key.weight, ours.value.weight]))
    ref.in_proj_bias.copy_(torch.cat([ours.query.bias, ours.key.bias, ours.value.bias]))
    ref.out_proj.weight.copy_(torch.eye(embedDim))
    ref.out_proj.bias.zero_()

ours.eval()
ref.eval()

for name, mask in [("no mask", None), ("with padding mask", pad_mask)]:
    h_ours, att_ours = ours(x, key_padding_mask=mask)
    h_ref, att_ref = ref(x, x, x, key_padding_mask=mask, average_attn_weights=False)

    h_diff = (h_ours - h_ref).abs().max().item()
    att_diff = (att_ours - att_ref).abs().max().item()
    match = torch.allclose(h_ours, h_ref, atol=1e-5) and torch.allclose(att_ours, att_ref, atol=1e-5)

    print(f"{name}:")
    print(f"  output shape    ours {tuple(h_ours.shape)}  torch {tuple(h_ref.shape)}")
    print(f"  attention shape ours {tuple(att_ours.shape)}  torch {tuple(att_ref.shape)}")
    print(f"  max |diff| output {h_diff:.2e}, attention {att_diff:.2e}  ->  {'MATCH' if match else 'MISMATCH'}")

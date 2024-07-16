import torch
from model.llama.helpers import apply_rotary_emb_rectangular, apply_rotary_emb, \
                    precompute_freqs_cis, precompute_freqs_cis_rect
from model.llama.rewritten_models import ModelArgs

args = ModelArgs()

### APPLY RECTANGULAR ROTARY EMBEDDINGS VERIFICATION #####
token = torch.tensor([[128000, 5592, 663]])
_, length = token.shape

freqs_cis = precompute_freqs_cis(
    args.dim // args.n_heads,
    args.max_seq_len * 2,
    args.rope_theta)[:length]

freqs_cis_rect = precompute_freqs_cis_rect(
    args.dim // args.n_heads,
    args.max_seq_len * 2,
    args.rope_theta)[:length]

def rotary_embedding():
    xq = torch.randn([1, 1, 32, 128])
    xk = torch.randn([1, 1, 8, 128])

    xq_out, xk_out = apply_rotary_emb(xq, xk, freqs_cis)
    xq_out_rect, xk_out_rect = apply_rotary_emb_rectangular(xq, xk, freqs_cis_rect)

    print(f"xq_out vs. xq_out_rect:", torch.allclose(xq_out, xq_out_rect))
    print(f"xk_out vs. xk_out_rect:", torch.allclose(xk_out, xk_out_rect))

def generate_mask(length, position):
    mask = torch.full((length, length), float("-inf"))
    mask = torch.triu(mask, diagonal=1)
    mask = torch.hstack(
        [torch.zeros((length, position)), mask]
    ).type_as(token)
    return mask

# Original model
# import source_model

# print("SOURCE")
# x = torch.randn([1,length,4096])
# source = source_model.Attention(args).eval()
# mask = generate_mask(length, 0)
# source_out = source(x, 0, freqs_cis, None)

# import rewritten_models
# print("REWRITTEN")
# mask = torch.full([1,32,1,512], float("-inf"))
# mask[:,:,:,-length:] = 0
# cache_k = torch.zeros([1,512,8,128])
# cache_v = torch.zeros([1,512,8,128])
# test = rewritten_models.Attention(args).eval()
# test.wq = source.wq
# test.wk = source.wk
# test.wv = source.wv
# test.wo = source.wo
# test_out, cache_k, cache_v = test(x, mask, freqs_cis_rect, cache_k, cache_v)
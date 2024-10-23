from model_utils import precompute_freqs_cis_rect, precompute_freqs_cis_rect_exp
import random
import torch

count = 0
for _ in range(1000):
    random_numbers = [random.randint(1, 100) for _ in range(3)]
    if torch.allclose(precompute_freqs_cis_rect(*random_numbers), precompute_freqs_cis_rect_exp(*random_numbers)):
        count += 1

print(f"{count} / 1000")
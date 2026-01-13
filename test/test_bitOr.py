import torch
from lib.utils.prune import symmetricDifference, bitOr

mask1 = torch.tensor(
    [1, 1, 0, 0])
mask2 = torch.tensor(
    [1, 0, 1, 0])

res1 = symmetricDifference(mask1, mask2)
res2 = bitOr(mask1, mask2).to(torch.float32)

print(res1)
print(res2)
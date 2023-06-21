import torch

def map_normalization(map, bs):
    for i in range(bs):
        mk = map[i, 0, :, :]
        mkmin = torch.min(mk[:, :])
        mkmax = torch.max(mk[:, :])
        mk = ((mk - mkmin) / (mkmax - mkmin))  #  Normalize f to the range [0,1]
        map[i, 0, :, :] = mk
    return map


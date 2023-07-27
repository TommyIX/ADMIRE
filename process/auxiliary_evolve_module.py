import torch
from models.PolyProcess import draw_poly

def auxevolvehandler(mapE, now_snake, image_size):
    snakepoly = draw_poly(now_snake.detach().cpu().numpy(), 1, [image_size, image_size],2)
    # 二值化mapE
    mapEthres = torch.where(mapE < torch.tensor(6.0), torch.ones_like(mapE), torch.zeros_like(mapE))

    coincide_rate = torch.sum(mapEthres.cpu() * snakepoly) / torch.sum(snakepoly)
    return coincide_rate.item()

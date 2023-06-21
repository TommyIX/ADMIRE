import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

def plot_vector_field(ax, vx, vy, map=None, color='orange', scalenum=25):
    # scalenum是调整箭头大小用的参数，一般来说这个值越大，箭头就会越短
    H, W = vx.shape
    Y, X = np.meshgrid(range(0, H, 5), range(0, W, 5))
    scale = np.sqrt(np.max(vx ** 2 + vy ** 2)) * scalenum
    ax.imshow(map, cmap='gray')
    ax.axis("off")
    im = ax.quiver(X, Y, vx[Y, X], -vy[Y, X], scale=scale, color=color, headwidth=5)
    return im

def plot_result(path, iou, n, i1, mode, snake, snake_hist, GT, mapE, mapA, mapB, image, plot_force, gx=None, gy=None, Fu=None, Fv=None, compressed_hist = False):
    if plot_force:
        fig0, (ax) = plt.subplots(ncols=6, nrows=1, figsize=(16, 6))
    else:
        fig0, (ax) = plt.subplots(ncols=4, nrows=1, figsize=(12, 6))
    im = ax[0].imshow(image[0,:,:],cmap='gray')
    if not compressed_hist:
        for i in range(0, len(snake_hist), 10):
            ax[0].plot(snake_hist[i][1, :], snake_hist[i][0, :], '-.', color=[i / len(snake_hist), i / len(snake_hist), 1 - i / len(snake_hist)], lw=1)
    else:
        for i in range(0, len(snake_hist)):
            ax[0].plot(snake_hist[i][1, :], snake_hist[i][0, :], '-.', color=[i / len(snake_hist), i / len(snake_hist), 1 - i / len(snake_hist)], lw=1)
    if not GT is None:
        ax[0].plot(GT[:, 1], GT[:, 0], '-', color=[0.2, 1, 0.2], lw=1)
    ax[0].plot(snake[:, 1], snake[:, 0], '--', lw=1, color=[1, 0, 0])
    ax[0].axis('off')
    ax[0].set_title(r'image', y=-0.3, fontsize=6)
    plt.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04).remove()

    im0 = ax[1].imshow(mapE)
    plt.colorbar(im0, ax=ax[1], fraction=0.046, pad=0.04)
    ax[1].axis('off')
    ax[1].set_title(r'mapE', y=-0.3, fontsize=6)

    im2 = ax[2].imshow(mapB)
    plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)
    ax[2].axis('off')
    ax[2].set_title(r'mapB', y=-0.3, fontsize=6)

    im3 = ax[3].imshow(mapA)
    plt.colorbar(im3, ax=ax[3], fraction=0.046, pad=0.04)
    ax[3].axis('off')
    ax[3].set_title(r'mapA', y=-0.3, fontsize=6)

    if plot_force:
        im_mixforce = plot_vector_field(ax[4],-(gx+Fu),-(gy+Fv), map=image[0,:,:], scalenum=10)
        ax[4].set_title('Force: CAT+grad', fontsize=6, y=-0.3)
        im_cat = plot_vector_field(ax[5], -gx, -gy, map=image[0,:,:], scalenum=25)
        ax[5].set_title('pure CAT', fontsize=6, y=-0.3)

    fig0.suptitle('mode:' + mode + ',epoch:' + str(n) + ',figure:' + str(i1) + ',iou = %.2f' % iou, fontsize=20)

    plt.savefig(path + 'epoch-' + str(n) + '-' + mode + '-num-' + str(i1) + '.jpg', dpi=200)
    plt.close()

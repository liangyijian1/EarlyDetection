import os

import numpy as np
from visualizer import get_local
get_local.activate()
import cv2
import torch
import torchvision
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.net.GlobalLocal.GlobalLocalTransformer import GlobalLocalTransformer


def makeInput():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
    imgName = 'early-431.jpg'
    patch_size = 40
    xinput = []
    img = cv2.imread('../source/testdir/test/early/{}'.format(imgName), 0)
    glo_tensor = torch.unsqueeze(transform(img), 0).to('cuda:0').requires_grad_(True)
    patches_list = os.listdir('../source/testdir/testPatches{}/early/{}'.format(patch_size, imgName[:-4]))
    loc_tensor = torch.zeros(size=(1, len(patches_list), patch_size, patch_size))
    for idx, item in enumerate(patches_list):
        patch = cv2.imread('../source/testdir/testPatches{}/early/{}/{}'.format(patch_size, imgName[:-4], item), 0)
        patch_tensor = transform(patch)
        loc_tensor[:, idx, :, :] = patch_tensor
    xinput.append(glo_tensor)
    xinput.append(torch.unsqueeze(loc_tensor[:, 0, :, :], 0).to('cuda:0').requires_grad_(True))
    return xinput


def hook_fn(grad):
    print(grad)


def main():
    import matplotlib.pyplot as plt
    net = GlobalLocalTransformer(inplace=1, nblock=1, backbone='octnet').to('cuda:0')
    net.load_state_dict(torch.load('../output/model/GlobalLocalTransformer/octnet/nblock1/patches40/net_80.pth'))
    net.eval()
    # target_layers = [net.fftlist[0].conv2]
    # cam = GradCAM(model=net, target_layers=target_layers, use_cuda=False)
    # # targets = [ClassifierOutputTarget(1)]
    # input_tensor = makeInput()
    #
    # grayscale_cam = cam(input_tensor=input_tensor)
    # grayscale_cam = grayscale_cam[0, :]
    # # visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255., grayscale_cam, use_rgb=True)
    # plt.imshow(grayscale_cam)
    # plt.show()
    xinput = makeInput()
    img = xinput[0]
    img = img[0, 0, :, :].detach().cpu().numpy()
    out = net(xinput)
    cache = get_local.cache
    heat = list(cache.values())[0][0]
    plt.subplot(1, 3, 1)
    plt.imshow(heat[0, 5, :, :])

    heat_map = cv2.resize(heat[0, 5, :, :], (256, 256))
    plt.subplot(1, 3, 2)
    plt.imshow(heat_map)

    heat_map = np.uint8(255 * heat_map)
    heat_map = cv2.applyColorMap(heat_map, cv2.COLORMAP_JET)
    plt.subplot(1, 3, 3)
    plt.imshow(img)

    plt.show()

if __name__ == '__main__':
    main()

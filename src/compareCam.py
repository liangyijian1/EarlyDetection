import os

from torch import nn
from visualizer import get_local
get_local.activate()
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision
from src.net.GlobalLocal.GlobalLocalTransformer import GlobalLocalTransformer


# 定义获取梯度的函数
def backward_hook(module, grad_in, grad_out):
    grad_block.append(grad_out[0].detach())


# 定义获取特征图的函数
def farward_hook(module, input, output):
    fmap_block.append(output)


# 计算grad-cam并可视化
def cam_show_img(img, feature_map, grads, out_dir):
    H, W = img.shape
    # cam = np.zeros(feature_map.shape[1:], dtype=np.float32)  # 4
    cam = np.zeros(shape=(feature_map.shape[1], feature_map.shape[2]), dtype=np.float32)  # 4
    grads = grads.reshape([grads.shape[0], -1])  # 5
    weights = np.mean(grads, axis=1)  # 6
    for i, w in enumerate(weights):
        cam += w * feature_map[i, :, :]  # 7
        # cam[i] = w * feature_map[i]  # 7

    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    cam = cv2.resize(cam, (W, H))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    cam_img = 0.2 * heatmap + cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap)
    plt.subplot(1, 3, 3)
    plt.imshow(cam_img)
    # os.makedirs('./tmp', exist_ok=True)
    plt.savefig('./tmp/{}'.format(out_dir))
    # plt.show()

    # cv2.imwrite(out_dir, cam_img)


def makeInput(imgName):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
    # 431
    img = cv2.imread('../source/extra/test/early/{}'.format(imgName), 0)
    glo_tensor = torch.unsqueeze(transform(img), 0).to('cuda:0').requires_grad_(True)
    return glo_tensor


if __name__ == '__main__':
    names = os.listdir('../source/extra/test/early')
    net = torchvision.models.resnet18(pretrained=False)
    net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    net.fc = nn.Linear(in_features=512, out_features=2, bias=True)
    net.load_state_dict(torch.load('../output/model/resnet18/net_4.pth'), strict=False)
    net.layer4[-1].register_forward_hook(farward_hook)
    net.layer4[-1].register_full_backward_hook(backward_hook)
    net = net.to('cuda:0')
    for imgName in names:
        xinput = makeInput(imgName)
        img = xinput[0, 0, :, :].detach().cpu().numpy()
        fmap_block = list()
        grad_block = list()
        net.eval()
        out = net(xinput)
        loss = out[0, out[0].argmax()]
        net.zero_grad()
        loss.backward()
        # 生成cam
        grads_val = grad_block[0].cpu().data.numpy().squeeze()
        fmap = fmap_block[0].cpu().data.numpy().squeeze()
        # 保存cam图片
        cam_show_img(img, fmap, grads_val, imgName)
        print(imgName, ' done')







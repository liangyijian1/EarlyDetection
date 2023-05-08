import os

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
    cam_img = 0.5 * heatmap + cv2.cvtColor(img * 255, cv2.COLOR_GRAY2RGB)
    # plt.subplot(1, 3, 1)
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
    # plt.subplot(1, 3, 2)
    # plt.imshow(heatmap)
    # plt.subplot(1, 3, 3)
    # plt.imshow(cam_img)
    # os.makedirs('./tmp', exist_ok=True)
    # plt.savefig('./tmp/{}'.format(out_dir))
    # plt.show()
    cv2.imwrite(str(1) + '-' + out_dir, cam_img)


def makeInput(imgName):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
    # 431
    patch_size = 56
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
    xinput.append(loc_tensor.to('cuda:0').requires_grad_(True))
    return xinput


if __name__ == '__main__':
    names = os.listdir('../source/testdir/test/early')
    for imgName in names:
        if imgName != 'early-1576.jpg':
            continue
        xinput = makeInput(imgName)
        img = xinput[0]
        loc_imgs = xinput[1]
        img = img[0, 0, :, :].detach().cpu().numpy()
        fmap_block = list()
        grad_block = list()
        backbone = 'vgg8'
        net = GlobalLocalTransformer(inplace=1, nblock=1, backbone=backbone).to('cuda:0')
        net.eval()
        if backbone == 'vgg8':
            net.local_feat.conv42.register_forward_hook(farward_hook)
            net.local_feat.conv42.register_backward_hook(backward_hook)
            # net.fftlist[0].conv2.conv1.register_forward_hook(farward_hook)
            # net.fftlist[0].conv2.conv1.register_backward_hook(backward_hook)
            # net.global_feat.conv42.register_forward_hook(farward_hook)
            # net.global_feat.conv42.register_backward_hook(backward_hook)
            net.load_state_dict(
                torch.load('../output/model/GlobalLocalTransformer/{}/nblock1/patches56/net_49.pth'.format(backbone)))
        elif backbone == 'octnet':
            net.local_feat.conv6.register_forward_hook(farward_hook)
            net.local_feat.conv6.register_backward_hook(backward_hook)
            net.load_state_dict(
                torch.load('../output/model/GlobalLocalTransformer/{}/nblock1/patches56/net_48.pth'.format(backbone)))
        out = net(xinput)
        g_out = out[0]
        l_out = out[1]
        p_patch = l_out[0]
        net.zero_grad()
        loss = p_patch[1]
        print('predict: {}, {}\n'.format(p_patch.argmax(), p_patch))
        loss.backward()
        # 生成cam
        grads_val = grad_block[0].cpu().data.numpy().squeeze()
        fmap = fmap_block[0].cpu().data.numpy().squeeze()
        # 保存cam图片
        for idx in range(loc_imgs.shape[1]):
            patchImg = loc_imgs[0, idx, :, :]
            cam_show_img(patchImg.detach().cpu().numpy(), fmap, grads_val, imgName)
        # cam_show_img(img, fmap, grads_val, '32.jpg')
        print(imgName, ' done')







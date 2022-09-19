import math
import os
import random
import sys
import traceback
import numpy as np
import cv2
import torch
import sympy
import torchvision
import shutil

from utils.utils import find_max_region, flatten, cropImg


class MarkLabel:

    def __init__(self, win_name: str, box_size: int):
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        self.win_name = win_name
        self.corners = []
        self.box_size = box_size

    def on_mouse_pick_points(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.corners.append((x, y))

    def markLabel(self, img, save_path: str, n_patch: int):
        ret = np.copy(img)
        show = np.copy(img)
        count = 0
        cv2.setMouseCallback(self.win_name, self.on_mouse_pick_points, ret)
        while True:
            cv2.imshow(self.win_name, show)
            key = cv2.waitKey(30)
            if key == 27:  # ESC
                break
            elif key == 13:  # enter
                start = self.corners[0]
                if start[0] > ret.shape[0] - self.box_size:
                    print(start)
                else:
                    end = [start[0] + self.box_size, start[1] + self.box_size]
                    cv2.rectangle(show, start, end, 255)
                    tmp = ret[start[1]:end[1], start[0]:end[0]]
                    cv2.imwrite(save_path + '/{}.jpg'.format(count.__str__()), tmp)
                    count += 1
                    print('保存成功')
                self.corners = []
            if count == n_patch:
                break
        cv2.destroyAllWindows()

    def validation(self, name, patch_num, patch_root_path):
        """
        MarkLabel('1', 40).validation('early', 9, '../source/testPatches40/')
        :param name:
        :param patch_num:
        :param patch_root_path:
        :return:
        """
        cv2.destroyAllWindows()
        patches_list = os.listdir(patch_root_path + name)
        for item in patches_list:
            tag = False
            patches = os.listdir(patch_root_path + name + '/' + item)
            if len(patches) != patch_num:
                tag = True
            for i in range(len(patches)):
                img = cv2.imread(patch_root_path + name + '/' + item + '/' + i.__str__() + '.jpg', 0)
                if img.shape[0] != self.box_size or img.shape[1] != self.box_size:
                    tag = True
            if tag:
                print(item)


def basicDenoise(rootPath: str, savePath: str):
    # basicDenoise('../source/original/', '../source/denoise/')
    imgPathNames = os.listdir(rootPath)
    for imgPathName in imgPathNames:
        imgNames = os.listdir(rootPath + imgPathName + '/')
        os.makedirs(savePath + imgPathName, exist_ok=True)
        for idx, imgName in enumerate(imgNames):
            try:
                if imgName[-3:] == 'jpg':
                    imgPath = rootPath + imgPathName + '/' + imgName
                    img = cv2.imread(imgPath, 0)
                    for i in range(35):
                        for j in range(img.shape[1]):
                            img[i][j] = 0
                    afterDenoise = cv2.fastNlMeansDenoising(img, h=10, templateWindowSize=7, searchWindowSize=21)
                    cv2.imwrite(savePath + imgPathName + '/' + imgName, afterDenoise)
                    print(imgName, ' 完成。还有{}个'.format(len(imgNames) - idx - 1))
            except Exception as e:
                with open('../output/failed.txt', 'a+') as f:
                    f.write('\n' + imgName + "处理时候发生错误，： " + e.__str__())
                print(imgName + "处理时候发生错误，： " + e.__str__())
                traceback.print_exc(file=sys.stdout)


def basicFlatten(img):
    tmp = np.copy(img)
    tmp = cv2.medianBlur(tmp, 17)
    ret, threshold = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imshow('1', threshold)
    cv2.waitKey(0)
    # stats对应的是x,y,width,height和面积， centroids为中心点， labels表示各个连通区在图上的表示
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(threshold, connectivity=8)
    # 选出背景的序号
    labelNum = 0
    for i in range(stats.shape[0]):
        if stats[i, 0] == 0 and stats[i, 1] == 0:
            labelNum = i
            break
    stats = np.delete(stats, [0], axis=0)
    num_labels = num_labels - 1
    # 将label列表中labelNum组全部置为0背景
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            if labels[i][j] == labelNum:
                labels[i][j] = 0
    output = np.zeros((threshold.shape[0], threshold.shape[1]), np.uint8)
    # 将图中的连通区域组合起来
    for i in range(1, num_labels + 1):
        mask = labels == i
        output[:, :][mask] = 255
    return output


def basicDataAugmentation(rootPath: str, imgPathNames: list, savePath: str):
    # basicDataAugmentation('../source/denoise/', ['early', 'normal'], '../source/basicAugmentation/')
    candidate_list = [
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ColorJitter(brightness=(0.8, 1.2)),
        torchvision.transforms.RandomAffine(0, translate=(0.05, 0.1), scale=(0.8, 1.2)),
        torchvision.transforms.RandomRotation(25),
    ]
    transform_list = [
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize(size=(256, 256)),
    ]
    for imgPathName in imgPathNames:
        imgNames = os.listdir(rootPath + imgPathName + '/')
        k = int(250 / len(imgNames))
        k = k if k > 1 else 1
        os.makedirs(savePath + imgPathName + '/', exist_ok=True)
        for idx, imgName in enumerate(imgNames):
            try:
                if imgName[-3:] == 'jpg':
                    img = cv2.imread(rootPath + imgPathName + '/' + imgName, 0)
                    count = 0
                    for i in range(k):
                        transform_list.append(random.choice(candidate_list))
                        transform = torchvision.transforms.Compose(transform_list)
                        tr = np.asarray(transform(img))
                        cv2.imwrite(savePath + imgPathName + '/' + count.__str__() + '-' + imgName, tr)
                        count += 1
                        transform_list.pop()
                    print(imgName, ' 完成。还有{}个'.format(len(imgNames) - idx - 1))
            except Exception as e:
                print(imgName + "处理时候发生错误，： " + e.__str__())
                traceback.print_exc(file=sys.stdout)


def basicTag(name, patches_root, imgs_root, box_size, n_patch):
    """
    basicTag('early','../source/patches56','../source/basicAugmentation',box_size=56,n_patch=6)
    :param name:
    :param patches_root:
    :param imgs_root:
    :return:
    """
    count = 0
    os.makedirs(patches_root + '/' + name, exist_ok=True)
    patches = os.listdir(patches_root + '/' + name)
    imgs = os.listdir(imgs_root + '/' + name)
    for item in imgs:
        count += 1
        if item[:-4] in patches:
            continue
        print(item + ',剩下 %d' % (len(imgs) - count))
        os.makedirs(patches_root + '/' + name + '/' + item[:-4], exist_ok=True)
        try:
            MarkLabel('1', box_size).markLabel(cv2.imread(imgs_root + '/' + name + '/' + item, 0),
                                               patches_root + '/' + name + '/' + item[:-4],
                                               n_patch=n_patch)
        except IndexError as e:
            print(item + '发生错误')
            MarkLabel('1', box_size).markLabel(cv2.imread(imgs_root + '/' + name + '/' + item, 0),
                                               patches_root + '/' + name + '/' + item[:-4],
                                               n_patch=n_patch)


def basicMove(sample_num):
    earlynames = os.listdir('../source/basicAugmentation/early')
    normalnames = os.listdir('../source/basicAugmentation/normal')
    imgNames = earlynames + normalnames
    random.shuffle(imgNames)
    testlist = random.sample(imgNames, sample_num)
    for item in testlist:
        shutil.move('../source/basicAugmentation/{}/{}'.format(item.split('-')[0], item),
                    '../source/test/{}/{}'.format(item.split('-')[0], item))
        shutil.move('../source/patches40/{}/{}'.format(item.split('-')[0], item[0:-4]),
                    '../source/testManualPatches/{}/{}'.format(item.split('-')[0], item[0:-4]))


def extractTestImagePatch(root_path: str,
                          patches_path: str,
                          deg: int,
                          patch_num: int,
                          box_size: int):
    """
    extractTestImagePatch(root_path='../source/test',
                          patches_path='../source/testPatches40',
                          deg=2,
                          patch_num=9,
                          box_size=40)
    """
    patch_num -= 1
    for _type in os.listdir(root_path):
        # _type = 'early'
        imgs = os.listdir(root_path + '/' + _type)
        for imgName in imgs:
            # if imgName != 'early-169.jpg':
            #     continue
            img = cv2.imread(root_path + '/' + _type + '/' + imgName, 0)
            location = []
            region = find_max_region(file=img, mbKSize=15)
            col = region.shape[1]
            row = region.shape[0]
            for i in range(col):
                temp = region[:, i]
                j = 0
                while j < row:
                    if temp[j] > 0:
                        location.append((j, i))
                        break
                    j += 1
            location = np.array(location)
            X = location[:, 1]
            y = location[:, 0]
            z1 = np.polyfit(X, y, deg)
            p1 = np.poly1d(z1)

            start_point = location[0]
            end_point = location[-1]
            a, b, c = p1.c[0], p1.c[1], p1.c[2]
            _x = sympy.symbols('x')
            f1 = sympy.sqrt(1 + 4 * a ** 2 * _x ** 2 + b ** 2 + 4 * a * b * _x)  # 求曲线长被积函数
            _s = sympy.Integral(f1, (_x, start_point[1], end_point[1])).doit().evalf()  # 整段曲线的长度
            step = math.floor(_s / patch_num)  # 均分后每段的长度
            # 求出所有cutpoint
            cutpoints = []  # 保存的是x值，对应于矩阵中的列数
            start_x = start_point[1]
            for i in range(patch_num):
                cutpoints.append(start_x)
                x_ = math.floor(sympy.nsolve(sympy.Integral(f1, (_x, start_x, _x)).doit() - step, 0))
                start_x = x_
            cutpoints.append(end_point[1])
            # 检查边界patch是否越界
            for idx, item in enumerate(cutpoints):
                l_border = math.ceil(box_size / 2)
                r_border = math.floor(img.shape[1] - box_size / 2)
                if item < l_border:
                    cutpoints[idx] = l_border
                if item > r_border:
                    cutpoints[idx] = r_border
            # 保存所有的patch
            c_img = np.copy(img)
            os.makedirs(patches_path + '/' + _type + '/' + imgName[:-4], exist_ok=True)
            for idx, cutpoint in enumerate(cutpoints):
                row = p1(cutpoint)
                if row < math.ceil(box_size / 2):
                    row = math.ceil(box_size / 2)
                if row > math.floor(img.shape[1] - box_size / 2):
                    row = math.floor(img.shape[1] - box_size / 2)
                current_point = [cutpoint, row]
                if idx == 0:
                    start = [int(current_point[0]), int(current_point[1] - box_size / 2)]
                    end = [int(current_point[0] + box_size), int(current_point[1] + box_size / 2)]
                elif idx == len(cutpoints) - 1:
                    start = [int(current_point[0] - box_size), int(current_point[1] - box_size / 2)]
                    end = [int(current_point[0]), int(current_point[1] + box_size / 2)]
                else:
                    start = [int(current_point[0] - box_size / 2), int(current_point[1] - box_size / 2)]
                    end = [int(start[0] + box_size), int(start[1] + box_size)]
                tmp = img[start[1]:end[1], start[0]:end[0]]
                cv2.imwrite(patches_path + '/' + _type + '/' + imgName[:-4] + '/{}.jpg'.format(idx.__str__()), tmp)
                cv2.rectangle(c_img, start, end, 255)
            print(_type + '/' + imgName)

def extractNormalImagePatch():
    n_patch = 5
    box_size = 56
    patches_path = '../source/patches56'
    root_path = '../source/basicAugmentation'
    for _type in os.listdir(root_path):
        _type = 'normal'
        imgs = os.listdir(root_path + '/' + _type)
        for imgName in imgs:
            # if imgName != 'normal-2034.jpg':
            #     continue
            img = cv2.imread(root_path + '/' + _type + '/' + imgName, 0)
            location = []
            region = find_max_region(file=img, mbKSize=15)
            col = region.shape[1]
            row = region.shape[0]
            for i in range(col):
                temp = region[:, i]
                j = 0
                while j < row:
                    if temp[j] > 0:
                        location.append((j, i))
                        break
                    j += 1
            location = np.array(location)
            X = location[:, 1]
            y = location[:, 0]
            z1 = np.polyfit(X, y, 2)
            p1 = np.poly1d(z1)

            start_point = location[0]
            end_point = location[-1]
            a, b, c = p1.c[0], p1.c[1], p1.c[2]
            _x = sympy.symbols('x')
            f1 = sympy.sqrt(1 + 4 * a ** 2 * _x ** 2 + b ** 2 + 4 * a * b * _x)  # 求曲线长被积函数
            _s = sympy.Integral(f1, (_x, start_point[1], end_point[1])).doit().evalf()  # 整段曲线的长度
            step = math.floor(_s / n_patch)  # 均分后每段的长度
            # 求出所有cutpoint
            cutpoints = []  # 保存的是x值，对应于矩阵中的列数
            start_x = start_point[1]
            for i in range(n_patch):
                cutpoints.append(start_x)
                x_ = math.floor(sympy.nsolve(sympy.Integral(f1, (_x, start_x, _x)).doit() - step, 0))
                start_x = x_
            cutpoints.append(end_point[1])
            # 检查边界patch是否越界
            for idx, item in enumerate(cutpoints):
                l_border = math.ceil(box_size / 2)
                r_border = math.floor(img.shape[1] - box_size / 2)
                if item < l_border:
                    cutpoints[idx] = l_border
                if item > r_border:
                    cutpoints[idx] = r_border
            # 保存所有的patch
            c_img = np.copy(img)
            os.makedirs(patches_path + '/' + _type + '/' + imgName[:-4], exist_ok=True)
            for idx, cutpoint in enumerate(cutpoints):
                row = p1(cutpoint)
                if row < math.ceil(box_size / 2):
                    row = math.ceil(box_size / 2)
                if row > math.floor(img.shape[1] - box_size / 2):
                    row = math.floor(img.shape[1] - box_size / 2)
                current_point = [cutpoint, row]
                if idx == 0:
                    start = [int(current_point[0]), int(current_point[1] - box_size / 2)]
                    end = [int(current_point[0] + box_size), int(current_point[1] + box_size / 2)]
                elif idx == len(cutpoints) - 1:
                    start = [int(current_point[0] - box_size), int(current_point[1] - box_size / 2)]
                    end = [int(current_point[0]), int(current_point[1] + box_size / 2)]
                else:
                    start = [int(current_point[0] - box_size / 2), int(current_point[1] - box_size / 2)]
                    end = [int(start[0] + box_size), int(start[1] + box_size)]
                tmp = img[start[1]:end[1], start[0]:end[0]]
                cv2.imwrite(patches_path + '/' + _type + '/' + imgName[:-4] + '/{}.jpg'.format(idx.__str__()), tmp)
                cv2.rectangle(c_img, start, end, 255)
            cv2.imwrite('./temp/{}'.format(imgName), c_img)
            print(_type + '/' + imgName)


if __name__ == '__main__':
    # 24 40 56 72 88  2034
    basicTag('early',
             '../source/patches56',
             '../source/basicAugmentation',
             box_size=56,
             n_patch=6)
    # MarkLabel('1', 56).validation('normal', 6, '../source/patches56/')
    # extractNormalImagePatch()
    #
    # extractTestImagePatch(root_path='../source/testdir/test',
    #                       patches_path='../source/testdir/test56',
    #                       deg=2,
    #                       patch_num=3,
    #                       box_size=56)

    # from pytorch_grad_cam import GradCAM
    # import torch.nn as nn
    # from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    # from pytorch_grad_cam.utils.image import show_cam_on_image
    # import matplotlib.pyplot as plt
    # transform = torchvision.transforms.Compose([
    #     torchvision.transforms.ToTensor(),
    #     torchvision.transforms.Grayscale()
    # ])
    # net = torchvision.models.resnet18(pretrained=True)
    # net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # net.fc = nn.Linear(in_features=512, out_features=2, bias=True)
    # net.load_state_dict(torch.load('../output/model/resnet18/net_15.pth'))
    #
    # # img = cv2.imread('../source/testdir/test/early/early-587.jpg')
    # img = cv2.imread('../source/testdir/test/early/early-540.jpg')
    # img_tensor = transform(img)
    # input_tensor = torch.unsqueeze(img_tensor, 0)
    #
    # target_layers = [net.layer4[-1]]
    # cam = GradCAM(model=net, target_layers=target_layers, use_cuda=True)
    # targets = [ClassifierOutputTarget(1)]
    #
    # grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    # grayscale_cam = grayscale_cam[0, :]
    # visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255., grayscale_cam, use_rgb=True)
    # plt.subplot(1, 2, 1)
    # plt.imshow(visualization)
    #
    # plt.subplot(1, 2, 2)
    # plt.imshow(grayscale_cam)
    # plt.show()







import functools
import operator
import sys

from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

sys.path.append('/root/autodl-octnet/eqrly/src')
sys.path.append('/root/autodl-octnet/eqrly/src/net/GlobalLocal')
from pytorch_grad_cam import GradCAM
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils.utils import confusionMatrixDisplay
import numpy as np
import os
import cv2
import torch
import torchvision.transforms
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from net.GlobalLocal.GlobalLocalTransformer import GlobalLocalTransformer
from net.GlobalLocal.loss import LinkCrossEntropy

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('device: ' + device)
writer = SummaryWriter(log_dir='../output/scalar/train')


class GlobalLocalTransformerDataset(Dataset):
    def __init__(self, patches_root_path, src_root_path, transform, patch_size):
        # root_path指的是每张图片对应的patches存放文件夹位置
        super(GlobalLocalTransformerDataset, self).__init__()
        self.transform = transform
        self.src_root_path = src_root_path
        self.patch_size = patch_size
        imgs = []
        self.patches_dict = {}
        self.label_dict = {}

        category_dirs = os.listdir(patches_root_path)
        for category in category_dirs:
            if os.path.isdir(patches_root_path + '/' + category):
                img_dir_names = os.listdir(patches_root_path + '/' + category)
                for patches_dir_name in img_dir_names:
                    if category == 'early':
                        self.label_dict[patches_dir_name + '.jpg'] = 1
                    else:
                        self.label_dict[patches_dir_name + '.jpg'] = 0
                    current_img_names = os.listdir(patches_root_path + '/' + category + '/' + patches_dir_name)
                    patches_list = []
                    for img_name in current_img_names:
                        patch_path = patches_root_path + '/' + category + '/' + patches_dir_name + '/' + img_name
                        patches_list.append(patch_path)
                    self.patches_dict[patches_dir_name] = patches_list
        print('Dataset init done\n')

    def __getitem__(self, index):
        patch_transform = torchvision.transforms.ToTensor()
        img_name = list(self.label_dict.keys())[index]
        img = self.transform(cv2.imread(self.src_root_path + '/' + img_name.split('-')[0] + '/' + img_name, 0))
        patch_path_list = self.patches_dict[img_name[:-4]]
        patches = torch.zeros((len(patch_path_list), self.patch_size, self.patch_size))
        for i, patch_path in enumerate(patch_path_list):
            patch = patch_transform(cv2.imread(patch_path, 0))
            patches[i] = patch

        return [img, patches], self.label_dict[img_name]

    def __len__(self):
        return len(self.label_dict)


def GlobalLocalTransformerTrain(train_loader: DataLoader,
                                epoch,
                                loss_fun,
                                lr,
                                patches_size,
                                backbone='vgg8',
                                pre_trained=None,
                                n_block=3):
    net = GlobalLocalTransformer(inplace=1, nblock=n_block, backbone=backbone).to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    avg_loss = []
    pthNumber = 0
    if pre_trained is not None:
        pthNumber = int(pre_trained.split('/')[-1][:-4].split('_')[1])
        preDict = torch.load(pre_trained)
        net.load_state_dict(preDict)
    with open('../output/backbone-{}-{}.txt'.format(backbone, patches_size), 'w+') as f:
        for k in range(pthNumber, epoch + pthNumber):
            print('\nEpoch: %d' % (k + 1))
            net.train()
            sum_loss = 0.0
            for i, train_data in enumerate(train_loader):
                inputs, labels = train_data  # 此处inputs应当包含了global和patches， labels也是对应的
                glo_inputs, patches = inputs[0].to(device), inputs[1].to(device)
                glo_label, loc_label = labels.to(device), labels.to(device)
                out = net(glo_inputs, patches)
                glo_out = out[0]
                loc_out = out[1:]
                loss = loss_fun(glo_out, loc_out, glo_label, loc_label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                sum_loss += loss.item()
                _iter = i + 1 + k * len(train_loader)
                print('Epoch: %d | Iter: %d | Loss: %.05f' % (k + 1, _iter, sum_loss / (i + 1)))
                f.write(
                    'Epoch: %d | Iter: %d | Loss: %.05f' % (k + 1, _iter, sum_loss / (i + 1)))
                f.write('\n')
                writer.add_scalar('train_loss', sum_loss / (i + 1), _iter)
            avg_loss.append('Epoch: %d | Average loss: %.05f' % (k + 1, sum_loss / (i + 1)))
            print('saving model')
            os.makedirs('../output/model/GlobalLocalTransformer/{}/patches{}/'.format(backbone, patches_size),
                        exist_ok=True)
            torch.save(net.state_dict(),
                       '../output/model/GlobalLocalTransformer/{}/patches{}/net_{}.pth'.format(backbone, patches_size,
                                                                                               k + 1))


def GlobalLocalTransformerTest(test_loader,
                               pth: str,
                               backbone: str,
                               loss_fun,
                               n_block,
                               s_test=False):
    net = GlobalLocalTransformer(inplace=1, nblock=n_block, backbone=backbone).to(device)
    net.load_state_dict(torch.load(pth))
    n_test = 1
    if not s_test:
        n_test = 10
    acc = []
    pre = []
    rec = []
    F1 = []
    with torch.no_grad():
        for k in range(n_test):
            y_pred = []
            y_true = []
            for i, test_data in enumerate(test_loader):
                net.eval()
                _inputs, _labels = test_data
                _inputs[0] = _inputs[0].to(device)
                _inputs[1] = _inputs[1].to(device)
                _glo_label, _loc_label = _labels.to(device), _labels.to(device)
                _out = net(_inputs)
                _glo_out = _out[0]
                _loc_out = _out[1:]
                _loss = loss_fun(_glo_out, _loc_out, _glo_label, _loc_label)
                _glo_predict = torch.argmax(_glo_out, dim=1)
                _loc_predicts = torch.zeros(len(_labels))
                for p, item in enumerate(_loc_out):
                    tmp = torch.argmax(item, dim=1)
                    for q, patch in enumerate(tmp):
                        if patch > 0:
                            _loc_predicts[q] = 1
                y_pred.append(_loc_predicts.cpu().numpy().astype(np.int_).tolist())
                y_true.append(_labels.cpu().numpy().tolist())
            y_pred = np.array(functools.reduce(operator.concat, y_pred), dtype=np.int64)
            y_true = np.array(functools.reduce(operator.concat, y_true), dtype=np.int64)
            # confusionMatrixDisplay(y_true, y_pred)
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted')
            recall = recall_score(y_true, y_pred, average='weighted')
            f1 = f1_score(y_true, y_pred, average='weighted')
            if not s_test:
                acc.append(accuracy)
                pre.append(precision)
                rec.append(recall)
                F1.append(f1)
                print('{} test'.format((k + 1).__str__()))
        with open('../output/Evaluation.txt', 'a+') as f:
            if not s_test:
                f.write(
                    'backbone-{}\tTotal Sample:{}  {}\naccuracy: best={:.3f}\tworst={:.3f}\taverage={:.3f}\nprecision: best={'
                    ':.3f}\tworst={:.3f}\taverage={:.3f}\nrecall: best={:.3f}\tworst={:.3f}\taverage={'
                    ':.3f}\nf1_score: best={:.3f}\tworst={:.3f}\taverage={:.3f}\n'
                    .format(backbone, len(y_pred), pth[-9:], max(acc), min(acc), sum(acc) / len(acc), max(pre),
                            min(pre), sum(pre) /
                            len(pre), max(rec), min(rec), sum(rec) / len(rec), max(F1), min(F1), sum(F1) / len(F1)))
            else:
                f.write(
                    'backbone-{}\n\tTotal Sample:{}, accuracy:{:.3f}, precision:{:.3f}, recall:{:.3f}, f1_score:{:.3f}\n'
                    .format(len(y_pred), accuracy, precision, recall, f1))


def ComparisonExperimentTrain(model: str,
                              train_loader: DataLoader,
                              loss_fun,
                              lr: float,
                              epoch: int,
                              pre_trained=None):
    net = None
    pthNumber = 0
    if model == 'vgg11':
        net = torchvision.models.vgg11(pretrained=True)
        net.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        net.add_module('add_linear', nn.Linear(1000, 2))
    elif model == 'resnet18':
        net = torchvision.models.resnet18(pretrained=True)
        net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        net.fc = nn.Linear(in_features=512, out_features=2, bias=True)
    elif model == 'micronet':
        pass
        # net = micronet()
        # net.features[0].stem[0] = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    elif model == 'efficientnet_b0':
        net = torchvision.models.efficientnet_b0(pretrained=True)
        net.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        net.classifier[1] = nn.Linear(in_features=1280, out_features=2, bias=True)
    elif model == 'efficientnet_b7':
        net = torchvision.models.efficientnet_b7(pretrained=True)
        net.features[0][0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        net.classifier[1] = nn.Linear(in_features=2560, out_features=2, bias=True)
    elif model == 'inception_v3':
        net = torchvision.models.inception_v3(True, True, transform_input=False)
        net.Conv2d_1a_3x3.conv = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
        net.fc = nn.Linear(in_features=2048, out_features=2, bias=True)
    if pre_trained is not None:
        pthNumber = pre_trained.split('/')[-1][:-4].split('_')[1]
        preDict = torch.load(pre_trained)
        net.load_state_dict(preDict)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    net = net.to(device)
    with open('../output/{}.txt'.format(model), 'w+') as f:
        for k in range(pthNumber, epoch + pthNumber):
            print('\nEpoch: %d' % (k + 1))
            net.train()
            sum_loss = 0.0
            for i, train_data in enumerate(train_loader):
                inputs, labels = train_data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                if model == 'inception_v3':
                    outputs = outputs[0]
                loss = loss_fun(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                sum_loss += loss.item()
                _iter = i + 1 + k * len(train_loader)
                print('Epoch: %d | Iter: %d | Loss: %.05f' % (k + 1, _iter, sum_loss / (i + 1)))
                f.write(
                    'Epoch: %d | Iter: %d | Loss: %.05f' % (k + 1, _iter, sum_loss / (i + 1)))
                f.write('\n')
                writer.add_scalar('train_loss', sum_loss / (i + 1), _iter)
            print('saving model')
            os.makedirs('../output/model/{}'.format(model), exist_ok=True)
            torch.save(net.state_dict(), '../output/model/{}/net_{}.pth'.format(model, k + 1))


def ComparisonExperimentTest(model: str,
                             pth: str,
                             test_loader,
                             loss_fun,
                             s_test=False):
    net = None
    if model == 'vgg11':
        net = torchvision.models.vgg11(pretrained=True)
        net.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        net.add_module('add_linear', nn.Linear(1000, 2))
    elif model == 'resnet18':
        net = torchvision.models.resnet18(pretrained=True)
        net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        net.fc = nn.Linear(in_features=512, out_features=2, bias=True)
    elif model == 'micronet':
        pass
        # net = micronet()
        # net.features[0].stem[0] = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    elif model == 'efficientnet_b0':
        net = torchvision.models.efficientnet_b0(pretrained=True)
        net.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        net.classifier[1] = nn.Linear(in_features=1280, out_features=2, bias=True)
    elif model == 'efficientnet_b7':
        net = torchvision.models.efficientnet_b7(pretrained=True)
        net.features[0][0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        net.classifier[1] = nn.Linear(in_features=2560, out_features=2, bias=True)
    elif model == 'inception_v3':
        net = torchvision.models.inception_v3(True, True, transform_input=False)
        net.Conv2d_1a_3x3.conv = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
        net.fc = nn.Linear(in_features=2048, out_features=2, bias=True)
    net.load_state_dict(torch.load(pth))
    net = net.to(device)
    n_test = 1
    if not s_test:
        n_test = 10
    acc = []
    pre = []
    rec = []
    F1 = []
    with torch.no_grad():
        for k in range(n_test):
            y_pred = []
            y_true = []
            for i, test_data in enumerate(test_loader):
                net.eval()
                _inputs, _labels = test_data
                _inputs, _labels = _inputs.to(device), _labels.to(device)
                _outputs = net(_inputs)
                # if model == 'inception_v3':
                #     _outputs = _outputs[0]
                _loc_predicts = torch.argmax(_outputs, 1)
                _loss = loss_fun(_outputs, _labels)
                y_pred.append(_loc_predicts.cpu().numpy().astype(np.int_).tolist())
                y_true.append(_labels.cpu().numpy().tolist())
            y_pred = np.array(functools.reduce(operator.concat, y_pred), dtype=np.int64)
            y_true = np.array(functools.reduce(operator.concat, y_true), dtype=np.int64)
            # confusionMatrixDisplay(y_true, y_pred)
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted')
            recall = recall_score(y_true, y_pred, average='weighted')
            f1 = f1_score(y_true, y_pred, average='weighted')
            if not s_test:
                acc.append(accuracy)
                pre.append(precision)
                rec.append(recall)
                F1.append(f1)
                print('{} test'.format((k + 1).__str__()))
        with open('../output/Evaluation.txt', 'a+') as f:
            if not s_test:
                f.write(
                    '{}\tTotal Sample:{}  {}\naccuracy: best={:.3f}\tworst={:.3f}\taverage={:.3f}\nprecision: best={'
                    ':.3f}\tworst={:.3f}\taverage={:.3f}\nrecall: best={:.3f}\tworst={:.3f}\taverage={'
                    ':.3f}\nf1_score: best={:.3f}\tworst={:.3f}\taverage={:.3f}\n'
                    .format(model, len(y_pred), pth[-10:], max(acc), min(acc), sum(acc) / len(acc), max(pre), min(pre),
                            sum(pre) /
                            len(pre), max(rec), min(rec), sum(rec) / len(rec), max(F1), min(F1), sum(F1) / len(F1)))
            else:
                f.write('{}\n\tTotal Sample:{}, accuracy:{:.3f}, precision:{:.3f}, recall:{:.3f}, f1_score:{:.3f}\n'
                        .format(model, len(y_pred), accuracy, precision, recall, f1))


def global_local(is_test: bool = False,
                 backbone: str = 'vgg8',
                 patch_size: int = 24,
                 n_block: int = 3):
    transform = torchvision.transforms.ToTensor()
    print('backbone: {} | patch size: {} | n_block: {}'.format(backbone, patch_size, n_block))

    if not is_test:  # Train
        trainDataset = GlobalLocalTransformerDataset('../source/patches{}'.format(patch_size),
                                                     '../source/basicAugmentation',
                                                     transform=transform,
                                                     patch_size=patch_size)
        GlobalLocalTransformerTrain(train_loader=DataLoader(trainDataset, batch_size=15, shuffle=True, drop_last=True),
                                    epoch=80,
                                    loss_fun=LinkCrossEntropy(nn.CrossEntropyLoss()),
                                    lr=0.001,
                                    patches_size=patch_size,
                                    backbone=backbone,
                                    # pre_trained='../output/model/GlobalLocalTransformer/efficientnet_b0/patches{}/net_47.pth'.format(patch_size),
                                    pre_trained=None,
                                    n_block=n_block
                                    )
    else:  # Test
        testDataset = GlobalLocalTransformerDataset('../source/testdir/testPatches{}'.format(patch_size),
                                                    '../source/testdir/test',
                                                    transform=transform,
                                                    patch_size=patch_size)
        for i in range(80, 81):
            print('current pth is:{}'.format(i.__str__()))
            GlobalLocalTransformerTest(test_loader=DataLoader(testDataset, 6, shuffle=False),
                                       # nblock1/patches24
                                       pth='../output/model/GlobalLocalTransformer/{}/nblock{}/patches{}/net_{}.pth'.format(
                                           backbone, n_block, patch_size, i),
                                       loss_fun=LinkCrossEntropy(nn.CrossEntropyLoss()),
                                       backbone=backbone,
                                       n_block=n_block)


def comparison_experiment(is_test: bool = False, model: str = 'vgg11'):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(299) if model == 'inception_v3' else torchvision.transforms.Compose([]),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Grayscale()
    ])
    if not is_test:  # Train
        trainDataset = torchvision.datasets.ImageFolder('../source/extra/basicAugmentation', transform)
        ComparisonExperimentTrain(model,
                                  train_loader=DataLoader(trainDataset, 2, shuffle=True),
                                  epoch=50,
                                  loss_fun=nn.CrossEntropyLoss(),
                                  lr=0.001
                                  )
    else:  # Test
        testDataset = torchvision.datasets.ImageFolder('../source/extra/test', transform)
        for i in range(1, 35):
            print('current pth is:{}'.format(i.__str__()))
            ComparisonExperimentTest(model,
                                     '../output/model/{}/net_{}.pth'.format(model, i),
                                     test_loader=DataLoader(testDataset, 1, shuffle=True),
                                     loss_fun=nn.CrossEntropyLoss())


if __name__ == '__main__':
    global_local(is_test=True,
                 backbone='octnet',
                 patch_size=40,
                 n_block=1)
    # comparison_experiment(model='resnet18',
    #                       is_test=True)

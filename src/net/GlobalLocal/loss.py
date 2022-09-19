# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

class LinkCrossEntropy(nn.Module):
    def __init__(self, loss_fun):
        super().__init__()
        self.loss_fun = loss_fun

    def forward(self, gloout, locout, glo_label, loc_labels):
        """

        :param gloout: global path输出，size = batch * 2
        :param locout: local path输出
        :param glo_label:
        :param loc_labels:
        :return:
        """
        loss = torch.zeros(len(locout) + 1)
        glo_loss = self.loss_fun(gloout, glo_label)
        loss[0] = glo_loss
        for i in range(len(locout)):
            loc_loss = self.loss_fun(locout[i], loc_labels)
            loss[i + 1] = loc_loss

        return loss.mean()



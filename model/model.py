import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import math
import modules
import torch.utils.model_zoo as model_zoo

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.feature =  modules.resnet50(pretrained=True)
        # self.feature.load_state_dict(torch.load(pretrained_url), strict=False )

        self.gazeEs = modules.ResGazeEs()
        # self.gazeEs.load_state_dict(torch.load(pretrained_url), strict=False )

        self.deconv = modules.ResDeconv(modules.BasicBlock)

    def forward(self, x_in, require_img=True):
        features = self.feature(x_in["face"])
        gaze = self.gazeEs(features)
        if require_img:
          img = self.deconv(features)
          img = torch.sigmoid(img)
        else:
          img = None
        return gaze, img

class Gelossop():
    def __init__(self, attentionmap, w1=1, w2=1):
        self.gloss = torch.nn.L1Loss().cuda()
        #self.gloss = torch.nn.MSELoss().cuda()
        self.recloss = torch.nn.MSELoss().cuda()
        self.attentionmap = attentionmap.cuda()
        self.w1 = w1
        self.w2 = w2
        

    def __call__(self, gaze, img, gaze_pre, img_pre):
        loss1 = self.gloss(gaze, gaze_pre)
        # loss2 = 1-self.recloss(img, img_pre)
        loss2 = 1 - (img - img_pre)**2
        zeros = torch.zeros_like(loss2)
        loss2 = torch.where(loss2 < 0.75, zeros, loss2)
        loss2 = torch.mean(self.attentionmap * loss2)

        return self.w1 * loss1 + self.w2 * loss2

class Delossop():
    def __init__(self):
        self.recloss = torch.nn.MSELoss().cuda()

    def __call__(self, img, img_pre):
        return self.recloss(img, img_pre)



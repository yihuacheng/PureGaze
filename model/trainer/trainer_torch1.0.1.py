import os, sys
sys.path.insert(0, os.getcwd())
sys.path.insert(0, "/home/chengyihua/utils/")
import model
import importlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import yaml
import cv2
import ctools
from easydict import EasyDict as edict
import torch.backends.cudnn as cudnn
import argparse


def main(train):

    # Setup-----------------------------------------------------------
    dataloader = importlib.import_module(f"reader.{config.reader}")

    torch.cuda.set_device(config.device)

    attentionmap = cv2.imread(config.map, 0)/255
    attentionmap = torch.from_numpy(attentionmap).type(torch.FloatTensor)

    data = config.data
    save = config.save
    params = config.params


    # Prepare dataset-------------------------------------------------
    dataset = dataloader.loader(data, params.batch_size, shuffle=True, num_workers=4)


    # Build model
        # build model ------------------------------------------------
    print("===> Model building <===")
    net = model.Model(); net.train(); net.cuda()

    if config.pretrain:
        net.load_state_dict(torch.load(config.pretrain), strict=False)

    print("optimizer building")
    geloss_op = model.Gelossop(attentionmap, w1=3, w2=1)
    deloss_op = model.Delossop()

    ge_optimizer = optim.Adam(net.feature.parameters(),
             lr=params.lr, betas=(0.9,0.95))

    ga_optimizer = optim.Adam(net.gazeEs.parameters(), 
             lr=params.lr, betas=(0.9,0.95))

    de_optimizer = optim.Adam(net.deconv.parameters(), 
            lr=params.lr, betas=(0.9,0.95))

    # scheduler = optim.lr_scheduler.StepLR(optimizer,
            #step_size=params.decay_step, gamma=params.decay)

    # prepare for training ------------------------------------

    length = len(dataset);
    total = length * params.epoch

    savepath = os.path.join(save.metapath, save.folder, f"checkpoint")

    if not os.path.exists(savepath):
        os.makedirs(savepath)

    timer = ctools.TimeCounter(total)

  
    print("Traning")
    with open(os.path.join(savepath, "train_log"), 'w') as outfile:
        for epoch in range(1, config["params"]["epoch"]+1):
            for i, (data, label) in enumerate(dataset):

                # Acquire data
                data["face"] = data["face"].cuda()
                label = label.cuda()
 
                # forward
                gaze, img = net(data)

                # loss calculation
                geloss = geloss_op(gaze, img, label, data["face"])

                ge_optimizer.zero_grad()
                ga_optimizer.zero_grad()
                geloss.backward(retain_graph=True)
                ge_optimizer.step()
                ga_optimizer.step()

                deloss = deloss_op(img, data["face"])
                de_optimizer.zero_grad()

                deloss.backward()
                de_optimizer.step()
                
                rest = timer.step()/3600

                # print logs
                if i % 20 == 0:
                    log = f"[{epoch}/{params.epoch}]: " + \
                          f"[{i}/{length}] " +\
                          f"gloss:{geloss} " +\
                          f"dloss:{deloss} " +\
                          f"lr:{params.lr} " +\
                          f"rest time:{rest:.2f}h"

                    print(log)
                    outfile.write(log + "\n")
                    sys.stdout.flush()   
                    outfile.flush()

            if epoch % config["save"]["step"] == 0:
                torch.save(net.state_dict(), os.path.join(savepath, f"Iter_{epoch}_{save.name}.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pytorch Basic Model Training')

    parser.add_argument('-c', '--config', type=str,
                        help='Path to the config file.')

    args = parser.parse_args()

    config = edict(yaml.load(open(args.config), Loader=yaml.FullLoader))

    main(config)
 

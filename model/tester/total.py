import os, sys
sys.path.insert(0, os.getcwd())
sys.path.insert(0, "/home/chengyihua/utils/")
import model
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2, yaml, copy
from importlib import import_module
from easydict import EasyDict as edict
import ctools, gtools
import argparse

    
def gazeto3d(gaze):
	# Only used for ETH, which conduct gaze as [pitch yaw].
  	assert gaze.size == 2, "The size of gaze must be 2"
  	gaze_gt = np.zeros([3])
  	gaze_gt[0] = -np.cos(gaze[0]) * np.sin(gaze[1])
  	gaze_gt[1] = -np.sin(gaze[0])
  	gaze_gt[2] = -np.cos(gaze[0]) * np.cos(gaze[1])
  	return gaze_gt


def main(train, test):

    # prepare parameters for test ---------------------------------------
    reader = import_module(f"reader.{test.reader}")

    data = test.data
    load = test.load
    torch.cuda.set_device(test.device)

    modelpath = os.path.join(train.save.metapath,
            train.save.folder, f"checkpoint")

    logpath = os.path.join(train.save.metapath, 
            train.save.folder, f"{test.savename}")

    if not os.path.exists(logpath):
        os.makedirs(logpath)

    if data.isFolder:
        data, _ = ctools.readfolder(data)
        print(data)

    # Read data -------------------------------------------------------- 
    dataset = reader.loader(data, 32, num_workers=4, shuffle=False)

    # Test-------------------------------------------------------------
    begin = load.begin; end = load.end; step = load.steps

    for saveiter in range(begin, end+step, step):
        print(f"Test: {saveiter}")

        # Load model --------------------------------------------------
        net = model.Model()

        modelname = f"Iter_{saveiter}_{train.save.name}.pt"

        statedict = torch.load( os.path.join(modelpath, modelname),
                    map_location={f"cuda:{train.device}":f"cuda:{test.device}"})

        net.cuda(); net.load_state_dict(statedict); net.eval()

        length = len(dataset); accs = 0; count = 0

        # Open log file ------------------------------------------------
        logname = f"{saveiter}.log"

        outfile = open(os.path.join(logpath, logname), 'w')
        outfile.write("name results gts\n")


        count = 0
        # Testing --------------------------------------------------------------
        with torch.no_grad():

            for j, (data, label) in enumerate(dataset):

                for key in data:
                    if key != 'name': data[key] = data[key].cuda()

                # Read data and predit--------------------------------------------
                names =  data["name"]
                gts = label.cuda()
               
                results, _ = net(data, require_img=False)

                # Cal error between each pair of result and gt ------------------
                for k, result in enumerate(results):

                    result = result.cpu().detach().numpy()
                    gt = gts[k].cpu().numpy()

                    accs += gtools.angular(gtools.gazeto3d(gt),
                            gazeto3d(result))
            
                    count += 1

                    name   = [data['name'][k]]
                    result = [str(u) for u in result] 
                    gt     = [str(u) for u in gt]
                   
                    log = name + [",".join(result)]  +  [",".join(gt)]

                    outfile.write(" ".join(log) + "\n")
            log = f"[{saveiter}] Total Num: {count}, avg: {accs/count}"
            outfile.write(log)
            print(log)

        outfile.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Pytorch Basic Model Training')

    parser.add_argument('-s', '--source', type=str,
                        help = 'config path about training')

    parser.add_argument('-t', '--target', type=str,
                        help = 'config path about test')

    args = parser.parse_args()

    # Read model from train config and Test data in test config.
    train_conf = edict(yaml.load(open(args.source), Loader=yaml.FullLoader))

    test_conf = edict(yaml.load(open(args.target), Loader=yaml.FullLoader))

    print("=======================>(Begin) Config of training<======================")
    print(ctools.DictDumps(train_conf))
    print("=======================>(End) Config of training<======================")
    print("")
    print("=======================>(Begin) Config for test<======================")
    print(ctools.DictDumps(test_conf))
    print("=======================>(End) Config for test<======================")

    main(train_conf, test_conf)




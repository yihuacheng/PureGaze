# PureGaze: Purifying Gaze Feature for Generalizable Gaze Estimation
![License CC BY-NC](https://img.shields.io/badge/license-CC_BY--NC-DodgerBlue.svg?style=plastic)

## Description 

Our work is accpeted by **AAAI 2022**.

<div align=center>  <img src="figures/overview.png" alt="overview" width="500" align="bottom" /> </div>

**Picture:** We propose a domain-generalization framework for gaze estimation. 
Our method is only trained in the source domain and brings improvement in *all unknown target domains*.
The key idea of our method is to purify the gaze feature with a self-adversarial framework.

<div align=center>  <img src="figures/pipeline.png" alt="pipeline" width="800" align="bottom" /> </div>  

**Picture:** Overview of the gaze feature purification. Our goal is to preserve the gaze-relevant feature and eliminate gaze-irrelevant
features. We define two tasks, which are to preserve gaze information and to remove general facial image
information. The two tasks are not cooperative but adversarial to purify feature. Simultaneously optimizing the two tasks, we
implicitly purify the gaze feature without defining gaze-irrelevant feature.

<div align=center>  <img src="figures/performance.png" alt="performance" width="800" align="bottom" /> </div>

**Performance:** PureGaze shows best performance among typical gaze estimation methods (w/o adaption), and has competitive result among domain adaption methods. Note that, PureGaze learns one optimal model for four tasks, while domain adaption methods need to learn a total of four models. This is an advantage of PureGaze.

<div align=center>  <img src="figures/visual.png" alt="visualization" width="800" align="bottom" /> </div>

**Feature visualization:** The result clearly explains the purification. Our purified feature contains less gaze-irrelevant feature and naturally improves the cross-domain performance.

## Usage

This is a re-implemented version by Pytorch1.7.1 (origin is Pytorch1.0.1). 

We provides an Res50-Version PureGaze. 
If you want to change the backbone to Res18, you could use the file in `Model/Res18`.

### Resourse

`Model/`: Implemented code.  
`Masker/`: The masker used for training.

### Get Started

1. You could find data processing code from [this link](https://github.com/yihuacheng/GazeEstimation-Summary).

2. modifing files in `config/` folder, and run commands like: 

    **Training:**`python trainer/trainer.py -c config/train/config-eth.yaml`

    **Test:**`python tester/total.py -s config/train/config-eth.yaml -t config/test/config-mpii.yaml`

    **Visual:**`python tester/visual.py -s config/train/config-eth.yaml -t config/test/config-mpii.yaml`

### Pre-trained model.
We provide a pre-trained model of Res50-version PureGaze.
You can find it from [this link](https://github.com/yihuacheng/GazeEstimation-Summary).

### Warning.
If you encounter an error of approximately 20 degrees in the ETH-XGaze dataset, it may be due to the disparity in gaze format between ETH-XGaze, which presents gaze as (Pitch Yaw), and our processed datasets including MPIIGaze, EyeDiap and Gaze360, which present gaze as (Yaw Pitch). To address this, please make the necessary modifications to the `gazeto3d(gaze)` function during testing.

### Citation.
```
@article{cheng2022puregaze,
  title={PureGaze: Purifying Gaze Feature for Generalizable Gaze Estimation},
  author={Yihua Cheng and Yiwei Bao and Feng Lu},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2022}
}
```

### Contact
Please email y.cheng.2@bham.ac.uk

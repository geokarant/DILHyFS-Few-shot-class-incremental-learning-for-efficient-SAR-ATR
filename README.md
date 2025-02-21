## DILHyFS  
## Install
```
conda create -n DILHyFS python=3.9
conda activate DILHyFS 
pip install -r requirements.txt
```
## Dependencies 
This code is implemented in PyTorch, and we perform the experiments under the following environment settings:
```
torch==2.0.1
torchvision==0.15.2
scikit_learn
scipy
tqdm
numpy==1.26.4
```
## Dataset Preparation 
- Create a folder "datasets/" under the root directory
- MSTAR: download the folders MSTAR from [link](https://itigr-my.sharepoint.com/personal/karantai_iti_gr/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fkarantai%5Fiti%5Fgr%2FDocuments%2Fdatasets&ga=1) and place them into the 'datasets/' folder
- MSTAR_OPENSAR: download the folder MSTAR_OPENSAR from [link](https://itigr-my.sharepoint.com/personal/karantai_iti_gr/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fkarantai%5Fiti%5Fgr%2FDocuments%2Fdatasets&ga=1) and place them into the 'datasets/' folder.
- SAR-AIRcraft-1.0: download the folder AIRCRAFT [link](https://itigr-my.sharepoint.com/personal/karantai_iti_gr/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fkarantai%5Fiti%5Fgr%2FDocuments%2Fdatasets&ga=1) and place them into the 'datasets/' folder.
## Training: 
- To train DILHyFS in the FSCIL scenario, place the pre-trained weights of the GFNet model [link](https://drive.google.com/file/d/1Nrq5sfHD9RklCMl6WkcVrAWI5vSVzwSm/view) into the 'weights/' folder.  
- For MSTAR dataset run:
```
python main.py --config=./exps/DILHyFS.json
```
- For cross-domain experiment run:
```
python main.py --config=./exps/cross_exp.json
```
- For the comparative analysis of the backbone networks run:
```
python main.py --config=./exps/ablation_exps/resnet.json
python main.py --config=./exps/ablation_exps/gfnet.json
```
- For DILHyFS evaluation on limited data scenario run:
```
python main.py --config=./exps/limited.json
```
## Acknowledgments 
We thank the following repos providing helpful components/functions in our work.
- [PILOT](https://github.com/sun-hailong/LAMDA-PILOT)  
- [RanPAC](https://github.com/RanPAC/RanPAC/)
- [GFNet](https://github.com/raoyongming/GFNet)

You should also cite the following:

MSTAR dataset:
```
@inproceedings{ross1998standard,
  title={Standard SAR ATR evaluation experiments using the MSTAR public release data set},
  author={Ross, T. and Worrell, S. and Velten, V. and Mossing, J. and Bryant, M.},
  booktitle={Algorithms for synthetic aperture radar imagery V},
  volume={3370},
  pages={566--573},
  year={1998},
  organization={SPIE}
}
```
OpenSARShip dataset:
```
@article{huang2017opensarship,
  title={OpenSARShip: A dataset dedicated to Sentinel-1 ship interpretation},
  author={Huang, L. and Liu, B. and Li, B. and Guo, W. and Yu, W. and Zhang, Z. and Yu, W.},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  volume={11},
  number={1},
  pages={195--208},
  year={2017},
  publisher={IEEE}
}
```
SarAircraft dataset:
```
@article{zhirui2023sar,
  title={SAR-AIRcraft-1.0: High-resolution SAR aircraft detection and recognition dataset},
  author={Zhirui, W. and Yuzhuo, K. and Xuan, Z. and Yuelei, W. and Ting, Z. and Xian, S.},
  journal={J. Radars},
  volume={12},
  number={4},
  pages={906--922},
  year={2023}}
```

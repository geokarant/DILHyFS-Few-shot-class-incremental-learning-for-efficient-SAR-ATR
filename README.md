## Few-shot class incremental learning for efficient SAR-ATR

Our paper Few-Shot Class-Incremental Learning For Efficient SAR Automatic Target Recognition has been accepted for publication in IEEE ICIP 2025.

If you use any code of this repo, please consider citing our work:

@article{karantaidis2025few,
  title={Few-Shot Class-Incremental Learning For Efficient SAR Automatic Target Recognition},
  author={Karantaidis, George and Pantsios, Athanasios and Kompatsiaris, Ioannis and Papadopoulos, Symeon},
  journal={arXiv preprint arXiv:2505.19565},
  year={2025}
}


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

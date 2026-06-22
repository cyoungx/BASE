# BASE: A boundary-aware adaptive semantic evidential framework for open-set skeleton-based action recognition

## Installation

- Python = 3.8
- PyTorch = 1.10.1
- h5py scikit-learn pyyaml tensorboardX tqdm einops pytorch_metric_learning matplotlib
- We have provided `requirements.txt` containing the full dependencies for checking.

## Datasets
- NTU 60
- NTU 120
- ToyotaSmartHome

#### NTU RGB+D 60 and 120
1. Request dataset here: https://rose1.ntu.edu.sg/dataset/actionRecognition
2. Download the skeleton-only datasets:
   1. `nturgbd_skeletons_s001_to_s017.zip` (NTU RGB+D 60)
   2. `nturgbd_skeletons_s018_to_s032.zip` (NTU RGB+D 120)
   3. Extract above files to `./data/nturgbd_raw`
#### ToyotaSmartHome
1. Request the dataset for 3D skeleton here https://project.inria.fr/toyotasmarthome/
### Data Processing

#### Directory Structure

Put downloaded data into the following directory structure:

```
- data/
  - ntu/
  - ntu120/
  - nturgbd_raw/
    - nturgb+d_skeletons/     # from `nturgbd_skeletons_s001_to_s017.zip`
      ...
    - nturgb+d_skeletons120/  # from `nturgbd_skeletons_s018_to_s032.zip`
      ...
```
#### Generating Data

- Generate NTU RGB+D 60 or NTU RGB+D 120 dataset:

```
 cd ./data/ntu # or cd ./data/ntu120
 # Get skeleton of each performer
 python get_raw_skes_data.py
 # Remove the bad skeleton 
 python get_raw_denoised_data.py
 # Transform the skeleton to the center of the first frame
 python seq_transformation.py
```

- Text embedding generation can see https://github.com/kaai520/PGFA

## Training & Testing

You can run the following command to conduct the experiments, which include both training and testing:
```
bash examples.sh
```
You can also run other experiments by modifying the configurations.

## Citation

Please consider citing this paper if you use ours code.

```bibtex
@article{yang2026base,
        title = {BASE: A boundary-aware adaptive semantic evidential framework for open-set skeleton-based action recognition},
        journal = {Expert Systems with Applications},
        volume = {331},
        pages = {133133},
        year = {2026},
        issn = {0957-4174},
        doi = {https://doi.org/10.1016/j.eswa.2026.133133},
        author = {Chao Yang and Dongyue Chen and Dingyu Xue and Shizhuo Deng and Tong Jia},
        }
```

Our code is based on the repository of [Navigating open set scenarios for skeleton-based action recognition](https://github.com/KPeng9510/OS-SAR). While citing our work, please also cite theirs.

```bibtex
@inproceedings{peng2024navigating,
  title={Navigating open set scenarios for skeleton-based action recognition},
  author={Peng, Kunyu and Yin, Cheng and Zheng, Junwei and Liu, Ruiping and Schneider, David and Zhang, Jiaming and Yang, Kailun and Sarfraz, M Saquib and Stiefelhagen, Rainer and Roitberg, Alina},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={5},
  pages={4487--4496},
  year={2024}
}
```

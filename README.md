**Official HUST-LFI database and MM-BLFQA metric code for the ACM MM 2026 paper "Perceptual Quality Assessment for Light Field Image Compression: A Subjective Database and Objective Metric".  Please refer to the [paper](https://dl.acm.org/doi/10.1145/3767308.3835943) for details.****

![image](https://github.com/oldblackfish/HUST-LFI/blob/main/fig/ACMMM2026_Poster.png)

**Note: First, we convert the dataset into H5 files using MATLAB. Then, we train and test the model in Python.**

# HUST-LFI Database

Please download the full database on [Baidu drive](https://pan.baidu.com/s/1RKPRMZ72cDzO8t3fzmkloA) (code: HUST).

# MM-BLFQA Metric

### Train
Train the model using the following command:
```
python Train.py  --trainset_dir ./Datasets/HUST_LFI_5x32x64/
```

### Test Overall Performance
Test the overall performance using the following command:
```
python Test.py
```

### Test Individual Distortion Type Performance
Test the individual distortion type performance using the following command:
```
 python Test_Dist.py
```

### Result
The performances of our MM-BLFQA metric on the HUST-LFI database can reproduce these performances using the h5 results we provide in './MM-BLFQA/Results/...'.

### Citation
Please cite the following paper if you use this repository in your reseach.
```
@inproceedings{10.1145/3767308.3835943,
author = {Zhou, Rui and Liu, Qiong and Yang, You},
title = {Perceptual Quality Assessment for Light Field Image Compression: A Subjective Database and Objective Metric},
year = {2026},
isbn = {9798400722134},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3767308.3835943},
doi = {10.1145/3767308.3835943},
abstract = {Light field images (LFIs) capture rich angular and spatial information, enabling immersive scene visualization. However, their substantial storage requirements necessitate compression, which inevitably introduces perceptual distortions. Despite the emergence of numerous advanced compression methods, existing quality assessment frameworks for LFIs remain narrow in scope, leaving the perceptual impact of diverse compression distortions largely unexplored. To address this, we conduct subjective experiments and present a new perceptual quality assessment database comprising 1000 distorted LFIs from 20 scenes, generated using 10 compression methods across five quality levels, with mean opinion scores collected from 29 qualified observers under controlled conditions. Building on this database, we propose a novel objective blind metric based on multi-modal and multi-task learning. The metric extracts 2D and 3D multi-modal patches from 4D LFIs, feeding them into image-domain and video-domain branches for preliminary quality assessment. A distortion-aware multi-task learning module then integrates cross-modal features to produce the final quality score. Extensive experiments on the proposed database demonstrate that the proposed metric outperforms representative quality assessment metrics in various aspects. The database and metric code are available at https://github.com/oldblackfish/HUST-LFI.},
booktitle = {Proceedings of the 34th ACM International Conference on Multimedia},
pages = {15958–15967},
numpages = {10},
keywords = {light field image quality assessment, database, blind metric},
location = {Rio de Janeiro, Brazil},
series = {MM '26}
}
```

### Contact
For any questions, feel free to contact: rui_z@hust.edu.cn

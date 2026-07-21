**Official HUST-LFI database and MM-BLFQA metric code for the ACM MM 2026 paper "Perceptual Quality Assessment for Light Field Image Compression: A Subjective Database and Objective Metric".**

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

### Contact
For any questions, feel free to contact: rui_z@hust.edu.cn

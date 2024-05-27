# FPFR-torch

This is the pytorch implemention of our paper "__Learning Fused Pixel and Feature-Based View Reconstructions for Light Fields__"  (__CVPR 2020 Oral__).

By [Jinglei Shi](https://jingleishi.github.io/),  [Xiaoran Jiang](https://scholar.google.com/citations?hl=zh-CN&user=zvdY0EcAAAAJ&view_op=list_works&sortby=pubdate)  and  [Christine Guillemot](https://people.rennes.inria.fr/Christine.Guillemot/)

<[Paper link](https://openaccess.thecvf.com/content_CVPR_2020/papers/Shi_Learning_Fused_Pixel_and_Feature-Based_View_Reconstructions_for_Light_Fields_CVPR_2020_paper.pdf)>

## Dependencies
```
python==3.8
pytorch==1.11.0
torchvision==0.12.0
```
## Examples
### Preparation of the dataset
Before training the network, the preparation of the dataset is as follows:
- Create a folder containing all light fields used for training and validation, and each light field is an individual folder, with all sub-apterture images named 'lf_row_column.png', where 'row' and 'column' are row and column indices.
- Create .txt files and add the names of the light field into them. A 'tri_trainslit.txt' for training set and a 'tri_testlist.txt' for test set.

### Training
Before launching the training, we should adapt the configuration (Lines 12 - 30) of the file 'train.py' as follows:
- *train_list* & *val_list*: Path to the files 'tri_trainlist.txt' and 'tri_testlist.txt'.
- *train_batchsize* & *val_batchsize*: Batch size for training and validation (default values for training is 4, for validation is 1).
- *train_patchsize* & *val_patchsize*: Patch size for training and validation (default values for training is [160,160], for validation is [768,768]).
- *train_ratio* & *val_ratio*: Proportions of light fields involved in the training and validation (default value for both is 1).
- *Dim*: Angular resolution for light fields (default value 9).
- *Min_len*: Minimal distance between corner views, for example, if Min_len=7 and Dim=9, subsets of light field in 7x7, 8x8 and 9x9 patterns will be used.
- *test_frequency*: The number of epoch to pass before testing.
- *save_frequency*: The number of epoch to pass before saving the trained model.
- *version_dir*: Path to the folder which stores the trained models and learning curves.
- *best_dir*: Path to the saved best model.
- *regular_dir*: Path to the saved regular model.

After configuring all settings, users can simply launch the training by:
```
python train.py
```

### Evaluation
To synthesize dense light fields with given corner views, users should adapt the configuration (Lines 12-19) of the file 'test.py' as follows:
- *data_folder*: Path to the folder containing four corner views.
- *result_folder*: Path to the folder containing synthesized sub-aperture views.
- *model_path*: Path to the trained model.
- *lt*, *rt*, *lb*, *rb*: Four corner views.
- *D*: Distance between corner views, for example, to synthesize a 9x9 light field, D should be set 8.

After configuring the above settings, users can launch the test by:
```
python test.py
```

### Pretrained Models
We provide three models respectively pretrained on HCI LF dataset ([synth.pkl](https://pan.baidu.com/s/1ZAIttST3AliL87-0y3RMmQ?pwd=0003)), UCSD LF dataset ([realworld.pkl](https://pan.baidu.com/s/1Y2rfeUa6F-PW7UgTuhWoew?pwd=0004)) and EPFL LF dataset ([epfl.pkl](https://pan.baidu.com/s/1SkwXVK3uoIUvC9wj0Q2onQ?pwd=0002)), users can just download them and put them into the folder 'saved_model'.


## Citation
Please consider citing our work if you find it useful.
```
@inproceedings{shi2020learning,
    title={Learning Fused Pixel and Feature-based View Reconstructions for Light Fields},
    author={Jinglei Shi and Xiaoran Jiang and Christine Guillemot},
    booktitle={IEEE. International Conference on Computer Vision and Pattern Recognition (CVPR)},
    pages={2555--2564},
    year={2020}}
```

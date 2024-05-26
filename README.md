# FPFR2

This is the repository of the paper "__Learning Fused Pixel and Feature-Based View Reconstructions for Light Fields__"  (__CVPR 2020 Oral__).

By [Jinglei Shi](https://jingleishi.github.io/),  [Xiaoran Jiang](https://scholar.google.com/citations?hl=zh-CN&user=zvdY0EcAAAAJ&view_op=list_works&sortby=pubdate)  and  [Christine Guillemot](https://people.rennes.inria.fr/Christine.Guillemot/)

<[Paper link](https://openaccess.thecvf.com/content_CVPR_2020/papers/Shi_Learning_Fused_Pixel_and_Feature-Based_View_Reconstructions_for_Light_Fields_CVPR_2020_paper.pdf)>

## Dependencies
```
python==3.x
pytorch==
```
## Examples
### Preparation of the dataset
Before training the network, the preparation of the dataset is as follows:
- Create a folder containing all light fields used for training and validation, and each light field is an individual folder, with all sub-apterture images named 'lf_row_column.png', where 'row' and 'column' are row and column indices.
- Create .txt files and add the names of the light field into them. A 'tri_trainslit.txt' for training set and a 'tri_testlist.txt' for test set.

### Training
Before launching the training, we should adapt the configuration (Line 12 - Line 30) of the file 'train.py' as follows:
- train_list & val_list: Path to the files 'tri_trainlist.txt' and 'tri_testlist.txt'.
- train_batchsize & val_batchsize: Batch size for training and validation (default values for training is 4, for validation is 1).
- train_patchsize & val_patchsize: Patch size for training and validation (default values for training is [160,160], for validation is [768,768]).
- train_ratio & val_ratio: Proportions of light fields involved in the training and validation (default value for both is 1).
- Dim: Angular resolution for light fields (default value 9).
- Min_len: Minimal distance between corner views, for example, if Min_len=7 and Dim=9, subsets of light field in 7x7, 8x8 and 9x9 patterns will be used.
- 

### Evaluation

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

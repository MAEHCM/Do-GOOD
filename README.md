# 📚Do-GOOD: A Fine-Grained Analysis of Distribution Shifts on Document Images

# Table of contents
* [Overview](#overview)
* [Requirement](#requirement)
* [Installation](#installation)
* [Datasets](#datasets)
* [Run-test](#run-test)

# Overview

The Do-GOOD warehouse is the analysis for document changes in the three modal distributions of image, layout and text. It covers the generation of nine kinds of OOD data, the application of five shift, the acquisition of FUNSD-H and FUNSD-R datasets, the generation of FUNSD-L datasets, and the running of two kinds of OOD baseline methods Deep Core and Mixup codes under all shift.

The shift type of the Do GOOD dataset is shown in the following figure.

![](https://user-images.githubusercontent.com/111342294/202709041-af2c99b2-5a6e-49b5-93ce-2c4883960601.png)



# Requirement

This code is developed with

```javascript
transformers              4.24.0 
pytesseract               0.3.9 
tesseract                 0.1.3     
textattack                0.3.7 
python                    3.9.11
yarl                      1.7.2
detectron2                0.6                         
editdistance              0.6.0                    
einops                    0.4.1
```

# Installation

Installation for Project，if you need to study the robustness of the model to text shift, you need to install [Textattack](https://github.com/QData/TextAttack) it first

```
git clone https://anonymous.4open.science/r/Do-GOOD-D88A && cd Do-GOOD
```

# Datasets

We provide manually labeled FUNSD-H and FUNSD-R, which can be obtained from the links below, and methods for generating FUNSD-L datasets.

| Dataset | Header      | Question      | Answer      | Other      | Total      | Link      |
|:--------:| :------------:| :------------:| :------------:| :------------:|:------------:|:------------:|
| FUNSD | 122 | 1077 | 821 | 312 | 2332 |[download](https://pan.baidu.com/s/18OHBdaJCtFWTovHulJGAiQ)|
| FUNSD-H | 126 | 981 | 755 | 380 | 2304 |[download](https://pan.baidu.com/s/15L3Kyc2-NcpXqb6o7cd-HQ)|
| FUNSD-R | 90 | 475 | 445 | 471 | 1487 |[download](https://pan.baidu.com/s/1yrm0YANgX290ZMhpTBi8Cg)|


### How to generate FUNSD-L

First generate strong and weak semantic entities and get the following files , `/weak_other_map` , `/strong_answer_map` , `/strong_question_map` , `/weak_Q_map`   , `/weak_A_map`,We provide five strong and weak semantic entity libraries extracted from our shuffle layout method on the FUNSD test set for five different pre-training models ,You can choose to fill in `v3`, `v2`, `v1`, `bros` or `lilt` in {model} and execute the following code
```
python map_{model}_funsd_L.py
```

Then modify the file path to generate FUNSD-L test data , which is saved in the `mix_test.txt` , you can modify the number of rows and columns generated by the layout, the size of the bounding box, the probability of random filling, and the number of documents generated

```
generate_ood_data("mix_test.txt", "/strong_question_map",
                  "/strong_answer_map", "/weak_Q_map",
                  "/weak_A_map", "/weak_other_map",50)
```

```
python gen_ood_mix.py
```

![](https://user-images.githubusercontent.com/111342294/202719602-47a09c21-0226-4221-9652-6d714b4a4a46.png)


### Nearest neighbor box merge

To facilitate use, we separately place it in the main directory, and adjust two parameters: lamda1 controls the horizontal distance, and lamda2 controls the vertical distance. We use the priority order of consolidation: horizontal first and then vertical

```
python merge_layout.py
```

![](https://user-images.githubusercontent.com/111342294/202724209-b915d944-dd62-4e77-a66e-781bc4b4a707.png)


### Replace document background with natural scene image

Separate text pixels and non text pixels in the document, and then overlay them into the natural scene [MSCOCO](https://cocodataset.org/#home)

```
python python mixup_image.py
```

![](https://user-images.githubusercontent.com/111342294/202724449-f8ee8ffd-c8aa-4dd7-b665-1a6558b5e7aa.png)

### Generate distorted images

Using pre-trained [DocGeoNet](https://github.com/fh2019ustc/DocGeoNet)(specific process reference), a forward propagation calculation of the normal document image is performed to get the distorted image, and then OCR again

```
python inference.py
```

![](https://user-images.githubusercontent.com/111342294/202726278-e0a89790-494e-46a6-8a2e-42009f2dfce4.png)


# Run-test

Select the model used and the task fill, the first { } select `v3`, `v2`, `v1`, `bros` or `lilt`, the second {} select`funsd` or `cdip`

Finetune your own LayoutLMv3 model or download our finetuned model [download](https://pan.baidu.com/s/1zwlTvQsJfQDVOo2UDMRgRA)，Select models and tasks to use

```javascript
python -m torch.distributed.launch --nproc_per_node --use_env finetune_{}_{}.py --config config.yaml --output_dir
```

For VQA tasks, use the command line alone，fill in the selected model at { }

```javascript
python docvqa_{}_main.py
```

#### 1.Add OOD Samples Into Original Training Set and Test

| Sample Num | 0      | 10      | 20      |
|:--------:| :------------:| :------------:| :------------:|
| +FUNSD_R | 56.56 | 61.53 | 65.10 |
| +FUNSD_H | 68.32 | 70.78 | 70.65 |


#### 2 Testing OOD datasets for downstream tasks

Select the model used and the task fill, the first { } select `v3`, `v2`, `v1`, `bros` or `lilt`, the second { } select`funsd` or `cdip` ， modify the following parameters to perform a shift operation on a mode.`--text_aug`,`--image_aug`,`--aut_layout`
```
--text_aug={'WordSwapMaskedLM','WordSwapEmbedding','WordSwapHomoglyphSwap','WordSwapChangeNumber','WordSwapRandomCharacterDeletion'} , --image_aug=True/False , --aug_layout=True/False
```

```
python demo_{}_ood_{}.py
```


we'll clean the code with detailed documents in the Github.

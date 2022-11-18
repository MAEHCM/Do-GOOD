# 📚Do-GOOD: A Fine-Grained Analysis of Distribution Shifts on Document Images

# Table of contents
- Overview
- Installation
- Requirements
- Datasets
- Quick Tutorial
- Test


# Overview

The Do GOOD warehouse is the benchmark for document changes in the three modal distributions of image, layout and text. It covers the generation of nine kinds of OOD data, the application of five shift, the acquisition of FUNSD-H and FUNSD-R datasets, the generation of FUNSD-L datasets, and the running of two kinds of OOD baseline methods Deep Core and Mixup codes under all shift.

The shift type of the Do GOOD dataset is shown in the following figure.

![](https://user-images.githubusercontent.com/111342294/202708365-6bf714b1-5c10-48b8-985f-c4fea55cd57c.png)


# Install

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
To ensure that you can finetune LayoutLM series
Then install textattach


# RUN

Finetune your own LayoutLMv3 model or download our finetuned model [link:GOOD](https://pan.baidu.com/s/1zwlTvQsJfQDVOo2UDMRgRA)


#### 1. Add a small amount/complete FUNSD-R/FUNSD-H data to the FUNSD training set to fine tune the pre trained layoutlmv3 and test the LayoutLMv3 trained on FUNSD dataset on FUNSD-R/FUNSD-H

```javascript
python demo.py
```

| FUNSD | FUNSD_R      | FUNSD_H      |
|:--------:| :------------:| :------------:|
| 90.29 | 90.95 | 91.21 |

#### 2. Generate FUNSD-L
 
##### 2.1 First generate strong and weak semantic entities and get the following files

```
python map_v3_funsd_qa.py
```
/weak_other_map  /strong_answer_map  /strong_question_map  /weak_Q_map  /weak_A_map

##### 2.2 Then modify the file path to generate FUNSD-L test data, which is saved in the mix_test.txt

```
generate_ood_data("mix_test.txt", "/strong_question_map",
                  "/strong_answer_map", "/weak_Q_map",
                  "/weak_A_map", "/weak_other_map",50)
```

```
python gen_ood_mix.py
```


##### 2.3 Test FUNSD-L
```
python demo_v3_ood_ner.py
```

#### 3. Get more image shifts and layout shifts from the following files for documnet classification

```
python mixup_image.py
python merge_layout.py
```

we'll clean the code with detailed documents in the Github.

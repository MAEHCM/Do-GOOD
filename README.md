# Do-GOOD

This is the implementation for the Do-GOOD




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
 
First generate strong and weak semantic entities and get the following files

```
python map_v3_funsd_qa.py
```
#1. weak_other_map 2. strong_answer_map 3. strong_question_map 4. weak_Q_map 5. weak_A_map

Then modify the file path to generate FUNSD-L test data, which is saved in the mix_ In test.txt

```
generate_ood_data("mix_test.txt", "/strong_question_map",
                  "/strong_answer_map", "/weak_Q_map",
                  "/weak_A_map", "/weak_other_map",50)
```

```
python gen_ood_mix.py
```
![](https://user-images.githubusercontent.com/111342294/201651932-f0c65c1e-3fed-42a0-b916-b575f2ba8df2.png#pic_center=60x60)


Test FUNSD-L
```
python demo_v3_ood_ner.py
```

#### 3. Get more image shifts and layout shifts from the following files for documnet classification

```
python mixup_image.py
python merge_layout.py
```

we'll clean the code with detailed documents in the Github.

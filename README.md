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

Finetune your own LayoutLMv3 model or download our finetuned model



1. You can add a small amount/complete FUNSD-R/FUNSD-H data to the FUNSD training set to fine tune the pre trained layoutlmv3

```javascript
python demo.py
```

2. You can also test the LayoutLMv3 trained on FUNSD dataset on FUNSD-R/FUNSD-H

```javascript
python demo.py
```

we'll clean the code with detailed documents in the Github.

# Result







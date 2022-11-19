'''import os
import shutil

import torch

root='/sample_cdip_test'
features_file=os.listdir(root)

class InputExample(object):

    def __init__(self, root, tokens, bboxes, attention_masks,labels):

        self.root = root
        self.tokens = tokens
        self.bboxes = bboxes
        self.attention_masks = attention_masks
        self.labels=labels

class outputExample(object):

    def __init__(self,tokens, bboxes,labels,root):

        self.tokens = tokens
        self.bboxes = bboxes
        self.labels=labels,
        self.root=root,

image_label_list=''
for feature_file in features_file:
    file_root=os.path.join(root,feature_file)
    feature=torch.load(file_root)[0]
    roots = feature.root
    label=feature.labels
    image_file=roots[0]
    image_label=label[0].strip('\n')

    shutil.copy(image_file, 'sample_cdip_image')
    image_label_list=image_label_list+image_file+'\t'+image_label+'\n'

with open('sample_cdip_label.txt','w',encoding='utf-8') as f:
    f.write(image_label_list)'''

# torch
import os
import torch
from PIL import Image
import cv2
from tqdm import trange, tqdm
from transformers import LayoutLMv3Tokenizer


from Process import Layoutlmv3FeatureExtractor, apply_tesseract

tokenizer = LayoutLMv3Tokenizer.from_pretrained("microsoft/layoutlmv3-base")

def pil_loader(path):
    img = cv2.imread(path, cv2.IMREAD_LOAD_GDAL)
    rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    return img.convert("RGB")


class outputExample(object):

    def __init__(self,tokens, bboxes,labels,root):

        self.tokens = tokens
        self.bboxes = bboxes
        self.labels=labels,
        self.root=root,

def normalize_bbox(bbox, width,length):
    return [
        int(1000 * bbox[0] / width),
        int(1000 * bbox[1] / length),
        int(1000 * bbox[2] / width),
        int(1000 * bbox[3] / length),
    ]
class InputExample(object):

    def __init__(self, root, tokens, bboxes, attention_masks,labels):

        self.root = root
        self.tokens = tokens
        self.bboxes = bboxes
        self.attention_masks = attention_masks
        self.labels=labels

img_files='sample_cdip_image'

image_file='sample_cdip_label.txt'

with open(image_file,'r',encoding='utf-8') as f:
    for idx,line in tqdm(enumerate(f)):
            image_temp_file=line.split('\t')[0]
            labels=line.split('\t')[1].strip('\n')

            image_name=image_temp_file.split('/')[-1][:-4]+'_rec.png'
            image_temp_file=os.path.join('/mnt/disk2/hjb/sample_cdip_distort',image_name)


            img = cv2.imread(image_temp_file, cv2.IMREAD_LOAD_GDAL)
            width,length=img.shape[0],img.shape[1]

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            sample=img.convert("RGB")

            words, boxes = apply_tesseract(sample,None)

            examples=[]
            examples.append(
                outputExample(
                    tokens=words,
                    bboxes=boxes,
                    labels=labels,
                    root=image_temp_file,
                )
            )
            features = torch.save(examples,'sample_cdip_distord_ocr/{}'.format(idx))

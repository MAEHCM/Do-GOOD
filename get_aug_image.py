import os
import numpy as np
import cv2
import torch
from torchvision import transforms
from PIL import Image



IMAGENET_DEFAULT_MEAN = [0.5, 0.5, 0.5]
IMAGENET_DEFAULT_STD = [0.5, 0.5, 0.5]


mean = IMAGENET_DEFAULT_MEAN
std = IMAGENET_DEFAULT_STD

def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    img = cv2.imread(path, cv2.IMREAD_LOAD_GDAL)
    rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    return img.convert("RGB")


normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

patch_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])

image_transform= transforms.Compose([
                transforms.Resize([224,224]),
                transforms.ToTensor(),
                normalize
            ])

def normalize_bbox(bbox, width,length):
    return torch.tensor([
        int(1000 * bbox[0] / width),
        int(1000 * bbox[1] / length),
        int(1000 * bbox[2] / width),
        int(1000 * bbox[3] / length),
    ])

def unnormalize_box(bbox, width, height):
    return [
        int(width * (bbox[0] / 1000)),
        int(height * (bbox[1] / 1000)),
        int(width * (bbox[2] / 1000)),
        int(height * (bbox[3] / 1000)),
    ]


def get_image(roots,bboxes=None,cdip=False,empty=False,mscoco=False,batch=None):

    batch_image1_transform=[]
    batch_image2_transform = []


    for idx,root in enumerate(roots):

        if mscoco==True:

            coco_file_root = 'MSCOCO/train2014'

            coco_file = os.listdir(coco_file_root)


            img_coco=pil_loader(os.path.join(coco_file_root,coco_file[batch]))

            img_cdip = pil_loader(root)

            box=bboxes[idx].detach().numpy()

            img_cdip_w, img_cdip_h = img_cdip.size
            img_coco = img_coco.resize((img_cdip_w, img_cdip_h))

            img_coco = np.array(img_coco)
            #img_w,img_h=img_coco.shape
            #img_coco=np.repeat(img_coco, 3, axis=1).reshape((img_w,img_h,3))
            img_cdip = np.array(img_cdip)

            for i in range(len(box)):
                flag=False
                if box[i][0] >= img_cdip_w or box[i][1] >= img_cdip_h or box[i][2] >= img_cdip_w or box[i][3] >= img_cdip_h:
                    continue
                for w in range(4):
                    if box[i][w]!=0 or box[i][w]!=1000:
                        flag=True

                if flag:
                    box[i] = unnormalize_box(box[i], img_cdip_w, img_cdip_h)

                    for j in range(box[i][0], box[i][2]):
                        for k in range(box[i][1], box[i][3]):
                            if img_cdip[k, j][0] != 255 and img_cdip[k, j][1] != 255 and \
                                    img_cdip[k, j][2] != 255:
                                img_coco[k, j] = img_cdip[k, j]


            img = Image.fromarray(img_coco).convert('RGB')
        else:
            img=Image.open(root).convert('RGB')

        width, length = img.size

        if empty == True:
            img = Image.new('RGB', (width, length), (255, 255, 255))

        image1 = image_transform(img).unsqueeze(0)
        image2 = image_transform(img).unsqueeze(0)

        batch_image1_transform.append(image1)
        batch_image2_transform.append(image2)


    res_image1_transform=torch.concat(batch_image1_transform,dim=0)
    res_image2_transform=torch.concat(batch_image2_transform,dim=0)


    if cdip:
        return res_image1_transform,res_image2_transform,bboxes

    return res_image1_transform,res_image2_transform




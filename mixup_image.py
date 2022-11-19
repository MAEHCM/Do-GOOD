from PIL import Image, ImageFont, ImageDraw
import numpy as np
from utils.Process import apply_tesseract

def unnormalize_box(bbox, width, height):
    return [
        int(width * (bbox[0] / 1000)),
        int(height * (bbox[1] / 1000)),
        int(width * (bbox[2] / 1000)),
        int(height * (bbox[3] / 1000)),
    ]

image_tar=Image.open('').convert('RGB')

image_sample=Image.open('').convert('RGB')


words,boxes=apply_tesseract(image_sample,None)

image_sample_w,image_sample_h=image_sample.size
image_tar=image_tar.resize((image_sample_w,image_sample_h))

src_image_tar=np.array(image_tar)
src_image_sample=np.array(image_sample)

for i in range(len(words)):
    boxes[i]=unnormalize_box(boxes[i], image_sample_w, image_sample_h)
    if boxes[i][0]>=image_sample_w or boxes[i][1]>=image_sample_h or boxes[i][2]>image_sample_w or boxes[i][3]>image_sample_h:
        continue
    for j in range(boxes[i][0],boxes[i][2]):
        for k in range(boxes[i][1],boxes[i][3]):
                if src_image_sample[k,j][0]!=255 and src_image_sample[k,j][1]!=255 and src_image_sample[k,j][2]!=255:
                    src_image_tar[k,j]=src_image_sample[k,j]

img=Image.fromarray(src_image_tar)


draw = ImageDraw.Draw(img)
font = ImageFont.load_default()

img.show()

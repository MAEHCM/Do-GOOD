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

def get_line_bbox(bboxs):

    x = [bboxs[i][j] for i in range(len(bboxs)) for j in range(0, len(bboxs[i]), 2)]
    y = [bboxs[i][j] for i in range(len(bboxs)) for j in range(1, len(bboxs[i]), 2)]

    x0, y0, x1, y1 = min(x), min(y), max(x), max(y)

    assert x1 >= x0 and y1 >= y0
    bbox = [[x0, y0, x1, y1] for _ in range(len(bboxs))]
    return bbox

img=Image.open('C:/Users/25441/Desktop/testing_data/images/82250337_0338.png').convert('RGB')
words,boxes=apply_tesseract(img,None)

for i in range(len(boxes)):
    boxes[i] = unnormalize_box(boxes[i], img.size[0], img.size[1])
####################################
draw = ImageDraw.Draw(img)
font = ImageFont.load_default()
for box in boxes:
    draw.rectangle(box,fill=(0,0,0))
img.show()


temp_box=[[] for _ in range(len(boxes))]
flag_ids=[[] for _ in range(len(boxes))]
flag=np.ones((img.size[0],img.size[1]),dtype=np.uint8)*100000

idx=0
for i in range(len(boxes)):
    x1, y1, x2, y2 = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
    if i!=0:
        if (flag[x1][y1]==100000 and flag[x1][y2]==100000 and flag[x2][y1]==100000 and flag[x2][y2]==100000):#如果点不在区域内
            idx+=1
            temp_box[idx].append(boxes[i])
            flag[x1 - 10:x2 + 10, y1-5:y2+5] = idx
            flag_ids[idx].append(i)
        else:
            a,b,c,d=flag[x1][y1],flag[x1][y2],flag[x2][y1],flag[x2][y2]
            if a!=100000:
                tem_idx=a
            elif b!=100000:
                tem_idx=b
            elif c!=100000:
                tem_idx=c
            else:
                tem_idx=d

            temp_box[tem_idx].append(boxes[i])
            flag[x1 - 10:x2 + 10, y1-5:y2+5] = tem_idx
            flag_ids[tem_idx].append(i)
    else:
        temp_box[idx].append(boxes[i])
        flag[x1 - 10:x2 + 10, y1-5:y2+5] = idx
        flag_ids[idx].append(i)

res_box=[]
res_idx=[]
for i in range(len(temp_box)):
    if len(temp_box[i])==0:
        continue
    else:
        temp=get_line_bbox(temp_box[i])
        for j in range(len(temp)):
            res_idx.append(flag_ids[i][j])
            res_box.append(temp[j])
result_box=[]

for i in range(len(res_box)):
    result_box.append(res_box[res_idx[i]])

draw = ImageDraw.Draw(img)
font = ImageFont.load_default()

for box in result_box:
    draw.rectangle(box,fill=(0,0,0))

img.show()




from transformers import LayoutLMv3Tokenizer


###################################################################
import argparse
import os
from seqeval.metrics import precision_score, recall_score, f1_score
import numpy as np
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader

import ruamel.yaml as yaml

# from create_pretrain_datset import CDIP_dataset
from model.Do_funsd import AET
from transformers import AutoProcessor

from get_aug_image import get_image
from torch.utils.data import Dataset

################################################################
data_dir="FUNSD/data"
img_path="FUNSD/training_data/images"
label_path='FUNSD/label.txt'


def get_labels(path):
    with open(path, 'r') as f:
        labels = f.read().splitlines()
    if 'O' not in labels:
        labels = ["O"] + labels
    return labels


labeles = get_labels(label_path)
label2idx = {label: i for i, label in enumerate(labeles)}
idx2label = {i: label for i, label in enumerate(labeles)}


def read_examples_from_file(data_dir, mode='train', shuffle_boxes=False):
    file_path = os.path.join(data_dir, "{}.txt".format(mode))
    # ../data/FUNSD/data\train.txt
    box_file_path = os.path.join(data_dir, "{}_box.txt".format(mode))
    # ../data/FUNSD/data\train_box.txt
    image_file_path = os.path.join(data_dir, "{}_image.txt".format(mode))
    # ../data/FUNSD/data\train_image.txt
    image_path = os.path.join(data_dir, "{}_image_path.txt".format(mode))

    guid_index = 1

    word = []
    box = []
    label = []
    actual_box = []
    ##########
    words = []
    boxes = []
    images = []
    labels = []
    actual_boxes = []

    images_path = []

    with open(file_path, encoding='utf-8') as f, \
            open(box_file_path, encoding='utf-8') as fb, \
            open(image_file_path, encoding='utf') as fi, \
            open(image_path, encoding='utf8') as fm:

        for line in fm:
            line = line.rstrip()
            img = Image.open(line).convert("RGB")
            images.append(img)
            images_path.append(line)

        for line, bline, iline in zip(f, fb, fi):
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if word:
                    words.append(word)
                    labels.append(label)
                    boxes.append(box)
                    actual_boxes.append(actual_box)
                    # 重置，更新
                    guid_index += 1
                    word = []
                    box = []
                    label = []
                    actual_box = []
            else:
                splits = line.split("\t")  # ['R&D', 'O\n']
                bsplits = bline.split("\t")  # ['R&D', '383 91 493 175\n']
                isplits = iline.split("\t")  # ['R&D', '292 91 376 175', '762 1000', '0000971160.png\n']
                assert len(splits) == 2
                assert len(bsplits) == 2
                assert len(isplits) == 4
                assert splits[0] == bsplits[0]

                word.append(splits[0])

                if len(splits) > 1:
                    t = splits[-1].replace("\n", "")
                    label.append(int(label2idx[t]))

                    bo = bsplits[-1].replace("\n", "")
                    bo = [int(b) for b in bo.split()]
                    box.append(bo)

                    actual = [int(b) for b in isplits[1].split()]
                    actual_box.append(actual)
        if word:
            words.append(word)
            labels.append(label)
            boxes.append(box)
            actual_boxes.extend(actual_box)

    if shuffle_boxes:
        #实体级shuffle boxes
        shuffled_boxes=[]
        for label,box in zip(labels,boxes):
            map_entity = []
            map_box={}

            idx = 0
            for j in range(len(box)):
                if j>0:
                    if box[j]!=box[j-1]:
                        idx+=1
                map_entity.append(idx)
                map_box[idx]=box[j]
            entity_indice=list(range(idx+1))
            random.shuffle(entity_indice)
            box_list = []
            ent_indice=0
            for ids in range(len(map_entity)):
                if ids>0:
                    if map_entity[ids]!=map_entity[ids-1]:
                        ent_indice+=1

                box_list.append(map_box[entity_indice[ent_indice]])
            shuffled_boxes.append(box_list)
        return words, labels, shuffled_boxes, images, actual_boxes, images_path
    else:
        return words, labels, boxes, images, actual_boxes, images_path


def unnormalize_box(bbox, width, height):
    return [
        width * (bbox[0] / 1000),
        height * (bbox[1] / 1000),
        width * (bbox[2] / 1000),
        height * (bbox[3] / 1000),
    ]


processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)


def Process(images, words, boxes, labels, actual_boxes):
    encoded_inputs = processor(images, words, boxes=boxes, word_labels=labels, padding="max_length", truncation=True)
    encoded_inputs_2 = processor(images, words, boxes=actual_boxes, word_labels=labels, padding="max_length",
                                 truncation=True)

    encoded_inputs['input_ids'] = torch.tensor(encoded_inputs['input_ids'])
    encoded_inputs['attention_mask'] = torch.tensor(encoded_inputs['attention_mask'])
    encoded_inputs['bbox'] = torch.tensor(encoded_inputs['bbox'])

    encoded_inputs['labels'] = torch.tensor(encoded_inputs['labels'])
    encoded_inputs['actual_box'] = torch.tensor(encoded_inputs_2['bbox'])

    return encoded_inputs


class V2Dataset(Dataset):
    def __init__(self, encoded_inputs, test_images_path):
        self.all_images = test_images_path
        self.all_input_ids = encoded_inputs['input_ids']
        self.all_attention_masks = encoded_inputs['attention_mask']
        self.all_bboxes = encoded_inputs['bbox']
        self.all_labels = encoded_inputs['labels']
        self.all_actual_boxes = encoded_inputs['actual_box']

    def __len__(self):
        return len(self.all_labels)

    def __getitem__(self, index):
        return (
            self.all_images[index],
            self.all_input_ids[index],
            self.all_attention_masks[index],
            self.all_bboxes[index],
            self.all_labels[index],
            self.all_actual_boxes[index]
        )


def main(model):

    #### Dataset ####
    print("Creating dataset")

    test_words, test_labels, test_boxes, test_images, test_actual_boxes, test_images_path = read_examples_from_file(
        data_dir, mode='test', shuffle_boxes=True,)
    test_dataset = V2Dataset(Process(test_images, test_words, test_boxes, test_labels, test_actual_boxes),
                             test_images_path)

    datasets_test = test_dataset

    data_test_loader = DataLoader(datasets_test, batch_size=1, num_workers=0)

    preds = None
    out_label_ids = None
    ocr_actual = None
    ocr = None
    image_paths = None
    model.eval()

    for j, (image_path,
            tokens,
            attention_masks,
            bboxes,
            labels, actual_box) in enumerate(data_test_loader):

        with torch.no_grad():

            images, images_aug = get_image(image_path)

            image = images.to(device, non_blocking=True)
            image_aug = images_aug.to(device, non_blocking=True)

            # torch.Size([6, 3, 224, 224])

            tokens = tokens.to(device, non_blocking=True)
            bboxes = bboxes.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            attention_masks = attention_masks.to(device, non_blocking=True)

            loss_cls, logits = model(text=tokens, bbox=bboxes, attention_mask=attention_masks,
                                     image=image, image_aug=image_aug, labels=labels)

        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = labels.detach().cpu().numpy()
            ocr_boxes = bboxes.detach().cpu().numpy()
            ocr_actual_boxes = actual_box.detach().cpu().numpy()
            out_tokens = tokens.detach().cpu().numpy()
            image_paths = image_path
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)
            ocr_boxes = np.append(ocr_boxes, bboxes.detach().cpu().numpy(), axis=0)
            ocr_actual_boxes = np.append(ocr_actual_boxes, actual_box.detach().cpu().numpy(), axis=0)
            out_tokens = np.append(out_tokens, tokens.detach().cpu().numpy(), axis=0)
            image_paths = np.append(image_paths, image_path, axis=0)

    preds = np.argmax(preds, axis=2)

    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]
    ocr_list = [[] for _ in range(out_label_ids.shape[0])]
    actual_ocr_list = [[] for _ in range(out_label_ids.shape[0])]
    token_list = [[] for _ in range(out_label_ids.shape[0])]

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != -100:
                out_label_list[i].append(idx2label[out_label_ids[i][j]])
                preds_list[i].append(idx2label[preds[i][j]])
                ocr_list[i].append(ocr_boxes[i][j])
                actual_ocr_list[i].append(ocr_actual_boxes[i][j])
                token_list[i].append(out_tokens[i][j])

    results = {
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list),
    }
    print(results)

    return preds_list, out_label_list, ocr_list, actual_ocr_list, image_paths, token_list


def iob_to_label(label):
    if label != "O":
        return label[2:]
    else:
        return "OTHER"


def string_box(box):
    return (
            str(box[0])
            + " "
            + str(box[1])
            + " "
            + str(box[2])
            + " "
            + str(box[3])
    )

tokenize_v3=LayoutLMv3Tokenizer.from_pretrained("microsoft/layoutlmv3-base")
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./config.yaml')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--output_dir', default='Pretrain/')
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)

    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    print(config)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    label2color = {
        'Strong': (255, 0, 0),
        'Weak': (0, 255, 0),
    }



    device = torch.device(args.device)

    #### Model ####
    print("Creating model")

    model = AET(config=config, text_encoder=args.text_encoder, init_deit=True)

    model = model.to(device)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['model']

        model.load_state_dict(state_dict)

        print('load checkpoint from %s' % args.checkpoint)


    # 预测N次，记录每个实体相邻预测不同的次数
    curr_pred, labels, _, boxes, img_paths, tokens = main(model)

    weak_flag = [[0 for _ in range(len(curr_pred[i]))] for i in range(len(curr_pred))]

    label2entity=[[[0 for _ in range(4)] for _ in range(len(curr_pred[i]))] for i in range(len(curr_pred))]

    map_label_id={'QUESTION':0,'ANSWER':1,'OTHER':2,'HEADER':3}

    for i in range(50):
        if i==0:
            prev_pred = labels
        else:
            prev_pred = curr_pred
        curr_pred, _, _, _, _, _ = main(model)
        for j, _ in enumerate(img_paths):
            start, end = 0, 0
            for k, (p1, p2) in enumerate(zip(prev_pred[j], curr_pred[j])):
                if p1 != p2:
                    weak_flag[j][k] += 1

                label2entity[j][k][map_label_id[iob_to_label(p2)]] += 1

            for k, p in enumerate(curr_pred[j]):

                if k > 0:
                    total_delta = 0
                    if (boxes[j][k].tolist() != boxes[j][k - 1].tolist()) or (k == len(curr_pred[j]) - 1):
                        end = k
                        for w in range(start, end):
                            total_delta += weak_flag[j][w]

                        if total_delta == 0:
                            for w in range(start, end):
                                weak_flag[j][w] = 0
                        else:
                            for w in range(start, end):
                                weak_flag[j][w] = -100

                        start = k


    for j in range(len(img_paths)):
            start, end = 0, 0
            for k,  p in enumerate(curr_pred[j]):

                if k > 0:
                    if (boxes[j][k].tolist() != boxes[j][k - 1].tolist()) or (k==len(curr_pred[j])-1):
                        end = k

                        total_1 = 0
                        total_2 = 0
                        total_3 = 0
                        total_4 = 0
                        for w in range(start, end):
                            total_1 += label2entity[j][w][0]
                            total_2 += label2entity[j][w][1]
                            total_3 += label2entity[j][w][2]
                            total_4 += label2entity[j][w][3]

                        sum=(total_1+total_2+total_3+total_4)
                        p_1,p_2,p_3,p_4=total_1/sum,total_2/sum,total_3/sum,total_4/sum
                        temp_list=sorted([p_1,p_2,p_3,p_4])
                        print(temp_list)

                        if abs(temp_list[3]-temp_list[2])<0.2 :
                            for w in range(start, end):
                                weak_flag[j][w]=-1

                        start = k

    new_attack_map= []
    strong_num,weak_num,medium_num=0,0,0

    res_attack={}


    for i, _ in enumerate(weak_flag):

        for j, _ in enumerate(weak_flag[i]):

            if weak_flag[i][j]==-1:
                label = "Weak"
                weak_num+=1

            elif weak_flag[i][j]==0:
                label = "Strong"
                strong_num+=1
            else:
                label="medium"
                medium_num+=1

            res_attack[string_box(boxes[i][j])]=label

    print('Strong num:',strong_num)
    print('Weak num:',weak_num)
    print('Medium num:',medium_num)

    with open('text_attack.json', 'w', encoding='utf-8') as w:
            json.dump(res_attack,w)

#Strong num: 955
#Weak num: 1321








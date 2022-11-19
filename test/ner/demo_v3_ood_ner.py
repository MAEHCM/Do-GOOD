from PIL import ImageDraw, ImageFont

import argparse
import os
from seqeval.metrics import precision_score, recall_score, f1_score
import numpy as np
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader

import ruamel.yaml as yaml

# from create_pretrain_datset import CDIP_dataset
from model.funsd_model import testmodel

################################################################
img_path="/FUNSD/training_data/images"

label_path='/FUNSD/label.txt'

new_test_data='/mix_test.txt'


def get_labels(path):
    with open(path, 'r') as f:
        labels = f.read().splitlines()
    if 'O' not in labels:
        labels = ["O"] + labels
    return labels


labeles = get_labels(label_path)
label2idx = {label: i for i, label in enumerate(labeles)}
idx2label = {i: label for i, label in enumerate(labeles)}

def normalize_bbox(bbox, width,length):
    return [
        int(1000 * bbox[0] / width),
        int(1000 * bbox[1] / length),
        int(1000 * bbox[2] / width),
        int(1000 * bbox[3] / length),
    ]


def read_examples_from_file(data_dir):
    guid_index = 0
    total_img = 0

    word = []
    box = []
    label = []
    ##########
    words = []
    boxes = []
    labels = []

    with open(data_dir, encoding='utf-8') as f:
        for line in f:
            if line == "" or line == "\n" or guid_index==510:
                if word:
                    words.append(word)
                    labels.append(label)
                    boxes.append(box)

                    word = []
                    box = []
                    label = []
                    guid_index=0
                    total_img+=1

            else:
                guid_index+=1
                splits = line.split("\t")  # ['R&D', 'O\n']

                assert len(splits) == 3

                word.append(int(splits[0]))

                if len(splits) > 1:
                    t = splits[1].replace("\n", "")
                    label.append(int(label2idx[t]))

                    bo = splits[-1].replace("\n", "")
                    bo = [int(b) for b in bo.split()]
                    bo = normalize_bbox(bo,800,1000)

                    box.append(bo)

        if word:
            words.append(word)
            labels.append(label)
            boxes.append(box)

        return words, labels, boxes,total_img

class InputExample(object):
    def __init__(self, tokens, bboxes, attention_masks,labels):
        self.tokens = tokens
        self.bboxes = bboxes
        self.attention_masks = attention_masks
        self.labels=labels


examples=[]

def Process(words, boxes, labels):
    for idx in range(len(words)):
        word=words[idx]
        box=boxes[idx]
        label=labels[idx]

        word=[0]+word
        word=word+[2]

        box.insert(0,[0,0,0,0])
        box.append([0,0,0,0])

        label=[-100]+label
        label=label+[-100]

        attention_mask = [1] * len(label)

        # padding on right
        padding_length = 512 - len(word)

        word += [1] * padding_length

        label += [-100] * padding_length

        for i in range(padding_length):
            box.append([0, 0, 0, 0])

        attention_mask+=[0]*padding_length

        assert len(word) == 512
        assert len(box) == 512
        assert len(attention_mask) == 512
        assert len(label)==512

        token = torch.tensor(word, dtype=torch.long)
        box = torch.tensor(box, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        label = torch.tensor(label, dtype=torch.long)

        examples.append(
            InputExample(
                tokens=token,
                bboxes=box,
                attention_masks=attention_mask,
                labels=label
            )
        )

    return examples


from torch.utils.data import Dataset


class V2Dataset(Dataset):
    def __init__(self, examples):
        self.all_example=examples
    def __len__(self):
        return len(self.all_example)

    def __getitem__(self, index):
        example=self.all_example[index]
        tokens=example.tokens
        bboxes=example.bboxes
        attention_masks=example.attention_masks
        labels=example.labels
        return (
            tokens,
            bboxes,
            attention_masks,
            labels,
        )

test_words, test_labels, test_boxes,total_img= read_examples_from_file(new_test_data)
test_dataset=V2Dataset(Process(test_words,test_boxes,test_labels))

from torchvision import transforms

image_transform= transforms.Compose([
                transforms.Resize([224,224]),
                transforms.ToTensor(),
            ])

def unnormalize_box(bbox, width, height):
    return [
        width * (bbox[0] / 1000),
        height * (bbox[1] / 1000),
        width * (bbox[2] / 1000),
        height * (bbox[3] / 1000),
    ]

def iob_to_label(label):
    if label !="O":
        return label[2:]
    else:
        return "OTHER"

def main(args, config):
    device = torch.device(args.device)

    #### Dataset ####
    print("Creating dataset")

    datasets_test = test_dataset

    data_test_loader = DataLoader(datasets_test, batch_size=4, num_workers=0)

    #### Model ####
    print("Creating model")

    model = testmodel()

    model = model.to(device)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['model']

        model.load_state_dict(state_dict)

        print('load checkpoint from %s' % args.checkpoint)

    preds = None
    out_label_ids = None
    ocr_actual = None
    ocr = None
    model.eval()

    for j, (
            tokens,
            bboxes,
            attention_masks,
            labels) in enumerate(data_test_loader):

        with torch.no_grad():

            img = Image.new('RGB', (800, 1000), (255, 255, 255))

            img = image_transform(img).unsqueeze(0)
            img_list=[]
            for i in range(tokens.shape[0]):
                img_list.append(img)
            img=torch.concat(img_list,dim=0)


            image = img.to(device, non_blocking=True)

            # torch.Size([6, 3, 224, 224])

            tokens = tokens.to(device, non_blocking=True)
            bboxes = bboxes.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            attention_masks = attention_masks.to(device, non_blocking=True)

            loss_cls, logits = model(text=tokens, bbox=bboxes, attention_mask=attention_masks,
                                     image=image, image_aug=None, labels=labels)

        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = labels.detach().cpu().numpy()
            ocr_boxes = bboxes.detach().cpu().numpy()
            out_tokens = tokens.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)
            ocr_boxes = np.append(ocr_boxes, bboxes.detach().cpu().numpy(), axis=0)
            out_tokens = np.append(out_tokens, tokens.detach().cpu().numpy(), axis=0)

    preds = np.argmax(preds, axis=2)

    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]
    ocr_list = [[] for _ in range(out_label_ids.shape[0])]
    token_list = [[] for _ in range(out_label_ids.shape[0])]

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != -100:
                out_label_list[i].append(idx2label[out_label_ids[i][j]])
                preds_list[i].append(idx2label[preds[i][j]])
                ocr_list[i].append(ocr_boxes[i][j])
                token_list[i].append(out_tokens[i][j])

    results = {
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list),
    }
    print('all results:',results)

    #在中间区域部分的 recall:在实际为'O'的样本中被预测为'O'样本的概率

    gt = 0
    pred_true = 0
    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != -100 :
                if j>0:
                    if ocr_boxes[i][j].tolist()==ocr_boxes[i][j-1].tolist():
                        continue

                if ocr_boxes[i][j][1]>150 :
                    if idx2label[out_label_ids[i][j]] == 'O':
                        gt += 1
                        if idx2label[preds[i][j]] == 'O':
                            pred_true += 1

    recall = pred_true / gt

    #other区域出错率
    error_other=1-recall

    results_other= {
        "other_region_error": error_other
    }

    print('other_region:', results_other)

    # recall:在实际为'question' 'answer' 'other' 的样本中被预测为'header'样本的概率
    gt = 0
    pred_true = 0
    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != -100:
                if j > 0:
                    if ocr_boxes[i][j].tolist() == ocr_boxes[i][j - 1].tolist():
                        continue
                if ocr_boxes[i][j][1] <= 150:
                    if idx2label[out_label_ids[i][j]][2:] == 'ANSWER' or idx2label[out_label_ids[i][j]][2:] == 'QUESTION' or idx2label[out_label_ids[i][j]][2:] == 'OTHER':
                        gt += 1
                        if idx2label[preds[i][j]][2:] == 'HEADER':
                            pred_true += 1

    header_recall = pred_true / gt

    results_other = {
        "header_region_error": header_recall
    }

    print('other_region:', results_other)
    ################################################################对于中间区域

    out_qa_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_qa_list = [[] for _ in range(out_label_ids.shape[0])]

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != -100:
                if ocr_boxes[i][j][1] > 150:
                    if iob_to_label(idx2label[out_label_ids[i][j]])=='QUESTION' or iob_to_label(idx2label[out_label_ids[i][j]])=='ANSWER':
                        out_qa_list[i].append(idx2label[out_label_ids[i][j]])
                        preds_qa_list[i].append(idx2label[preds[i][j]])

    results3 = {
        "error": 1 - recall_score(out_qa_list, preds_qa_list),
    }

    print('question and answer region:', results3)

    return preds_list, out_label_list, ocr_list, token_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./config.yaml')
    parser.add_argument('--checkpoint', default='/funsd_v3_align_69.pth')
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--output_dir', default='train/')
    parser.add_argument('--device', default="cuda")
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    print(config)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    preds_list, out_label_list, ocr_list, token_list=main(args, config)

    label2color = {
                   'OTHER': (0, 225, 255),
                   'HEADER': (225, 0, 255),
                   'QUESTION': (50, 50, 150),
                   'ANSWER': (225, 225, 0),
                   }

    for idx in range(total_img):
        image = Image.new('RGB', (800, 1000), (255, 255, 255))

        width,length=image.size

        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        for prediction, box in zip(out_label_list[idx], ocr_list[idx]):
            # prediction_label = iob_to_label(label_map[prediction]).lower()
            box = unnormalize_box(box, width, length)
            prediction_label = iob_to_label(prediction)

            # 画
            draw.rectangle(box, outline=label2color[prediction_label])
            draw.text((box[0] + 10, box[1] - 10), text=prediction_label, fill=label2color[prediction_label], font=font)

        image.save('/FUNSD_output/gt{}.png'.format(idx))
        print('finish gt{}'.format(idx))
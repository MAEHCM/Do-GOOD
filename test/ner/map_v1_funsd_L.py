import argparse
import os
from seqeval.metrics import precision_score, recall_score, f1_score
import numpy as np
import random
from pathlib import Path
import torch
from PIL import Image
import ruamel.yaml as yaml
from model.funsd_model import testmodel
from transformers import BertTokenizer, AutoConfig, LayoutLMv3Processor, AutoProcessor, LayoutLMv2Processor

################################################################
data_dir="/FUNSD/new_data"
img_path="/FUNSD/new_test_data/images"
label_path='/FUNSD/label.txt'


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

tokenizer = BertTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")

def Process(words, boxes, labels):
    input_ides=[]
    attention_masks=[]
    bboxes=[]
    input_labels=[]
    segment_ids=[]

    encoded_inputs={}
    for word,box,label in zip(words,boxes,labels):

        token_words = []
        token_boxes = []
        token_labels = []

        for str_word, str_box ,str_label in zip(word, box,label):
            split_token = tokenizer.tokenize(str_word)

            token_words.extend(split_token)
            token_boxes.extend([str_box] * len(split_token))
            token_labels.extend([str_label]*len(split_token))

        if len(token_words) > 510:
            token_words = token_words[: 510]
            token_boxes = token_boxes[: 510]
            token_labels = token_labels[: 510]

        token_words += [tokenizer.sep_token_id]
        token_boxes += [[1000,1000,1000,1000]]
        token_labels += [-100]


        token_words = [tokenizer.cls_token_id] + token_words
        token_boxes = [[0,0,0,0]] + token_boxes
        token_labels = [-100] + token_labels

        input_ids = tokenizer.convert_tokens_to_ids(token_words)
        input_mask = [1] * len(input_ids)
        seg_ids = [0] * len(input_ids)
        padding_length = 512 - len(input_ids)


        input_ids += [tokenizer.pad_token_id] * padding_length
        input_mask += [0] * padding_length
        token_labels += [-100] * padding_length
        token_boxes += [[0,0,0,0]] * padding_length
        seg_ids+=[0]* padding_length

        assert len(input_ids)==512
        assert len(input_mask)==512
        assert len(token_labels)==512
        assert len(token_boxes)==512
        assert len(seg_ids)

        input_ides.append(input_ids)
        attention_masks.append(input_mask)
        bboxes.append(token_boxes)
        input_labels.append(token_labels)
        segment_ids.append(seg_ids)

    encoded_inputs["input_ids"]=torch.tensor(input_ides)
    encoded_inputs["attention_mask"]=torch.tensor(attention_masks)
    encoded_inputs["bbox"]=torch.tensor(bboxes)
    encoded_inputs["labels"] = torch.tensor(input_labels)
    encoded_inputs["segment_ids"]=torch.tensor(segment_ids)

    return encoded_inputs

from torch.utils.data import Dataset, DataLoader

class V2Dataset(Dataset):
    def __init__(self, encoded_inputs):
        #self.all_images = encoded_inputs['image']
        self.all_input_ids = encoded_inputs['input_ids']
        self.all_attention_masks = encoded_inputs['attention_mask']
        self.all_bboxes = encoded_inputs['bbox']
        self.all_labels = encoded_inputs['labels']
        self.all_seg=encoded_inputs["segment_ids"]

    def __len__(self):
        return len(self.all_labels)

    def __getitem__(self, index):
        return (
            #self.all_images[index],
            self.all_input_ids[index],
            self.all_attention_masks[index],
            self.all_bboxes[index],
            self.all_labels[index],
            self.all_seg[index]
        )


def main(model):

    #### Dataset ####
    print("Creating dataset")

    test_words, test_labels, test_boxes, test_images, test_actual_boxes, test_images_path = read_examples_from_file(
        data_dir, mode='test', shuffle_boxes=True,)

    test_dataset = V2Dataset(Process(test_words, test_boxes, test_labels))

    datasets_test = test_dataset

    data_test_loader = DataLoader(datasets_test, batch_size=1, num_workers=0)

    preds = None
    out_label_ids = None
    model.eval()

    for j, (
            tokens,
            attention_masks,
            bboxes,
            labels, seg_ids) in enumerate(data_test_loader, ):

        with torch.no_grad():

            tokens = tokens.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            bboxes = bboxes.to(device, non_blocking=True)
            attention_masks = attention_masks.to(device, non_blocking=True)
            seg_ids = seg_ids.to(device, non_blocking=True)

            loss_cls, logits = model(text=tokens, bbox=bboxes, attention_mask=attention_masks,
                                     image=None,
                                     image_aug=None, labels=labels, seg_ids=seg_ids)

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
    print(results)

    '''temp_labels = [[] for i in range(len(out_label_list))]
    temp_boxes = [[] for i in range(len(out_label_list))]
    temp_tokens = [[] for i in range(len(out_label_list))]
    temp_preds = [[] for i in range(len(out_label_list))]

    for i, _ in enumerate(token_list):
        for j, _ in enumerate(token_list[i]):
            temp_str = tokenizer.convert_ids_to_tokens(int(token_list[i][j]))
            if temp_str[0] != 'Ġ':
                continue
            else:
                temp_labels[i].append(out_label_list[i][j])
                temp_boxes[i].append(ocr_list[i][j])
                temp_tokens[i].append(token_list[i][j])
                temp_preds[i].append(preds_list[i][j])

    out_label_list = temp_labels
    ocr_list = temp_boxes
    token_list = temp_tokens
    preds_list = temp_preds'''

    return preds_list, out_label_list, ocr_list, token_list,test_images,test_words


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./config.yaml')
    parser.add_argument('--checkpoint', default='/funsd_v1_69.pth')
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--output_dir', default='train/')
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

    model = testmodel(config=config, text_encoder=args.text_encoder, init_deit=True)

    model = model.to(device)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['model']

        model.load_state_dict(state_dict)

        print('load checkpoint from %s' % args.checkpoint)

    curr_pred, labels,boxes, tokens,images,words = main(model)


    weak_flag = [[0 for _ in range(len(curr_pred[i]))] for i in range(len(curr_pred))]

    label2entity=[[[0 for _ in range(4)] for _ in range(len(curr_pred[i]))] for i in range(len(curr_pred))]

    map_label_id={'QUESTION':0,'ANSWER':1,'OTHER':2,'HEADER':3}

    for i in range(30):
        if i==0:
            prev_pred = labels
        else:
            prev_pred = curr_pred
        curr_pred, _, _,_ ,images,_= main(model)
        for j in range(len(curr_pred)):
            start, end = 0, 0
            for k, (p1, p2) in enumerate(zip(prev_pred[j], curr_pred[j])):
                if p1 != p2:
                    weak_flag[j][k] += 1

                label2entity[j][k][map_label_id[iob_to_label(p2)]] += 1

            for k, p in enumerate(curr_pred[j]):

                if k > 0:
                    total_delta=0
                    if (boxes[j][k].tolist() != boxes[j][k - 1].tolist()) or (k == len(curr_pred[j]) - 1):
                        end = k
                        for w in range(start, end):
                            total_delta+=weak_flag[j][w]

                        if total_delta==0:
                            for w in range(start, end):
                                weak_flag[j][w]=0
                        else:
                            for w in range(start, end):
                                weak_flag[j][w]=-100

                        start = k

    for j in range(len(curr_pred)):
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
                        #if p_3-p_2<0.3 and p_3-p_2>0 and p_2-max(p_4,p_1)>0:#answer稍强文本或者other稍强且answer排第二文本
                        if  p_3-p_2<0.2 and p_3-p_2>0 and p_2-max(p_4,p_1)>0:#answer稍强文本或者other稍强且answer排第二文本
                            for w in range(start, end):
                                weak_flag[j][w]=-1

                        elif p_3-p_1<0.2 and p_3-p_1>0 and p_1-max(p_4,p_2)>0:#question稍强文本或者other稍强且question排第二文本
                            for w in range(start, end):
                                weak_flag[j][w]=-2

                        elif (p_3 - p_4 < 0.3 and p_3 - p_4 > 0 and p_4 - max(p_1, p_2) > 0 and labels[j][k-1]=='O') \
                                or (p_1 - p_4 < 0.3 and p_1 - p_4 > 0 and p_4 - max(p_2, p_3) > 0 and labels[j][k-1][2:]=='QUESTION')\
                                or (p_2 - p_4 < 0.3 and p_2 - p_4 > 0 and p_4 - max(p_1, p_3) > 0 and labels[j][k-1][2:]=='ANSWER'):  # quesation稍强文本且header排第二文本
                            for w in range(start, end):
                                weak_flag[j][w] = -3

                        start = k


    strong_num, weak_A_num, weak_Q_num ,weak_he_num= 0, 0, 0,0

    strong_map, weak_A_map, weak_Q_map ,weak_he_map= [], [], [],[]

    for i, _ in enumerate(weak_flag):
        w, h = images[i].size

        for j, _ in enumerate(weak_flag[i]):

            box = unnormalize_box(boxes[i][j], w, h)
            if weak_flag[i][j] == -1:
                weak_A_num += 1
                label = "Weak_A"
                weak_A_map.append(words[i][j] + '\t' + labels[i][j] + '\t' + string_box(box) + '\t' + label + '\n')

            elif weak_flag[i][j] == -2:
                weak_Q_num += 1
                label = "Weak_Q"
                weak_Q_map.append(words[i][j] + '\t' + labels[i][j] + '\t' + string_box(box) + '\t' + label + '\n')

            elif weak_flag[i][j] == -3:
                weak_he_num += 1
                label = "Weak_O"
                weak_he_map.append(words[i][j] + '\t' + labels[i][j] + '\t' + string_box(box) + '\t' + label + '\n')

            elif weak_flag[i][j] == 0:
                strong_num += 1
                label = "Strong"
                strong_map.append(words[i][j] + '\t' + labels[i][j])

    print(f"strong num: {strong_num}")
    print(f"weak_Q_num: {weak_Q_num}")
    print(f"weak_A_num: {weak_A_num}")
    print(f"weak_Header_num: {weak_he_num}")


    with open('/v1_ood/weak_A_map', 'w', encoding='utf-8') as w:
        idx = 0
        for tex, weak_data in enumerate(weak_A_map):

            token = weak_data.split('\t')[0]
            label = weak_data.split('\t')[1]
            act_box = weak_data.split('\t')[2]
            weak_qa_label=weak_data.split('\t')[-1]

            if label == 'O':

                if tex == 0:
                    w.write(token + '\t' + label + '\t' + str(idx) + '\n')
                else:
                    if weak_A_map[tex].split('\t')[2] == weak_A_map[tex - 1].split('\t')[2]:
                        w.write(token + '\t' + label + '\t' + str(idx) + '\n')
                    else:
                        w.write(token + '\t' + label + '\t' + str(idx + 1) + '\n')
                        idx += 1

    with open('/v1_ood/weak_Q_map', 'w', encoding='utf-8') as w:
        for tex, weak_data in enumerate(weak_Q_map):

            token = weak_data.split('\t')[0]
            label = weak_data.split('\t')[1]
            act_box = weak_data.split('\t')[2]
            weak_qa_label=weak_data.split('\t')[-1]

            if label == 'O':

                if tex == 0:
                    w.write(token + '\t' + label + '\t' + str(idx) + '\n')
                else:
                    if weak_Q_map[tex].split('\t')[2] == weak_Q_map[tex - 1].split('\t')[2]:
                        w.write(token + '\t' + label + '\t' + str(idx) + '\n')
                    else:
                        w.write(token + '\t' + label + '\t' + str(idx + 1) + '\n')
                        idx += 1


    with open('/v1_ood/strong_question_map', 'w', encoding='utf-8') as w:
        idx = -1
        for strong_data in strong_map:
            label = strong_data.split('\t')[-1]
            if label[2:] == 'QUESTION':
                if label[0] == 'B':
                    idx += 1
                w.write(strong_data + '\t' + str(idx) + '\n')

    with open('/v1_ood/strong_answer_map', 'w', encoding='utf-8') as w:
        idx = -1
        for strong_data in strong_map:
            label = strong_data.split('\t')[-1]
            if label[2:] == 'ANSWER':
                if label[0] == 'B':
                    idx += 1
                w.write(strong_data + '\t' + str(idx) + '\n')
    with open('/v1_ood/weak_other_map', 'w', encoding='utf-8') as w:
        idx = 0
        for tex, weak_data in enumerate(weak_he_map):

            token = weak_data.split('\t')[0]
            label = weak_data.split('\t')[1]
            act_box = weak_data.split('\t')[2]
            weak_qa_label=weak_data.split('\t')[-1]

            if label == 'O' or label[2:]=='QUESTION' or label[2:]=='ANSWER':

                if tex == 0:
                    w.write(token + '\t' + label + '\t' + str(idx) + '\n')
                else:
                    if weak_he_map[tex].split('\t')[2] == weak_he_map[tex - 1].split('\t')[2]:
                        w.write(token + '\t' + label + '\t' + str(idx) + '\n')
                    else:
                        w.write(token + '\t' + label + '\t' + str(idx + 1) + '\n')
                        idx += 1
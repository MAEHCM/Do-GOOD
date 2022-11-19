from PIL import Image, ImageDraw, ImageFont

import argparse
import os
from seqeval.metrics import precision_score, recall_score, f1_score
import numpy as np
import random
import json
from pathlib import Path
import torch
from PIL import Image
import ruamel.yaml as yaml
from model.funsd_model import testmodel
from transformers import BertTokenizer, AutoConfig, LayoutLMv3Processor, AutoProcessor, LayoutLMv3Tokenizer, \
    LayoutLMv2Processor, LayoutLMv2Tokenizer, LayoutLMTokenizer

import nltk
from textattack.transformations import CompositeTransformation
from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
from textattack.constraints.overlap import MaxWordsPerturbed
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.constraints.overlap.levenshtein_edit_distance import LevenshteinEditDistance


from textattack.transformations.word_swaps import WordSwapMaskedLM,\
    WordSwapEmbedding,\
    WordSwapChangeNumber,\
    WordSwapHomoglyphSwap,\
    WordSwapRandomCharacterDeletion

from textattack.augmentation import Augmenter

################################################################
#data_dir = "FUNSD_2/data"
data_dir='FUNSD/data'
label_path ='FUNSD/label.txt'


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

tokenizer_v1=LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")

def read_examples_from_file(args,data_dir, mode='train', shuffle_token=False, shuffle_entity=False):
    if args.text_aug:
        file_path = os.path.join(data_dir, args.text_aug + "_" + "{}.txt".format(mode))
        # ../data/FUNSD/data\train.txt
        box_file_path = os.path.join(data_dir, args.text_aug + "_" + "{}_box.txt".format(mode))
        # ../data/FUNSD/data\train_box.txt
        image_file_path = os.path.join(data_dir, args.text_aug + "_" + "{}_image.txt".format(mode))
        # ../data/FUNSD/data\train_image.txt
        image_path = os.path.join(data_dir, "{}_image_path.txt".format(mode))
        map_weak_strong = os.path.join(data_dir, 'text_attack.json')
    else:
        file_path = os.path.join(data_dir, "{}.txt".format(mode))
        # ../data/FUNSD/data\train.txt
        box_file_path = os.path.join(data_dir, "{}_box.txt".format(mode))
        # ../data/FUNSD/data\train_box.txt
        image_file_path = os.path.join(data_dir, "{}_image.txt".format(mode))
        # ../data/FUNSD/data\train_image.txt
        image_path = os.path.join(data_dir, "{}_image_path.txt".format(mode))
        map_weak_strong = os.path.join(data_dir, 'text_attack.json')

    guid_index = 0

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
    entity_set=[]

    with open(file_path, encoding='utf-8') as f, \
            open(box_file_path, encoding='utf-8') as fb, \
            open(image_file_path, encoding='utf') as fi, \
            open(image_path, encoding='utf8') as fm:

        if args.aug_label != None:
            with open(map_weak_strong, encoding='utf-8') as ft:
                    box2label = json.load(ft)

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
                    bo = bsplits[-1].replace("\n", "")

                    if args.aug_label!=None:
                        if bo not in box2label.keys():
                            label.append(-100)
                        else:
                            if box2label[bo]==args.aug_label:
                                label.append(int(label2idx[t]))
                                entity_set.append(bo)
                            else:
                                label.append(-100)
                    else:
                        label.append(int(label2idx[t]))

                    bo = [int(b) for b in bo.split()]

                    box.append(bo)

                    actual = [int(b) for b in isplits[1].split()]
                    actual_box.append(actual)

        if word:
            words.append(word)
            labels.append(label)
            boxes.append(box)
            actual_boxes.extend(actual_box)

        entity_set=set(entity_set)
        print('{} length is {}'.format(args.text_aug,len(entity_set)))

    '''if shuffle:
        shuffled_boxes = []
        for box in boxes:
            indices = list(range(len(box)))
            random.shuffle(indices)
            box_list = []
            for index in indices:
                box_list.append(box[index])
            shuffled_boxes.append(box_list)
        return words, labels, shuffled_boxes, images, actual_boxes, images_path'''
    if shuffle_token:
        ################################token级打乱阅读顺序
        shuffled_words = []
        shuffled_labels = []
        shuffled_boxes = []
        shuffled_actual_boxes = []

        for word, label, box, actual_box in zip(words, labels, boxes, actual_boxes):
            indices = list(range(len(box)))
            random.shuffle(indices)

            word_list = []
            label_list = []
            box_list = []
            actual_box_list = []

            for index in indices:
                word_list.append(word[index])
                box_list.append(box[index])
                label_list.append(label[index])
                actual_box_list.append(actual_box[index])

            shuffled_words.append(word_list)
            shuffled_boxes.append(box_list)
            shuffled_labels.append(label_list)
            shuffled_actual_boxes.append(actual_box_list)

        return shuffled_words, shuffled_labels, shuffled_boxes, images, shuffled_actual_boxes, images_path
    elif shuffle_entity:
        # 实体级打乱阅读顺序
        shuffled_words = []
        shuffled_labels = []
        shuffled_boxes = []
        shuffled_actual_boxes = []

        for word, label, box, actual_box in zip(words, labels, boxes, actual_boxes):
            map_entity = []

            map_word = {}
            map_label = {}
            map_box = {}
            map_actual_box = {}

            idx = 0
            for j in range(len(box)):
                if j > 0:
                    if box[j] != box[j - 1]:
                        idx += 1
                map_entity.append(idx)

                map_box[idx] = box[j]
                map_word[idx] = word[j]
                map_label[idx] = label[j]
                map_box[idx] = box[j]
                map_actual_box[idx] = actual_box[j]

            entity_indice = list(range(idx + 1))
            random.shuffle(entity_indice)

            word_list = []
            label_list = []
            box_list = []
            actual_box_list = []

            for ent_indice in entity_indice:
                for id, entity in enumerate(map_entity):
                    if entity == ent_indice:
                        word_list.append(word[id])
                        box_list.append(box[id])
                        label_list.append(label[id])
                        actual_box_list.append(actual_box[id])

            shuffled_words.append(word_list)
            shuffled_labels.append(label_list)
            shuffled_boxes.append(box_list)
            shuffled_actual_boxes.append(actual_box_list)

        return shuffled_words, shuffled_labels, shuffled_boxes, images, shuffled_actual_boxes, images_path
    else:
        return words, labels, boxes, images, actual_boxes, images_path


class InputExample(object):
    def __init__(self,tokens, bboxes, attention_masks,labels):
        self.tokens = tokens
        self.bboxes = bboxes
        self.attention_masks = attention_masks
        self.labels=labels


examples=[]
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

            for i in range(len(split_token)):
                if i==0:
                    token_labels.extend([str_label])
                else:
                    token_labels.extend([-100])

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

    encoded_inputs["labels"] = torch.tensor(input_labels)
    if args.text_aug == 'WordSwapRandomCharacterDeletion':
        transformation = CompositeTransformation([WordSwapRandomCharacterDeletion()])
    elif args.text_aug == 'WordSwapChangeNumber':
        transformation = CompositeTransformation([WordSwapChangeNumber()])
    elif args.text_aug == 'WordSwapHomoglyphSwap':
        transformation = CompositeTransformation([WordSwapHomoglyphSwap()])

    if args.text_aug !='WordSwapMaskedLM' and args.text_aug!='WordSwapEmbedding' and args.text_aug!=None:
        constraints = [StopwordModification()]
        augmenter = Augmenter(transformation=transformation,
                              constraints=constraints,
                              pct_words_to_swap=1.0,
                              transformations_per_example=1)

        for i in range(len(input_labels)):
            for j in range(len(input_labels[i])):
                if input_labels[i][j]!=-100:
                    temp_num=input_ides[i][j]
                    temp_str=tokenizer_v1.convert_ids_to_tokens(temp_num)
                    res_str=temp_str[0]+augmenter.augment(temp_str[1:])[0]
                    res_input_ids=tokenizer_v1.convert_tokens_to_ids(res_str)
                    input_ides[i][j]=res_input_ids

    encoded_inputs["input_ids"]=torch.tensor(input_ides)
    encoded_inputs["attention_mask"]=torch.tensor(attention_masks)
    encoded_inputs["bbox"]=torch.tensor(bboxes)
    encoded_inputs["segment_ids"]=torch.tensor(segment_ids)

    return encoded_inputs

from torch.utils.data import Dataset, DataLoader

class V2Dataset(Dataset):
    def __init__(self, encoded_inputs):
        self.all_input_ids = encoded_inputs['input_ids']
        self.all_attention_masks = encoded_inputs['attention_mask']
        self.all_bboxes = encoded_inputs['bbox']
        self.all_labels = encoded_inputs['labels']

    def __len__(self):
        return len(self.all_labels)

    def __getitem__(self, index):
        return (
            self.all_input_ids[index],
            self.all_attention_masks[index],
            self.all_bboxes[index],
            self.all_labels[index],
        )


def main(args, config, shuffle_token=False, shuffle_entity=False):
    device = torch.device(args.device)

    #### Dataset ####
    print("Creating dataset")

    test_words, test_labels, test_boxes, test_images, test_actual_boxes, test_images_path = read_examples_from_file(args,
        data_dir,mode='test')

    test_dataset = V2Dataset(Process(test_words, test_boxes, test_labels))

    datasets_test = test_dataset

    data_test_loader = DataLoader(datasets_test, batch_size=1, num_workers=0)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    #### Model ####
    print("Creating model")

    model = testmodel(config=config, text_encoder=args.text_encoder, init_deit=True)

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
    image_paths = None
    model.eval()

    for j, (
            tokens,
            attention_masks,
            bboxes,
            labels) in enumerate(data_test_loader):
        with torch.no_grad():
            tokens = tokens.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            bboxes = bboxes.to(device, non_blocking=True)
            attention_masks = attention_masks.to(device, non_blocking=True)

            loss_cls, logits = model(text=tokens, bbox=bboxes, attention_mask=attention_masks,
                                     image=None,
                                     image_aug=None, labels=labels)

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = labels.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)

    preds = np.argmax(preds, axis=2)

    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != -100:
                out_label_list[i].append(idx2label[out_label_ids[i][j]])
                preds_list[i].append(idx2label[preds[i][j]])

    results = {
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list),
    }
    print(results)

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
    parser.add_argument('--checkpoint', default='funsd_v1_69.pth')
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--output_dir', default='train/')
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)

    parser.add_argument("--text_aug", type=str, default='WordSwapRandomCharacterDeletion')
    #TOTAL Weak Strong
    parser.add_argument("--aug_label", type=str, default=None)

    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    print(config)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config,shuffle_token=False,shuffle_entity=False)



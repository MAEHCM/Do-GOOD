import argparse
import os
from seqeval.metrics import precision_score, recall_score, f1_score
import numpy as np
import random
import json
from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import DataLoader
import ruamel.yaml as yaml
from model.funsd_model import testmodel
from transformers import BertTokenizer, AutoProcessor, LayoutLMv3Tokenizer
from get_aug_image import get_image
from textattack.transformations import CompositeTransformation
from textattack.constraints.pre_transformation import StopwordModification

from textattack.transformations.word_swaps import WordSwapChangeNumber,\
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

tokenizer_v3=LayoutLMv3Tokenizer.from_pretrained("microsoft/layoutlmv3-base")

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

        for i in range(len(encoded_inputs['labels'])):
            for j in range(len(encoded_inputs['labels'][i])):
                if encoded_inputs['labels'][i][j]!=-100:
                    temp_num=encoded_inputs['input_ids'][i][j]
                    temp_str=tokenizer_v3.convert_ids_to_tokens(temp_num)
                    res_str=temp_str[0]+augmenter.augment(temp_str[1:])[0]
                    res_input_ids=tokenizer_v3.convert_tokens_to_ids(res_str)
                    encoded_inputs['input_ids'][i][j]=res_input_ids

    encoded_inputs['input_ids'] = torch.tensor(encoded_inputs['input_ids'])
    encoded_inputs['attention_mask'] = torch.tensor(encoded_inputs['attention_mask'])
    encoded_inputs['bbox'] = torch.tensor(encoded_inputs['bbox'])

    encoded_inputs['labels'] = torch.tensor(encoded_inputs['labels'])
    encoded_inputs['actual_box'] = torch.tensor(encoded_inputs_2['bbox'])

    return encoded_inputs


from torch.utils.data import Dataset


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


def main(args, config, shuffle_token=False, shuffle_entity=False):
    device = torch.device(args.device)

    #### Dataset ####
    print("Creating dataset")

    test_words, test_labels, test_boxes, test_images, test_actual_boxes, test_images_path = read_examples_from_file(
        args,data_dir, mode='test', shuffle_token=shuffle_token, shuffle_entity=shuffle_entity)

    test_dataset = V2Dataset(Process(test_images, test_words, test_boxes, test_labels, test_actual_boxes),
                             test_images_path)

    datasets_test = test_dataset

    data_test_loader = DataLoader(datasets_test, batch_size=1, num_workers=0)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./config.yaml')
    parser.add_argument('--checkpoint', default='funsd_v3_align_69.pth')
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
    # WordSwapMaskedLM(method="bert-attack", max_candidates=12)
    # WordSwapEmbedding()
    # WordSwapHomoglyphSwap()
    # WordSwapChangeNumber()
    # WordSwapRandomCharacterDeletion()
    preds_list1, out_label_list1, ocr_list1, actual_ocr_list1, image_paths1, token_list1 = main(args, config,
                                                                                                shuffle_token=False,
                                                                                                shuffle_entity=False,
                                                                                                )



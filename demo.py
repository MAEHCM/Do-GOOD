from PIL import ImageDraw,ImageFont
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
from model.funsd_model import testmodel
from transformers import AutoProcessor
from get_aug_image import get_image

################################################################
data_dir="FUNSD/data"
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


def read_examples_from_file(data_dir, mode='train', shuffle_token=False,shuffle_entity=False,aug_text=None,shuffle_token_boxes=None,shuffle_entity_boxes=None):

    if aug_text:
        file_path = os.path.join(data_dir, str(aug_text)+"_"+"{}.txt".format(mode))
        box_file_path = os.path.join(data_dir, str(aug_text)+"_"+"{}_box.txt".format(mode))
        image_file_path = os.path.join(data_dir, str(aug_text)+"_"+"{}_image.txt".format(mode))
        image_path = os.path.join(data_dir, "{}_image_path.txt".format(mode))

    else:
        file_path = os.path.join(data_dir, "{}.txt".format(mode))
        box_file_path = os.path.join(data_dir, "{}_box.txt".format(mode))
        image_file_path = os.path.join(data_dir, "{}_image.txt".format(mode))
        image_path = os.path.join(data_dir, "{}_image_path.txt".format(mode))

    guid_index = 1

    word = []
    box = []
    label = []
    actual_box = []
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
                    guid_index += 1
                    word = []
                    box = []
                    label = []
                    actual_box = []
            else:
                splits = line.split("\t")
                bsplits = bline.split("\t")
                isplits = iline.split("\t")
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
    
    if shuffle_token_boxes:
        shuffled_boxes = []
        for box in boxes:
            indices = list(range(len(box)))
            random.shuffle(indices)
            box_list = []
            for index in indices:
                box_list.append(box[index])
            shuffled_boxes.append(box_list)
        return words, labels, shuffled_boxes, images, actual_boxes, images_path
    elif shuffle_token:
        ################################token级打乱阅读顺序
        shuffled_words = []
        shuffled_labels = []
        shuffled_boxes = []
        shuffled_actual_boxes = []

        for word,label,box,actual_box in zip(words,labels,boxes,actual_boxes):
            indices = list(range(len(box)))
            random.shuffle(indices)

            word_list=[]
            label_list=[]
            box_list = []
            actual_box_list=[]

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
    elif shuffle_entity_boxes:
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
    elif shuffle_entity:
        #实体级打乱阅读顺序
        shuffled_words = []
        shuffled_labels = []
        shuffled_boxes = []
        shuffled_actual_boxes = []

        for word,label,box,actual_box in zip(words,labels,boxes,actual_boxes):
            map_entity = []

            map_word = {}
            map_label = {}
            map_box = {}
            map_actual_box = {}

            idx = 0
            for j in range(len(box)):
                if j>0:
                    if box[j]!=box[j-1]:
                        idx+=1
                map_entity.append(idx)

                map_box[idx] = box[j]
                map_word[idx] = word[j]
                map_label[idx] = label[j]
                map_box[idx] = box[j]
                map_actual_box[idx] = actual_box[j]

            entity_indice=list(range(idx+1))
            random.shuffle(entity_indice)

            word_list = []
            label_list = []
            box_list = []
            actual_box_list = []

            for ent_indice in entity_indice:
                for id ,entity in enumerate(map_entity):
                    if entity==ent_indice:
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
    encoded_inputs_2 = processor(images, words, boxes=actual_boxes, word_labels=labels, padding="max_length", truncation=True)

    encoded_inputs['input_ids'] = torch.tensor(encoded_inputs['input_ids'])
    encoded_inputs['attention_mask'] = torch.tensor(encoded_inputs['attention_mask'])
    encoded_inputs['bbox'] = torch.tensor(encoded_inputs['bbox'])

    encoded_inputs['labels'] = torch.tensor(encoded_inputs['labels'])
    encoded_inputs['actual_box'] = torch.tensor(encoded_inputs_2['bbox'])

    return encoded_inputs


from torch.utils.data import Dataset


class DoDataset(Dataset):
    def __init__(self, encoded_inputs, test_images_path):
        self.all_images = test_images_path
        self.all_input_ids = encoded_inputs['input_ids']
        self.all_attention_masks = encoded_inputs['attention_mask']
        self.all_bboxes = encoded_inputs['bbox']
        self.all_labels = encoded_inputs['labels']
        self.all_actual_boxes=encoded_inputs['actual_box']

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


def main(args, shuffle_token=False,shuffle_entity=False,shuffle_entity_boxes=False,shuffle_token_boxes=False,aug_text=None):
    device = torch.device(args.device)

    print("Creating dataset")

    test_words, test_labels, test_boxes, test_images, test_actual_boxes, test_images_path = read_examples_from_file(
        data_dir, mode='test', shuffle_token=shuffle_token,shuffle_entity=shuffle_entity,shuffle_entity_boxes=shuffle_entity_boxes,shuffle_token_boxes=shuffle_token_boxes,aug_text=aug_text)

    test_dataset = DoDataset(Process(test_images, test_words, test_boxes, test_labels, test_actual_boxes), test_images_path)

    datasets_test = test_dataset

    data_test_loader = DataLoader(datasets_test, batch_size=1, num_workers=0)

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
    image_paths=None
    model.eval()

    for j, (image_path,
            tokens,
            attention_masks,
            bboxes,
            labels,actual_box) in enumerate(data_test_loader):

        with torch.no_grad():

            images, images_aug = get_image(image_path,empty=False)

            image = images.to(device, non_blocking=True)
            image_aug = images_aug.to(device, non_blocking=True)

            tokens = tokens.to(device, non_blocking=True)
            bboxes = bboxes.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            attention_masks = attention_masks.to(device, non_blocking=True)


            loss_cls, logits = model(text=tokens, bbox=bboxes, attention_mask=attention_masks,
                                     image=image, image_aug=image_aug, labels=labels,layout=True,model='v3')


        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = labels.detach().cpu().numpy()
            ocr_boxes=bboxes.detach().cpu().numpy()
            ocr_actual_boxes = actual_box.detach().cpu().numpy()
            out_tokens=tokens.detach().cpu().numpy()
            image_paths=image_path
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)
            ocr_boxes=np.append(ocr_boxes, bboxes.detach().cpu().numpy(), axis=0)
            ocr_actual_boxes=np.append(ocr_actual_boxes, actual_box.detach().cpu().numpy(), axis=0)
            out_tokens=np.append(out_tokens, tokens.detach().cpu().numpy(), axis=0)
            image_paths=np.append(image_paths,image_path,axis=0)

    preds = np.argmax(preds, axis=2)


    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]
    ocr_list=[[]for _ in range(out_label_ids.shape[0])]
    actual_ocr_list = [[] for _ in range(out_label_ids.shape[0])]
    token_list=[[] for _ in range(out_label_ids.shape[0])]

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

    return preds_list,out_label_list,ocr_list,actual_ocr_list,image_paths,token_list


def iob_to_label(label):
    if label !="O":
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
    parser.add_argument('--checkpoint', default='your model path')
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
    # WordSwapMaskedLM(method="bert-attack", max_candidates=12)
    # WordSwapEmbedding()
    # WordSwapHomoglyphSwap()
    # WordSwapChangeNumber()
    # WordSwapRandomCharacterDeletion()
    preds_list1, out_label_list1, ocr_list1, actual_ocr_list1, image_paths1,token_list1 = main(args,
                                                                                               shuffle_token=False,
                                                                                               shuffle_entity=False,
                                                                                               shuffle_token_boxes=False,
                                                                                               shuffle_entity_boxes=False,
                                                                                               aug_text=None)
    label2color = {
                   'OTHER': (0, 225, 255),
                   'HEADER': (225, 0, 255),
                   'QUESTION': (50, 50, 150),
                   'ANSWER': (225, 225, 0),
                   }
    for idx,img in enumerate(image_paths1):
        image = Image.open(img).convert('RGB')

        width,length=image.size

        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        for prediction, box in zip(preds_list1[idx], ocr_list1[idx]):
            box = unnormalize_box(box, width, length)
            prediction_label = iob_to_label(prediction)

            # 画
            draw.rectangle(box, outline=label2color[prediction_label],fill=label2color[prediction_label])
            draw.text((box[0] + 10, box[1] - 10), text=prediction_label, fill=label2color[prediction_label], font=font)

        image.save('your save img path'.format(idx))
        print('finish {}'.format(idx))



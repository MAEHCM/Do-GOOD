import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.utils.data.dataset import Dataset
from model.CDIP_model import Do_GOOD
from transformers import BertTokenizer, AutoConfig, LayoutLMv3Processor, AutoProcessor, LayoutLMv3Tokenizer
from utils import util
from get_aug_image import get_image

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

tokenizer_v3=LayoutLMv3Tokenizer.from_pretrained("microsoft/layoutlmv3-base")

def get_labels(path):
    with open(path,'r') as f:
        labels=f.read().splitlines()
    return labels

label_path='/mnt/disk2/hjb/archive/label.txt'

def normalize_bbox(bbox, width, length):
    return [
        int(1000 * bbox[0] / width),
        int(1000 * bbox[1] / length),
        int(1000 * bbox[2] / width),
        int(1000 * bbox[3] / length),
    ]


labeles=get_labels(label_path)
label2idx={label:i for i,label in enumerate(labeles)}
idx2label={i:label for i,label in enumerate(labeles)}

print(idx2label)
################################################################
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
        self.labels=labels
        self.root=root

def get_line_bbox(bboxs):
    x = [bboxs[i][j] for i in range(len(bboxs)) for j in range(0, len(bboxs[i]), 2)]
    y = [bboxs[i][j] for i in range(len(bboxs)) for j in range(1, len(bboxs[i]), 2)]

    x0, y0, x1, y1 = min(x), min(y), max(x), max(y)

    assert x1 >= x0 and y1 >= y0
    bbox = [[x0, y0, x1, y1] for _ in range(len(bboxs))]
    return bbox

def aug_layout(boxes):

    temp_box = [[] for _ in range(len(boxes))]
    flag_ids = [[] for _ in range(len(boxes))]

    flag = np.ones((1000, 1000), dtype=np.uint8) * 100000

    idx = 0
    for i in range(len(boxes)):

        if boxes[i][0]==0 and boxes[i][1]==0 and boxes[i][2]==0 and boxes[i][3]==0:
            continue

        x1, y1, x2, y2 = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]

        if (x1 >= 970 or x2 >= 970 or y1 == 1000 or y2 == 1000):
            temp_box[idx].append(boxes[i])
            flag_ids[idx].append(i)
        else:
            if i != 0:

                if (flag[x1][y1] == 100000 and flag[x1][y2] == 100000 and flag[x2][y1] == 100000 and flag[x2][
                    y2] == 100000):  # 如果点不在区域内
                    idx += 1
                    temp_box[idx].append(boxes[i])
                    flag[x1 - 70:x2 + 70, y1-30:y2+30] = idx
                    flag_ids[idx].append(i)
                else:
                    a, b, c, d = flag[x1][y1], flag[x1][y2], flag[x2][y1], flag[x2][y2]
                    if a != 100000:
                        tem_idx = a
                    elif b != 100000:
                        tem_idx = b
                    elif c != 100000:
                        tem_idx = c
                    else:
                        tem_idx = d

                    temp_box[tem_idx].append(boxes[i])
                    flag[x1 - 70:x2 + 70, y1-30:y2+30] = tem_idx
                    flag_ids[tem_idx].append(i)
            else:
                temp_box[idx].append(boxes[i])
                flag[x1 - 70:x2 + 70, y1-30:y2+30] = idx
                flag_ids[idx].append(i)

    res_box = []
    res_idx = []
    result_box = []

    for i in range(len(temp_box)):
        if len(temp_box[i]) == 0:
            continue
        else:
            temp = get_line_bbox(temp_box[i])
            for j in range(len(temp)):
                res_idx.append(flag_ids[i][j])
                res_box.append(temp[j])

    for i in range(len(res_idx)):
        result_box.append(res_box[res_idx[i]-1])
    result_box.insert(0, [0, 0, 0, 0])
    for i in range(512-len(result_box)):
        result_box.append([0, 0, 0, 0])

    result_box = torch.tensor(result_box, dtype=torch.long)
    return res_idx,result_box

tokenizer=LayoutLMv3Tokenizer.from_pretrained("microsoft/layoutlmv3-base")

class CDIP_dataset(Dataset):
    def __init__(self,roots='/mnt/disk2/hjb/for_train'):

        global aug_bboxes
        features_file=os.listdir(roots)

        if args.text_aug == 'WordSwapRandomCharacterDeletion':
            transformation = CompositeTransformation([WordSwapRandomCharacterDeletion()])
        elif args.text_aug == 'WordSwapChangeNumber':
            transformation = CompositeTransformation([WordSwapChangeNumber()])
        elif args.text_aug == 'WordSwapHomoglyphSwap':
            transformation = CompositeTransformation([WordSwapHomoglyphSwap()])
        elif args.text_aug == 'WordSwapMaskedLM':
            transformation = CompositeTransformation([WordSwapMaskedLM()])
        elif args.text_aug == 'WordSwapEmbedding':
            transformation = CompositeTransformation([WordSwapEmbedding()])

        self.all_root = []
        self.all_tokens = []
        self.all_bboxes = []
        self.all_attention_masks = []
        self.all_labels = []

        for feature_file in features_file:
            file_root=os.path.join(roots,feature_file)
            f=torch.load(file_root)

            root = f[0].root
            tokens = f[0].tokens
            bboxes = f[0].bboxes
            labels = int(f[0].labels[0].strip('\n'))

            if args.text_aug != None:
                temp_tokens=[]
                constraints = [RepeatModification(), StopwordModification()]

                augmenter = Augmenter(transformation=transformation,
                                      constraints=constraints,
                                      pct_words_to_swap=1.0,
                                      transformations_per_example=1)
                for i in range(len(tokens)):
                    if len(tokens[i])>10:
                        res_str = tokens[i]
                    else:
                        res_str = augmenter.augment(tokens[i])[0]
                    temp_tokens.append(res_str)

                tokenizer_res = tokenizer.encode_plus(temp_tokens, boxes=bboxes)

            elif args.aut_layout:
                _, aug_bboxes = aug_layout(bboxes)
                tokenizer_res = tokenizer.encode_plus(tokens, boxes=aug_bboxes)

            else:
                tokenizer_res = tokenizer.encode_plus(tokens, boxes=bboxes)

            if len(tokenizer_res['input_ids']) > 511:
                tokenizer_res['input_ids'] = tokenizer_res['input_ids'][: 511]
                tokenizer_res['bbox'] = tokenizer_res['bbox'][: 511]
                tokenizer_res['attention_mask'] = tokenizer_res['attention_mask'][:511]

                tokenizer_res['input_ids'][-1] = tokenizer.sep_token_id
                tokenizer_res['bbox'][-1] = tokenizer.sep_token_box
                tokenizer_res['attention_mask'][-1] = 1

            for i in range(512 - len(tokenizer_res['input_ids'])):
                tokenizer_res['input_ids'].append(tokenizer.pad_token_id)
                tokenizer_res['bbox'].append(tokenizer.pad_token_box)
                tokenizer_res['attention_mask'].append(1)


            self.all_root.extend(root)
            self.all_tokens.append(torch.tensor(tokenizer_res['input_ids'], dtype=torch.long))
            self.all_bboxes.append(torch.tensor(tokenizer_res['bbox'], dtype=torch.long))
            self.all_attention_masks.append(torch.tensor(tokenizer_res['attention_mask'], dtype=torch.long))
            self.all_labels.append(torch.tensor(labels, dtype=torch.long))

    def __len__(self):
        return len(self.all_root)

    def __getitem__(self, index):
        return (
            self.all_root[index],
            self.all_tokens[index],
            self.all_bboxes[index],
            self.all_attention_masks[index],
            self.all_labels[index]
        )

#####################################################################


def main(args, config):
    util.init_distributed_mode(args)
    #Not using distributed mode
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + util.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset ####
    print("Creating dataset")
    datasets_test=CDIP_dataset('/mnt/disk2/hjb/sample_cdip_distord_ocr')

    data_test_loader=DataLoader(datasets_test, batch_size=1, num_workers=0,drop_last=True)

    #### Model ####
    print("Creating model")

    model = Do_GOOD()

    model = model.to(device)

    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['model']
        model.load_state_dict(state_dict)
        print('load checkpoint from %s' % args.checkpoint)

    print("Start training")
    start_time = time.time()

    correct = 0
    number = 0
    model.eval()
    label_preds = [0 for _ in range(16)]
    label_gts=[0 for _ in range(16)]

    for j, (image_path,
            tokens,
            bboxes,
            attention_masks,
            labels) in enumerate(data_test_loader):
        with torch.no_grad():
            if args.image_aug:
                batch_flag=j
            else:
                batch_flag=None
            images, images_aug = get_image(image_path,bboxes,cdip=False,empty=False,mscoco=args.image_aug,batch=batch_flag)

            image = images.to(device, non_blocking=True)
            image_aug = images_aug.to(device, non_blocking=True)

            # torch.Size([6, 3, 224, 224])

            tokens = tokens.to(device, non_blocking=True)
            bboxes = bboxes.to(device, non_blocking=True)

            attention_masks = attention_masks.to(device, non_blocking=True)

            labels = labels.to(device, non_blocking=True)

            loss_cls, logits = model(text=tokens, bbox=bboxes, attention_mask=attention_masks,
                                                  image=image,
                                                  image_aug=image_aug, labels=labels,layout=True)

            correct += (torch.argmax(logits, dim=-1) == labels).float().sum().detach().cpu().numpy()
            number += labels.shape[0]


        label_pred = torch.argmax(logits, dim=-1)[0].detach().cpu().numpy()
        label_gt=labels[0].detach().cpu().numpy()

        if label_pred==label_gt:
            label_gts[label_gt]+=1
            label_preds[label_gt]+=1
        else:
            label_gts[label_gt]+=1

    results = {
        # 此处必须用ner的形式才可以计算结果
        "Accuracy": '{:.7f}'.format(correct / number)
    }

    print(results)


    resu=''
    for i in range(16):
        if label_gts[i]!=0:
            print('{} of label Acc is {}%'.format(i,label_preds[i]/label_gts[i]))
            if args.text_aug:
                resu += '{},label of {} acc: {}%'.format(args.text_aug,i, str(label_preds[i] / label_gts[i]))+'\n'
            elif args.image_aug:
                resu += '{},label of {} acc: {}%'.format('mscoco', i, str(label_preds[i] / label_gts[i])) + '\n'
        else:
            print('{} of label Acc is 0.00%'.format(i))
            resu += 'label of {} acc: 0.00%'.format(i)+'\n'

    if args.image_aug:
        with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
            f.write('image_coco'+results["Accuracy"] + "\n")
            f.write(resu)
    else:
        with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
            f.write(args.text_aug+results["Accuracy"] + "\n")
            f.write(resu)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./config.yaml')
    parser.add_argument('--checkpoint', default='/cdip_sample_12.pth')
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--output_dir', default='train/')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=False, type=bool)

    # WordSwapMaskedLM(method="bert-attack", max_candidates=12)
    # WordSwapEmbedding()
    # WordSwapHomoglyphSwap()
    # WordSwapChangeNumber()
    # WordSwapRandomCharacterDeletion()

    parser.add_argument("--text_aug", type=str, default=None)
    parser.add_argument("--image_aug", type=str, default=False)
    parser.add_argument("--aut_layout", type=str, default=None)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    print(config)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)
import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.utils.data.dataset import Dataset
from model.CDIP_model import Do_GOOD
from transformers import BertTokenizer, AutoConfig, LayoutLMv3Processor, AutoProcessor, LayoutLMv2Tokenizer
from utils import util
from utils.dataset import create_dataset,create_sampler,create_loader
from utils.scheduler_factory import create_scheduler
from utils.optim_factory import create_optimizer
from get_aug_image import get_image

def get_labels(path):
    with open(path,'r') as f:
        labels=f.read().splitlines()
    return labels

label_path='/archive/label.txt'

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
        self.labels=labels,
        self.root=root,

tokenizer=LayoutLMv2Tokenizer.from_pretrained("microsoft/layoutlmv2-base-uncased")


class CDIP_dataset(Dataset):
    def __init__(self,root='/for_train'):

        features_file=os.listdir(root)

        features_list=[]

        for feature_file in features_file:
            file_root=os.path.join(root,feature_file)
            features=torch.load(file_root)
            features_list.append(features)

        self.features=features_list

        self.all_root = []
        self.all_tokens = []
        self.all_bboxes = []
        self.all_attention_masks = []
        self.all_labels = []

        for f in self.features:

            root=f[0].root
            tokens=f[0].tokens
            bboxes=f[0].bboxes
            labels=int(f[0].labels[0].strip('\n'))

            tokenizer_res=tokenizer.encode_plus(text=tokens,boxes=bboxes)

            if len(tokenizer_res['input_ids']) > 511:
                tokenizer_res['input_ids'] = tokenizer_res['input_ids'][: 511]
                tokenizer_res['bbox'] = tokenizer_res['bbox'][: 511]
                tokenizer_res['attention_mask'] = tokenizer_res['attention_mask'][:511]

                tokenizer_res['input_ids'][-1] = tokenizer.sep_token_id
                tokenizer_res['bbox'][-1] = tokenizer.sep_token_box
                tokenizer_res['attention_mask'][-1] = 1

            for i in range(512-len(tokenizer_res['input_ids'])):
                tokenizer_res['input_ids'].append(tokenizer.pad_token_id)
                tokenizer_res['bbox'].append(tokenizer.pad_token_box)
                tokenizer_res['attention_mask'].append(1)

            self.all_root.extend(root)
            self.all_tokens.append(torch.tensor(tokenizer_res['input_ids'],dtype=torch.long))
            self.all_bboxes.append(torch.tensor(tokenizer_res['bbox'],dtype=torch.long))
            self.all_attention_masks.append(torch.tensor(tokenizer_res['attention_mask'],dtype=torch.long))
            self.all_labels.append(torch.tensor(labels,dtype=torch.long))


    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return (
            self.all_root[index],
            self.all_tokens[index],
            self.all_bboxes[index],
            self.all_attention_masks[index],
            self.all_labels[index]
        )

#####################################################################

def train(model,data_loader, data_test_loader,optimizer, epoch, warmup_steps, device, scheduler, config):
    # train
    model.train()

    metric_logger = util.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', util.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', util.SmoothedValue(window_size=50, fmt='{value:.4f}'))


    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 100
    warmup_iterations = warmup_steps * step_size

    if args.distributed:
        data_loader.sampler.set_epoch(epoch)

    for i, (image_path,
               tokens,
               bboxes,
            attention_masks,
               labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        optimizer.zero_grad()

        images, images_aug= get_image(image_path,bboxes,cdip=False)

        image = images.to(device, non_blocking=True)
        image_aug = images_aug.to(device, non_blocking=True)

        #torch.Size([6, 3, 224, 224])

        labels=labels.to(device,non_blocking=True)
        tokens=tokens.to(device,non_blocking=True)
        bboxes=bboxes.to(device,non_blocking=True)


        attention_masks=attention_masks.to(device,non_blocking=True)


        loss_cls,_=model(text=tokens,bbox=bboxes,attention_mask=attention_masks,image=image,image_aug=image_aug,labels=labels)

        loss=loss_cls

        loss.backward()

        optimizer.step()

        metric_logger.update(loss=loss.item())

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])


        if epoch == 0 and i % step_size == 0 and i <= warmup_iterations:
            scheduler.step(i // step_size)


    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())

    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


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
    #42

    start_epoch = 0

    max_epoch = config['schedular']['epochs']#30
    warmup_steps = config['schedular']['warmup_epochs']#20

    #### Dataset ####
    print("Creating dataset")
    datasets = [CDIP_dataset('/sample_cdip_train')]
    datasets_test=CDIP_dataset('/sample_cdip_test')

    if args.distributed:
        num_tasks = util.get_world_size()
        global_rank = util.get_rank()
        samplers = create_sampler(datasets, [True], num_tasks, global_rank)
    else:
        samplers = [None]



    data_loader = \
    create_loader(datasets, samplers, batch_size=[4], num_workers=[0], is_trains=[True],
                  collate_fns=[None])[0]

    data_test_loader=DataLoader(datasets_test, batch_size=4, num_workers=0,drop_last=True)

    #### Model ####
    print("Creating model")

    model = Do_GOOD()

    model = model.to(device)

    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))

    arg_opt = util.AttrDict(config['optimizer'])
    #{'opt': 'adamw', 'lr': 0.0001, 'weight_decay': 0.02}
    optimizer = create_optimizer(arg_opt, model)

    arg_sche = util.AttrDict(config['schedular'])
    #{'sched': 'cosine', 'lr': 0.0001, 'epochs': 30, 'min_lr': 1e-05, 'decay_rate': 1, 'warmup_lr': 1e-05, 'warmup_epochs': 20, 'cooldown_epochs': 0}
    lr_scheduler= create_scheduler(arg_sche, optimizer)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['model']
        model.load_state_dict(state_dict)
        print('load checkpoint from %s' % args.checkpoint)

    model_without_ddp = model
    #False 不采用分布式
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],find_unused_parameters=True)
        model_without_ddp = model.module

    print("Start training")
    start_time = time.time()

    for epoch in range(start_epoch, max_epoch):

        if epoch > 0:
            lr_scheduler.step(epoch + warmup_steps)

        train_stats = train(model,data_loader,data_test_loader, optimizer, epoch, warmup_steps, device, lr_scheduler, config)
        if util.is_main_process():
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         }
            save_obj = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'config': config,
                'epoch': epoch,
            }
            torch.save(save_obj, os.path.join(args.output_dir, 'v2_cdip_%02d.pth' % epoch))

            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

        #dist.barrier()

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
                optimizer.zero_grad()

                images, images_aug = get_image(image_path,bboxes,cdip=False)

                image = images.to(device, non_blocking=True)
                image_aug = images_aug.to(device, non_blocking=True)

                # torch.Size([6, 3, 224, 224])

                tokens = tokens.to(device, non_blocking=True)
                bboxes = bboxes.to(device, non_blocking=True)

                attention_masks = attention_masks.to(device, non_blocking=True)

                labels = labels.to(device, non_blocking=True)

                loss_cls, logits = model(text=tokens, bbox=bboxes, attention_mask=attention_masks,
                                                      image=image,
                                                      image_aug=image_aug, labels=labels)

            correct += (torch.argmax(logits, dim=-1) == labels).float().sum().detach().cpu().numpy()
            number += labels.shape[0]

            for i in range(4):
                label_pred = torch.argmax(logits, dim=-1)[i].detach().cpu().numpy()
                label_gt=labels[i].detach().cpu().numpy()

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


        for i in range(16):
            print('{} of label Acc is'.format(label_preds[i]/label_gts[i]))

        with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
            f.write(results["Accuracy"] + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./config.yaml')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--output_dir', default='train/')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    print(config)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)
import argparse
import json
from textattack.transformations import CompositeTransformation
import os

from PIL import Image
from transformers import AutoTokenizer, LayoutLMv3Tokenizer

#####################################################################
import nltk
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

###################################################################



def normalize_bbox(bbox, width, length):
    return [
        int(1000 * bbox[0] / width),
        int(1000 * bbox[1] / length),
        int(1000 * bbox[2] / width),
        int(1000 * bbox[3] / length),
    ]


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


def actual_bbox_string(box, width, length):
    return (
            str(box[0])
            + " "
            + str(box[1])
            + " "
            + str(box[2])
            + " "
            + str(box[3])
            + "\t"
            + str(width)
            + " "
            + str(length)
    )


def get_line_bbox(bboxs):
    x = [bboxs[i][j] for i in range(len(bboxs)) for j in range(0, len(bboxs[i]), 2)]
    y = [bboxs[i][j] for i in range(len(bboxs)) for j in range(1, len(bboxs[i]), 2)]

    x0, y0, x1, y1 = min(x), min(y), max(x), max(y)

    assert x1 >= x0 and y1 >= y0
    bbox = [[x0, y0, x1, y1] for _ in range(len(bboxs))]
    return bbox


def convert(args):
    with open(
            os.path.join(args.output_dir, args.text_aug+'_'+args.data_split + ".txt.tmp"),
            "w",
            encoding="utf8",
    ) as fw, open(
        os.path.join(args.output_dir, args.text_aug+'_'+args.data_split + "_box.txt.tmp"),
        "w",
        encoding="utf8",
    ) as fbw, open(
        os.path.join(args.output_dir, args.text_aug+'_'+args.data_split + "_image.txt.tmp"),
        "w",
        encoding="utf8",
    ) as fiw :
        # Set up transformation using CompositeTransformation()

        # WordSwapRandomCharacterDeletion()
        # WordSwapChangeNumber()
        # WordSwapEmbedding()
        # WordSwapHomoglyphSwap()
        if args.text_aug=='WordSwapMaskedLM':
            transformation = CompositeTransformation([WordSwapMaskedLM()])
        elif args.text_aug=='WordSwapRandomCharacterDeletion':
            transformation = CompositeTransformation([WordSwapRandomCharacterDeletion()])
        elif args.text_aug == 'WordSwapChangeNumber':
            transformation = CompositeTransformation([WordSwapChangeNumber()])
        elif args.text_aug == 'WordSwapEmbedding':
            transformation = CompositeTransformation([WordSwapEmbedding()])
        elif args.text_aug == 'WordSwapHomoglyphSwap':
            transformation = CompositeTransformation([WordSwapHomoglyphSwap()])


        # constraints.append(MaxWordsPerturbed(max_percent=0.4))
        constraints = [RepeatModification(), StopwordModification()]
        # constraints.append(LevenshteinEditDistance(max_edit_distance=2))

        # Create augmenter with specified parameters
        augmenter = Augmenter(transformation=transformation,
                              constraints=constraints,
                              pct_words_to_swap=0.1,
                              transformations_per_example=1)

        for file in os.listdir(args.data_dir):
            file_path = os.path.join(args.data_dir, file)
            with open(file_path, "r", encoding="utf8") as f:
                data = json.load(f)
            image_path = file_path.replace("annotations", "images").replace("json", "png")

            file_name = os.path.basename(image_path)

            image = Image.open(image_path)

            width, length = image.size

            tokens = []
            bboxes = []
            ner_tags = []

            for item in data["form"]:
                words, label = item["words"], item["label"]
                words = [w for w in words if w["text"].strip() != ""]
                line_word = []

                for w in words:
                    if w["text"].strip() != "":
                        if args.text_aug == "WordSwapMaskedLM" or args.text_aug == "WordSwapEmbedding":
                            attack_str = augmenter.augment(w["text"])
                            line_word.append(attack_str[0])
                        else:
                            line_word.append(w["text"])

                if len(words) == 0:
                    continue
                if label == "other":
                    for idx,w in enumerate(line_word):
                        tokens.append(w)
                        ner_tags.append("O")
                else:
                    tokens.append(line_word[0])
                    ner_tags.append("B-" + label.upper())
                    for idx,w in enumerate(line_word[1:]):
                        tokens.append(w)
                        ner_tags.append("I-" + label.upper())
                cur_line_bboxes=[]
                if label == "other":
                    for w in words:
                        cur_line_bboxes.append(normalize_bbox(w["box"], width,length))
                else:
                    cur_line_bboxes.append(normalize_bbox(words[0]["box"], width,length))
                    for w in words[1:]:
                        cur_line_bboxes.append(normalize_bbox(w["box"], width,length))

                cur_line_bboxes = get_line_bbox(cur_line_bboxes)
                bboxes.extend(cur_line_bboxes)

            for i in range(len(bboxes)):
                fw.write(tokens[i] + "\t" + ner_tags[i] + "\n")
                fbw.write(
                    tokens[i]
                    + "\t"
                    + string_box(bboxes[i])
                    + "\n"
                )
                fiw.write(
                    tokens[i]
                    + "\t"
                    + actual_bbox_string(bboxes[i], width, length)
                    + "\t"
                    + file_name
                    + "\n"
                )

            fw.write("\n")
            fbw.write("\n")
            fiw.write("\n")


def seg_file(file_path, tokenizer, max_len, flag=False):
    subword_len_counter = 0
    output_path = file_path[:-4]
    with open(file_path, "r", encoding="utf8"
              ) as f_p, open(
        output_path, "w", encoding="utf8"
    ) as fw_p:
        path_list = []


        for line in f_p:

            line = line.rstrip()

            if flag:
                if line!="":
                    cur_star=line.split("\t")[-1]
                else:
                    path_list.append(cur_star)

            if not line:
                fw_p.write(line + "\n")
                subword_len_counter = 0
                continue

            token = line.split("\t")[0]
            current_subwords_len = len(tokenizer.tokenize(token))

            if current_subwords_len == 0:
                continue

            if (subword_len_counter + current_subwords_len) > max_len:
                fw_p.write("\n" + line + "\n")
                if flag:
                    img = line.split("\t")[-1]
                    path_list.append(img)

                subword_len_counter = current_subwords_len
                continue

            subword_len_counter += current_subwords_len

            fw_p.write(line + "\n")
        if flag:
            #############################################/mnt/disk2/hjb/FUNSD/data
            with open("/FUNSD/test_image_path.txt", 'w') as ft:
                for i in range(len(path_list)):
                    ft.write('/FUNSD/testing_data/images/' + path_list[i] + '\n')


def seg(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        do_lower_case=True
    )
    seg_file(
        os.path.join(args.output_dir, args.text_aug+'_'+args.data_split + ".txt.tmp"),
        tokenizer,
        args.max_len,
        False
    )
    seg_file(
        os.path.join(args.output_dir, args.text_aug+'_'+args.data_split + "_box.txt.tmp"),
        tokenizer,
        args.max_len,
        False
    )
    seg_file(
        os.path.join(args.output_dir, args.text_aug+'_'+args.data_split + "_image.txt.tmp"),
        tokenizer,
        args.max_len,
        True
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/FUNSD/testing_data/annotations")
    parser.add_argument("--data_split", type=str, default="test")
    parser.add_argument("--output_dir", type=str, default="/FUNSD/data")
    parser.add_argument("--model_name_or_path", type=str, default="microsoft/layoutlmv3-base")
    parser.add_argument("--max_len", type=int, default=510)

    parser.add_argument("--text_aug", type=str, default="WordSwapEmbedding")

    args = parser.parse_args()

    convert(args)

    seg(args)

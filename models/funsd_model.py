from torch import nn

from transformers import LayoutLMv2ForTokenClassification, \
    LayoutLMForTokenClassification,\
LayoutLMv3ForTokenClassification


img_path = "FUNSD/training_data/images"
label_path = 'FUNSD/label.txt'


def get_labels(path):
    with open(path, 'r') as f:
        labels = f.read().splitlines()
    if 'O' not in labels:
        labels = ["O"] + labels
    return labels


labeles = get_labels(label_path)
label2idx = {label: i for i, label in enumerate(labeles)}
idx2label = {i: label for i, label in enumerate(labeles)}

print(idx2label)


class testmodel(nn.Module):
    def __init__(self,
                 ):
        super().__init__()

        self.fusion_model_v1 = LayoutLMForTokenClassification.from_pretrained("microsoft/layoutlm-base-uncased",
                                                                             label2id=label2idx,
                                                                             id2label=idx2label,
                                                                             )
        self.fusion_model_v2 = LayoutLMv2ForTokenClassification.from_pretrained("microsoft/layoutlmv2-base-uncased",
                                                                                label2id=label2idx,
                                                                                id2label=idx2label,
                                                                                )
        self.fusion_model_v3 = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base",
                                                                                label2id=label2idx,
                                                                                id2label=idx2label,
                                                                                )



    def forward(
            self,
            text=None,
            bbox=None,
            attention_mask=None,
            image=None,
            labels=None,
            seg_ids=None,
            model=None,
    ):

        if model=='v1':
            outputs = self.fusion_model_v1(
                input_ids=text,
                bbox=bbox,
                attention_mask=attention_mask,
                image=image,
                labels=labels,
                token_type_ids=seg_ids
            )
        elif model=='v2':
            outputs = self.fusion_model_v2(
                input_ids=text,
                bbox=bbox,
                attention_mask=attention_mask,
                image=image,
                labels=labels,
                token_type_ids=seg_ids
            )
        elif model == 'v3':
            outputs = self.fusion_model_v3(
                input_ids=text,
                bbox=bbox,
                attention_mask=attention_mask,
                image=image,
                labels=labels,
                token_type_ids=seg_ids
            )

        return outputs.loss, outputs.logits


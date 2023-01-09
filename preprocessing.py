# -*- coding: utf-8 -*-
import os
import spacy
import torch
import logging
import argparse

from tqdm import tqdm
from fairseq.models.bart import BARTModel

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


def read_lines(file_path):
    files = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            files.append(line.strip())
    return files


def get_indices(target, tokens):
    """
    Get the index of token that is part of the target.

    Args:
        target: "Mohammad Javad Zarif"
        tokens: ['Moh', 'ammad', ' Jav', 'ad', ' Zar', 'if', ' has', ' spent', ...]

    Return:
        List[int]: [1, 1, 1, 1, 1, 1, 0, 0, ...]
    """
    all_indices = []
    for i, t in enumerate(tokens):
        t = t.strip()
        indices = []
        if t in target:
            indices.append(i)
            if t == target:
                all_indices.extend(indices)
                break
            elif i + 1 < len(tokens):
                for ni, rt in enumerate(tokens[i + 1:]):
                    t += rt
                    indices.append(i + ni + 1)
                    if t == target:
                        all_indices.extend(indices)
                        break
                    elif t not in target:
                        break
    return all_indices


def build_mask(target, sentence, encode_func, decode_func):
    """
    Args:
        target (List[str]): "Mohammad Javad Zarif"
        sentence (str): "Mohammad Javad Zarif has spent more time with..."

    Return:
        List[int]: 1 if the token in this position is part of an entity
    """
    assert target in sentence
    tokens = [decode_func(torch.tensor([i])) for i in encode_func(sentence)]
    indices = get_indices(target, tokens)
    mask = torch.zeros(len(tokens), dtype=torch.long)
    for i in indices:
        mask[i] = 1
    return mask


def main(args):
    # load BART model
    bart = BARTModel.from_pretrained(args.bart_dir,
                                     checkpoint_file='model.pt',
                                     data_name_or_path=args.bart_dir)
    bart.cuda()
    bart.eval()
    bart.half()

    # load summaries
    summaries = read_lines(args.summary_path)

    # tokenization
    encode_func = lambda x: bart.task.source_dictionary.encode_line(bart.bpe.encode(x), append_eos=True).long()
    decode_func = bart.decode

    nlp = spacy.load(args.spacy_tokenizer)

    ref_masks = []
    for summ in tqdm(summaries):
        t_ents = [e.text for e in nlp(summ).ents]

        mask = None
        for e in t_ents:
            tmp_mask = build_mask(e, summ, encode_func, decode_func)
            if mask is None:
                mask = tmp_mask
            else:
                mask.masked_fill_(tmp_mask.bool(), 1)

        # no entities found in the summary
        if mask is None:
            length = encode_func(summ).shape[0]
            mask = torch.zeros(length, dtype=torch.long)

        ref_masks.append(mask)

    with open(args.output_file, "w") as wf:
        for mask in ref_masks:
            mask_str = " ".join([str(i) for i in mask.tolist()])
            wf.write(mask_str)
            wf.write("\n")

    logging.info("Masks saved at: {}".format(args.output_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bart_dir",
        type=str,
        default="BART_models/bart.large.xsum"
    )
    parser.add_argument(
        "--summary_path",
        type=str,
        default="val.target"
    )
    parser.add_argument(
        "--spacy_tokenizer",
        type=str,
        default="en_core_web_sm",
        const="en_core_web_sm",
        nargs="?",
        choices=["en_core_web_sm", "en_core_web_trf"],
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="val.mask"
    )
    args = parser.parse_args()
    main(args)
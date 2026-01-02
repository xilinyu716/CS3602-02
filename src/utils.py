import torch
from datasets import load_dataset
from transformers import AutoTokenizer

def load_data(dataset_name, split="test", streaming=False):
    assert dataset_name == "wikitext"
    return load_dataset("/home/xlyu/datasets/wikitext", split=split)


def get_tokenizer(model_name="EleutherAI/pythia-70m"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def calculate_ppl(nlls, token_count):
    return torch.exp(torch.stack(nlls).sum() / token_count)

# -*- coding: utf-8 -*-
import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
import os
import re
# import tkitFile
import regex
# from elasticsearch import Elasticsearch
# from elasticsearch_dsl import Search
# from elasticsearch_dsl import Q
# from config import *
from tqdm import tqdm
import time


class tkitMarkerFast:
    """[自动从ner标注结果中提取数据]
    """

    def __init__(self, model_path="../model", device='cpu'):
        self.model_path = model_path
        self.labels_file = os.path.join(model_path, "labels.txt")
        self.device = device
        pass

    def __del__(self):
        # self.release()
        pass

    def release(self):
        # print("释放显存")
        self.model.cpu()

        torch.cuda.empty_cache()
        pass
        # torch.cuda.empty_cache()
        del self.model
        del self.tokenizer
        del self.lablels_dict
        # gc.collect()
    # @profile

    def load_model(self):
        # tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        # model.to(self.device)
        f2 = open(self.labels_file, 'r')
        lablels_dict = {}
        for i, line in enumerate(f2):
            # l=line.split(" ")
            l = line.replace("\n", '')
            # print(l)
            lablels_dict[i] = l
        f2.close()
        self.lablels_dict = lablels_dict
        # self.model=model
        # self.tokenizer=tokenizer
        # self.model.eval()
        return self.model, self.tokenizer
    # @profile

    def filterPunctuation(self, x):
        x = regex.sub(r'[‘’]', "'", x)
        x = regex.sub(r'[“”]', '"', x)
        x = regex.sub(r'[…]', '...', x)
        x = regex.sub(r'[—]', '-', x)
        x = regex.sub(r"&nbsp", "", x)
        return x

    def pre(self, word, text):

        model = self.model
        # text=word+" [SEP] "+text
        # lenth = 500-len(word)
        all_ms = []
        n = 0
        h_i = 2+len(word)
        with torch.no_grad():
            # model = AutoModelForTokenClassification.from_pretrained(self.model_path)
            # model.to(self.device)
            text = self.filterPunctuation(text)

            ids = self.tokenizer.encode_plus(
                word, text_mini, max_length=256, add_special_tokens=True)
            # print(ids)
            input_ids = torch.tensor(
                ids['input_ids']).unsqueeze(0)  # Batch size 1
            labels = torch.tensor(
                [1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
            outputs = model(input_ids, labels=labels)
            # print("outputs",outputs)
            tmp_eval_loss, logits = outputs[:2]
            torch.argmax(logits, axis=2).tolist()[0]

            # for text_mini in self.cut_text(text, lenth):
            #     # text_mini=word+"[SEP]"+text_mini
            #     # print(word,"text_mini",text_mini)
            #     n = n+1
            #     ids = self.tokenizer.encode_plus(
            #         word, text_mini, max_length=512, add_special_tokens=True)
            #     # print(ids)
            #     input_ids = torch.tensor(
            #         ids['input_ids']).unsqueeze(0)  # Batch size 1
            #     labels = torch.tensor(
            #         [1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
            #     outputs = model(input_ids, labels=labels)
            #     # print("outputs",outputs)
            #     tmp_eval_loss, logits = outputs[:2]
            #     # ids=tokenizer.encode(text)
            #     # print(ids['token_type_ids'])

            #     # print("\n".join([i for i in self.lablels_dict.keys()]))
            #     words = []
            #     for i, m in enumerate(torch.argmax(logits, axis=2).tolist()[0]):
            #         # print(m)
            #         # if i<h_i:
            #         #     continue

            #         # print(i,m,ids['input_ids'][i],self.tokenizer.convert_ids_to_tokens(ids['input_ids'][i]),self.lablels_dict[m])
            #         # print(h_i)
            #         word = self.tokenizer.convert_ids_to_tokens(
            #             ids['input_ids'][i])
            #         # try:
            #         #     word=text_mini[i-h_i]
            #         # except:
            #         #     continue
            #         # print(word)

            #         if m >= len(self.lablels_dict):
            #             mark_lable = "X"
            #         else:
            #             mark_lable = self.lablels_dict[m]

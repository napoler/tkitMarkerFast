# -*- coding: utf-8 -*-
import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
import os
import re
# import tkitFile
import regex
from tqdm import tqdm
import time

import BMESBIO2Data

class MarkerFast:
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

    def pre(self, text):
        """[自动预测文本的标记数据]
        

        Args:
            text ([type]): [输入文本即可限制256]

        Returns:
            [标记后,words,mark,data]: [返回标记后数据和标记信息 tag格式数据]
        """
        data=[]
        model = self.model
        # text=word+" [SEP] "+text
        # lenth = 500-len(word)
        # all_ms = []
        # n = 0
    
        with torch.no_grad():
            text = self.filterPunctuation(text)

            ids = self.tokenizer.encode_plus(
                text, None, max_length=256, add_special_tokens=True,truncation=True)
            # print(ids)
            input_ids = torch.tensor(
                ids['input_ids']).unsqueeze(0)  # Batch size 1
            labels = torch.tensor(
                [1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
            outputs = model(input_ids, labels=labels)
            # print("outputs",outputs) 
            words=self.tokenizer.tokenize(text)
            tmp_eval_loss, logits = outputs[:2]
            # print("words",words)

            for i,(m,w) in enumerate( zip(torch.argmax(logits, axis=2).tolist()[0],words)):
                # print(w)
                if m >=len(self.lablels_dict):
                    mark_lable="X"
                else:
                    mark_lable=self.lablels_dict[m]
                    # print(w,mark_lable)
                # print(words[i],mark_lable)
                    data.append(w+" "+mark_lable+"")
            M2D=BMESBIO2Data.BMESBIO2Data()
            # print(M2D.toData(data))
            # (['【', '禁', '忌', '证', '】', '顽', '固', '、', '难', '治', '性', '高', '血', '压', '#', '禁', '忌', '症', '、', '严', '重', '的', '心', '血', '管', '疾', '病', '#', '禁', '忌', '症', '及', '甲', '亢', '#', '禁', '忌', '症', '患', '者', '。'], [{'type': '禁忌症', 'word': ['固', '、'], 'start': 6, 'end': 7}, {'type': '禁忌症', 'word': ['治', '性', '高', '血', '压', '#', '忌', '症', '、'], 'start': 9, 'end': 18}, {'type': '禁忌症', 'word': ['重', '的', '心', '血', '管', '疾', '病', '#'], 'start': 20, 'end': 27}, {'type': '禁忌症', 'word': ['亢', '#', '禁', '忌', '症', '患'], 'start': 33, 'end': 38}])
            words,mark =M2D.toData(data)

            # print("".join(M2D.data2BMES(words,mark)))
        
            #返回标记后数据集
            return "".join(M2D.data2BMES(words,mark)),words,mark,data
                
            

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
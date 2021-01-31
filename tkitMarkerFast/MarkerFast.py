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
# from tkitJson import Config
import tkitJson
import BMESBIO2Data

class MarkerFast:
    """[自动从ner标注结果中提取数据]
    """

    def __init__(self, model_path="../model", device='cpu'):
        """[初始化自动标记系统]

        Args:
            model_path (str, optional): [模型地址]. Defaults to "../model".
            device (str, optional): [使用cpu黑色cuda]. Defaults to 'cpu'.
        """
        self.model_path = model_path
        self.labels_file = os.path.join(model_path, "labels.txt")
        self.device = device
        pass

    def __del__(self):
        # self.release()
        pass

    def release(self):
        """[释放模型]
        """
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
        """[加载模型]

        Returns:
            [type]: [返回model, tokenizer]
        """
        # tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        Config=tkitJson.Config(os.path.join(self.model_path,"config.json"))
        self.config=Config.read()
        # print(data.get("id2label"))
        
        # model.to(self.device)
        # f2 = open(self.labels_file, 'r')
        # lablels_dict = {}
        # for i, line in enumerate(f2):
        #     # l=line.split(" ")
        #     l = line.replace("\n", '')
        #     # print(l)
        #     lablels_dict[i] = l
        # f2.close()
        self.lablels_dict = self.config.get("id2label")
        # self.model=model
        # self.tokenizer=tokenizer
        # self.model.eval()
        return self.model, self.tokenizer
    # @profile
    def cut_sent(self,para):
        """[中文分句函数]

        Args:
            para ([type]): [句子段落]

        Returns:
            [type]: [句子列表]
        """
        para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
        para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
        para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
        para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
        # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
        para = para.rstrip()  # 段尾如果有多余的\n就去掉它
        # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
        return para.split("\n")
    def filterPunctuation(self, x):
        """[过滤中文标点]

        Args:
            x ([type]): [输入文本]

        Returns:
            [type]: [输出文本]
        """
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
            # print(len(torch.argmax(logits, axis=2).tolist()[0][1:-1]))
            # print(len(words))
            for i,(m,w) in enumerate( zip(torch.argmax(logits, axis=2).tolist()[0][1:-1],words)):
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

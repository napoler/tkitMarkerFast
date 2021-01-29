import tkitMarkerFast
import tkitJson
import re

import tqdm
# 初始化
model = tkitMarkerFast.MarkerFast()
# 加载模型
model.load_model()
# 【禁忌证】 [@顽固、难治性高血压#禁忌症*]、[@严重的心血管疾病#禁忌症*]及[@甲亢#禁忌症*]患者。


Tjson=tkitJson.Json("../1.json")
for i,item in tqdm.tqdm(enumerate( Tjson.load())):
    
    if len(item["title"])>20:
        # break
        continue
    else:
        with open('data/'+str(i)+item["title"]+'.txt','w') as f:    #设置文件对象
            # f.write(str) 
            # model.cut_sent(""item["data"])
            for it in item["data"]: 
                # print(it)
                if len(it)>2:
                    
                    p=model.pre(it)[0]
                    # print(p)
                    f.write(p+"\n") 
                    pass
                else:
                    f.write(it) 
                

                
                
        






# text = "【禁忌证】 顽固、难治性高血压#禁忌症、严重的心血管疾病#禁忌症及甲亢#禁忌症患者。"
# text,_,_,_=model.pre(text)
# print(text)

# # 【禁忌证】顽[@固、#禁忌症*]难[@治性高血压#禁忌症、#禁忌症*]严[@重的心血管疾病禁忌症*]禁忌症及甲[@亢#禁忌症患#禁忌症*]者。


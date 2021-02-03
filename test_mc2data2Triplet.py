import tkitMarkerFast
import tkitJson
import re
"""[用于将预测的数据转为三元格式]
"""
import tqdm
# 初始化
model = tkitMarkerFast.MarkerFast()
# 加载模型
model.load_model()
# 【禁忌证】 [@顽固、难治性高血压#禁忌症*]、[@严重的心血管疾病#禁忌症*]及[@甲亢#禁忌症*]患者。


Tjson=tkitJson.Json("../newData.json")
Newjson=tkitJson.Json("../newTriplet.json")
data=[]
for i,item in tqdm.tqdm(enumerate( Tjson.load())):

    # item["prediction"]=[]
    
    for it in item["prediction"]:
        for line in it["marked"]:
            # print (line)
            data.append({"name":item['title'],"zh":item['zh'],"en":item['en'],"type":line["type"],"word":"".join(line["word"])})
        pass
#         print(it)
#         if len(it)>2:
            
#             sent,words,mark,taged=model.pre(it)
#             # print({"sent":sent,'words':words,"marked":mark,"taged":data})
#             item["prediction"].append({"sent":sent,'words':words,"marked":mark,"taged":taged})
            
#             # f.write(p+"\n") 
#             pass
#         else:
#             # f.write(it) 
#             pass
#     data.append(item)
print(data)
Newjson.save(data)
            

                
                
        






# text = "【禁忌证】 顽固、难治性高血压#禁忌症、严重的心血管疾病#禁忌症及甲亢#禁忌症患者。"
# text,_,_,_=model.pre(text)
# print(text)

# # 【禁忌证】顽[@固、#禁忌症*]难[@治性高血压#禁忌症、#禁忌症*]严[@重的心血管疾病禁忌症*]禁忌症及甲[@亢#禁忌症患#禁忌症*]者。


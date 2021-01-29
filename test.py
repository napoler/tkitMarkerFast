import tkitMarkerFast

# 初始化
model = tkitMarkerFast.MarkerFast()
# 加载模型
model.load_model()
# 【禁忌证】 [@顽固、难治性高血压#禁忌症*]、[@严重的心血管疾病#禁忌症*]及[@甲亢#禁忌症*]患者。
text = "【禁忌证】 顽固、难治性高血压#禁忌症、严重的心血管疾病#禁忌症及甲亢#禁忌症患者。"
text,_,_,_=model.pre(text)
print(text)

# 【禁忌证】顽[@固、#禁忌症*]难[@治性高血压#禁忌症、#禁忌症*]严[@重的心血管疾病禁忌症*]禁忌症及甲[@亢#禁忌症患#禁忌症*]者。


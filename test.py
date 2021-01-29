import tkitMarkerFast
model = tkitMarkerFast.MarkerFast()
model.load_model()
# 【禁忌证】 [@顽固、难治性高血压#禁忌症*]、[@严重的心血管疾病#禁忌症*]及[@甲亢#禁忌症*]患者。
text = "【禁忌证】 顽固、难治性高血压#禁忌症、严重的心血管疾病#禁忌症及甲亢#禁忌症患者。"
text,_,_,_=model.pre(text)
print(text)


import tkitJson

import csv
Newjson=tkitJson.Json("../newTriplet.json")
data=[]

#python2可以用file替代open
with open("data.csv","w") as csvfile: 
    writer = csv.writer(csvfile)
 
    #先写入columns_name
    writer.writerow(["name","type","word"])
    for i,item in enumerate( Newjson.load()):
        #写入多行用writerows
        print(item)
        try:
            data.append([item['name'],item['type'],item['word']])
            pass
        except:
            pass
        
    
    writer.writerows(data)
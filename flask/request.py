import requests
url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'bp':0, 'sg':0, 'al':0, 'su':0, 'rbc':0, 'pc':0, 'pcc':0, 'ba':0, 'bgr':0,'bu':0,'sc':0, 'sod':0, 'pot':0, 'hemo':0, 'pcv':0, 'wc':0, 'rc':0, 'htn':0, 'dm':0, 'cad':0,'appet':0, 'pe':0, 'ane':0})
print(r.json())

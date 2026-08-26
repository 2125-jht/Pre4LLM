# coding = utf-8

import json

f = open("kai_kuiba_config.json")

data = json.load(f)

del data['loss_functions']['slide_ctr_6']
del data['loss_functions']['slide_l2r_6']


with open('kai_kuiba_config.json', 'w') as fw:
  fw.write(json.dumps(data, indent=2))




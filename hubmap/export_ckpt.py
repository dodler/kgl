import torch
import sys
from collections import OrderedDict

ckpt_path = sys.argv[1]
out_name = ckpt_path.split('/')[-1] + '.pth'
out_name = out_name.replace('=', '_')
print('using checkpoint', ckpt_path)
ckpt = torch.load(ckpt_path, map_location='cpu')['state_dict']

d = OrderedDict()
for k in ckpt.keys():
    d[k.replace('model.', '')] = ckpt[k]

print('saving checkpoint to', out_name)
torch.save(d, out_name)

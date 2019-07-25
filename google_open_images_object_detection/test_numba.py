import time

import pandas as pd
from lxml import etree
from numba import jit

attr_desc = pd.read_csv('/var/ssd_1t/open_images_obj_detection/meta/class-descriptions-boxable.csv', header=None)
attr_desc.columns = ['cls', 'desc']


# @jit
def test():
    t = etree.Element('test')
    t.text = str(attr_desc.iloc[0].desc)
    return etree.tostring(t, pretty_print=True)


test()

s = time.time()
for i in range(30000):
    test()

print(time.time() - s)

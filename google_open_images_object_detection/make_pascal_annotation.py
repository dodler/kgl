import pandas as pd
from PIL import Image
from lxml import etree
from numba import jit
from tqdm import *
from multiprocessing import Pool

attr_desc = pd.read_csv('/var/ssd_1t/open_images_obj_detection/meta/class-descriptions-boxable.csv', header=None)
attr_desc.columns = ['cls', 'desc']

boxes = pd.read_csv('/var/ssd_1t/open_images_obj_detection/meta/train-annotations-bbox.csv')
boxes['ImageIDINT'] = boxes.ImageID.apply(lambda x: int(x, 16))
box_unique_int_idx = boxes.ImageIDINT.unique()
box_unique_str_idx = boxes.ImageID.unique()

attr_desc = pd.read_csv('/var/ssd_1t/open_images_obj_detection/meta/class-descriptions-boxable.csv', header=None)
attr_desc.columns = ['cls', 'desc']

cls2name = {}
for i in range(attr_desc.shape[0]):
    cls2name[attr_desc.iloc[i, 0]] = attr_desc.iloc[i, 1]


@jit
def make_elm_with_text(tag, text):
    elm = etree.Element(tag)
    elm.text = text
    return elm


@jit
def make_default_source():
    root = etree.Element('source')

    db = make_elm_with_text('database', 'openimages')
    an = make_elm_with_text('annotation', 'OpenImages 2019')
    im = make_elm_with_text('image', 'flickr')

    root.append(db)
    root.append(an)
    root.append(im)

    return root


@jit
def make_size(width, height):
    result = etree.Element('size')
    result.append(make_elm_with_text('width', str(width)))
    result.append(make_elm_with_text('height', str(height)))
    result.append(make_elm_with_text('depth', '3'))

    return result


def make_single_object_elm(sobj, w, h):
    result = etree.Element('object')
    result.append(make_elm_with_text('name', cls2name[sobj.LabelName]))

    result.append(make_elm_with_text('pose', 'Unspecified'))
    result.append(make_elm_with_text('truncated', str(sobj.IsTruncated)))
    result.append(make_elm_with_text('difficult', '0'))

    box = etree.Element('bndbox')

    box.append(make_elm_with_text('xmin', str(sobj.XMin * w)))
    box.append(make_elm_with_text('xmax', str(sobj.XMax * w)))
    box.append(make_elm_with_text('ymin', str(sobj.YMin * h)))
    box.append(make_elm_with_text('ymax', str(sobj.YMax * h)))

    result.append(box)

    return result


def make_xml_string(obj):
    obj = obj.reset_index()

    root = etree.Element('annotation')
    f = etree.Element('folder')
    f.text = 'train'
    root.append(f)

    src = make_default_source()
    root.append(src)

    img = Image.open(f'/var/ssd_1t/open_images_obj_detection/train/{obj.ImageID.values[0]}.jpg')
    w = img.width
    h = img.height
    root.append(make_size(w, h))

    for i in range(obj.shape[0]):
        t = make_single_object_elm(obj.iloc[i], w, h)
        root.append(t)

    return etree.tostring(root, pretty_print=True)


make_elm_with_text('test', 'tt')
make_default_source()
make_size(10, 10)


def write_obj(sidx, obj_xml_str):
    with open('/home/lyan/Documents/output/' + sidx + '.xml', 'w') as f:
        f.write(obj_xml_str)


for i in tqdm(range(box_unique_int_idx.shape[0])):
    idx = box_unique_int_idx[i]
    sidx = box_unique_str_idx[i]
    s = make_xml_string(boxes[boxes.ImageIDINT == idx])
    s = str(s, 'utf-8')

    write_obj(sidx, s)

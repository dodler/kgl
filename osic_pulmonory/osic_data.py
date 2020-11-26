import os
import os.path as osp

import cv2
import numpy as np
import pandas as pd
import pydicom
import scipy.ndimage as ndimage
from scipy.stats import kurtosis
from scipy.stats import skew
from skimage import measure, morphology, segmentation


def get_img(path):
    d = pydicom.dcmread(path)
    img = cv2.resize((d.pixel_array - d.RescaleIntercept) / (d.RescaleSlope * 1000), (512, 512))
    return img


class OSICData:
    BAD_ID = ['ID00011637202177653955184', 'ID00052637202186188008618']

    def __init__(self, keys, a, tab, batch_size=32, aug=None):
        self.keys = [k for k in keys if k not in self.BAD_ID]
        self.a = a
        self.tab = tab
        self.batch_size = batch_size

        self.aug = aug

        self.train_images = []
        self.train_a = []
        self.train_tab = []

        train = pd.read_csv('/var/ssd_1t/kaggle_osic/train.csv')

        for p in keys.patient.unique():
            ldir = os.listdir(f'/var/ssd_1t/kaggle_osic/train/{p}/')
            numb = [float(i[:-4]) for i in ldir]
            images_p = [i for i in os.listdir(f'/var/ssd_1t/kaggle_osic/train/{p}/')
                        if int(i[:-4]) / len(ldir) < 0.8 and int(i[:-4]) / len(ldir) > 0.15]
            images_p = [osp.join(p, k) for k in images_p]
            self.train_images.extend(images_p)
            a_coef = a[p]
            tab_data = tab[p]

            self.train_a.extend([a_coef] * len(images_p))
            self.train_tab.extend([tab_data] * len(images_p))
        self.train_images = [k for k in self.train_images if k.endswith('.jpg')]
        print('osic data images len', len(self.train_images))

    def __len__(self):
        return len(self.train_images)

    def __getitem__(self, idx):
        a, tab = [], []

        img_path = '/var/ssd_1t/kaggle_osic/train/{}'.format(self.train_images[idx])
        img_path = img_path.replace('.dcm', '.jpg')
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = img.reshape(img.shape[0], img.shape[1], 1)

        img = self.aug(image=img)['image']

        a.append(self.train_a[idx])

        tab.append(self.train_tab[idx])
        a, tab = np.array(a), np.array(tab)
        return [img, tab], a


# aggregation of ct features and tabular data

def load_scan(path):
    """
    Loads scans from a folder and into a list.

    Parameters: path (Folder path)

    Returns: slices (List of slices)
    """
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: int(x.InstanceNumber))

    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        try:
            slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        except:
            try:
                slice_thickness = slices[0].SliceThickness
            except:
                slice_thickness = 0.5  # random value for now

    # print(slice_thickness)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices


def transform_to_hu(slices):
    """
    transform dicom.pixel_array to Hounsfield.
    Parameters: list dicoms
    Returns:numpy Hounsfield
    """

    images = np.stack([file.pixel_array for file in slices])
    images = images.astype(np.int16)

    # convert ouside pixel-values to air:
    # I'm using <= -1000 to be sure that other defaults are captured as well
    images[images <= -1000] = 0

    # convert to HU
    for n in range(len(slices)):

        intercept = slices[n].RescaleIntercept
        slope = slices[n].RescaleSlope

        if slope != 1:
            images[n] = slope * images[n].astype(np.float64)
            images[n] = images[n].astype(np.int16)

        images[n] += np.int16(intercept)

    return np.array(images, dtype=np.int16)


def generate_internal_mask(image):
    """
    Generates markers for a given image.
    Parameters: image
    Returns: Internal Marker, External Marker, Watershed Marker
    """

    # Creation of the internal Marker
    marker_internal = image < -400
    marker_internal = segmentation.clear_border(marker_internal)
    marker_internal_labels = measure.label(marker_internal)

    areas = [r.area for r in measure.regionprops(marker_internal_labels)]
    areas.sort()

    if len(areas) > 2:
        for region in measure.regionprops(marker_internal_labels):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                    marker_internal_labels[coordinates[0], coordinates[1]] = 0

    marker_internal = marker_internal_labels > 0

    return marker_internal


def generate_markers(image):
    """
    Generates markers for a given image.
    Parameters: image
    Returns: Internal Marker, External Marker, Watershed Marker
    """

    # Creation of the internal Marker
    marker_internal = image < -400
    marker_internal = segmentation.clear_border(marker_internal)
    marker_internal_labels = measure.label(marker_internal)

    areas = [r.area for r in measure.regionprops(marker_internal_labels)]
    areas.sort()

    if len(areas) > 2:
        for region in measure.regionprops(marker_internal_labels):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                    marker_internal_labels[coordinates[0], coordinates[1]] = 0

    marker_internal = marker_internal_labels > 0

    # Creation of the External Marker
    external_a = ndimage.binary_dilation(marker_internal, iterations=10)
    external_b = ndimage.binary_dilation(marker_internal, iterations=55)
    marker_external = external_b ^ external_a

    # Creation of the Watershed Marker
    marker_watershed = np.zeros((image.shape[0], image.shape[1]), dtype=np.int)
    marker_watershed += marker_internal * 255
    marker_watershed += marker_external * 128

    return marker_internal, marker_external, marker_watershed


def seperate_lungs_Watershed(image, iterations=1):
    """
    Segments lungs using various techniques.

    Parameters: image (Scan image), iterations (more iterations, more accurate mask)

    Returns:
        - Segmented Lung
        - Lung Filter
        - Outline Lung
        - Watershed Lung
        - Sobel Gradient
    """

    marker_internal, marker_external, marker_watershed = generate_markers(image)

    '''
    Creation of Sobel Gradient
    '''

    # Sobel-Gradient
    sobel_filtered_dx = ndimage.sobel(image, 1)
    sobel_filtered_dy = ndimage.sobel(image, 0)
    sobel_gradient = np.hypot(sobel_filtered_dx, sobel_filtered_dy)
    sobel_gradient *= 255.0 / np.max(sobel_gradient)

    '''
    Using the watershed algorithm


    We pass the image convoluted by sobel operator and the watershed marker
    to morphology.watershed and get a matrix matrix labeled using the 
    watershed segmentation algorithm.
    '''
    watershed = morphology.watershed(sobel_gradient, marker_watershed)

    return watershed


def crop_image(img: np.ndarray):
    edge_pixel_value = img[0, 0]
    mask = img != edge_pixel_value
    return img[np.ix_(mask.any(1), mask.any(0))]


def resize_image(img: np.ndarray, reshape=(512, 512)):
    img = [cv2.resize(im, (512, 512)) for im in img]
    return img


def preprocess_img(img, local_pd):
    # if local_pd.resize_type == 'resize':
    #    img = [resize_image(im) for im in img]
    if local_pd.resize_type == 'crop':
        img = [crop_image(im) for im in img]

    return np.array(img, dtype=np.int16)


def func_volume(patient_scan, patient_mask):
    pixel_spacing = patient_scan.PixelSpacing
    slice_thickness = patient_scan.SliceThickness
    slice_volume = np.count_nonzero(patient_mask) * pixel_spacing[0] * pixel_spacing[1] * slice_thickness
    return slice_volume


def caculate_lung_volume(patient_scans, patient_images):
    """
    caculate volume of lung from mask
    Parameters: list dicom scans,list patient CT image
    Returns: volume cm³　(float)
    """
    # patient_masks = pool.map(generate_internal_mask,patient_images)
    patient_masks = list(map(generate_internal_mask, patient_images))  # non-parallel version
    lung_volume = np.array(list(map(func_volume, patient_scans, patient_masks))).sum()

    return lung_volume * 0.001


def caculate_histgram_statistical(patient_images, thresh=[-500, -50]):
    """
    caculate hisgram kurthosis of lung hounsfield
    Parameters: list patient CT image 512*512,thresh divide lung
    Returns: histgram statistical characteristic(Mean,Skew,Kurthosis)
    """
    statistical_characteristic = dict(Mean=0, Skew=0, Kurthosis=0)
    num_slices = len(patient_images)
    patient_images = patient_images[int(num_slices * 0.1):int(num_slices * 0.9)]
    patient_images_mean = np.mean(patient_images, 0)

    s_pixel = patient_images_mean.flatten()
    s_pixel = s_pixel[np.where((s_pixel) > thresh[0] & (s_pixel < thresh[1]))]

    statistical_characteristic['Mean'] = np.mean(s_pixel)
    statistical_characteristic['Skew'] = skew(s_pixel)
    statistical_characteristic['Kurthosis'] = kurtosis(s_pixel)

    return statistical_characteristic

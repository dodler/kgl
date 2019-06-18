import cv2
from turbojpeg import TurboJPEG, TJPF_GRAY, TJSAMP_GRAY

jpeg = TurboJPEG('/usr/lib/x86_64-linux-gnu/libturbojpeg.so.0')


def read_turbo(path):
    with open(path, 'rb') as f:
        img = jpeg.decode(f.read())
    return img


def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized
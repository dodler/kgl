from imgaug import augmenters as iaa
import imgaug as ia


def get_augs(img_size=320):
    train_aug = iaa.Sequential([
        iaa.Fliplr(0.5),
        # iaa.Affine(
        #     scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},  # scale images to 80-120% of their size, individually per axis
        #     translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # translate by -20 to +20 percent (per axis)
        #     rotate=(-45, 45),  # rotate by -45 to +45 degrees
        #     shear=(-16, 16),  # shear by -16 to +16 degrees
        #     order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
        #     cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
        #     mode=ia.ALL),
        iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation
        iaa.Crop(percent=(0.05,0.5)),
        iaa.Resize(size=(img_size, img_size))
        # iaa.PadToFixedSize(img_size, img_size)
    ])

    valid_aug = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Resize(size=(img_size, img_size))
        # iaa.CropToFixedSize(img_size, img_size),
        # iaa.PadToFixedSize(img_size, img_size)
    ])

    return train_aug, valid_aug

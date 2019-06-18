from turbojpeg import TurboJPEG, TJPF_GRAY, TJSAMP_GRAY

jpeg = TurboJPEG('/usr/lib/x86_64-linux-gnu/libturbojpeg.so.0')


def read_turbo(path):
    with open(path, 'rb') as f:
        img = jpeg.decode(f.read())
    return img

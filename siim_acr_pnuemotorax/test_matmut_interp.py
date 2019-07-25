import torch


def downsample_mm_upsample(x, y):
    assert len(x.shape) == 3
    assert len(x.shape) == len(y.shape)
    x_ = torch.nn.functional.interpolate(x.unsqueeze(0), (4096, 8))
    y_ = torch.nn.functional.interpolate(y.unsqueeze(0), (8, 4096))

    z = torch.matmul(x_, y_)
    return torch.nn.functional.interpolate(z, (x.shape[1], y.shape[2])).squeeze(0)


if __name__=='__main__':
    x = torch.randn(1, 65536, 8, dtype=torch.float32)
    y = torch.randn(1, 8, 65536, dtype=torch.float32)

    print(downsample_mm_upsample(x,y).shape)

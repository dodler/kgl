import torch


def get_seg_head(in_channels=1792, out_channels=1):
    kernel_size = 3
    mid_channel = 128
    seg_head = torch.nn.Sequential(
        torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(mid_channel),
        torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(mid_channel),
        torch.nn.ConvTranspose2d(in_channels=mid_channel, out_channels=mid_channel, kernel_size=3, stride=4, padding=1,
                                 output_padding=1),

        torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(mid_channel),
        torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(mid_channel),
        torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(out_channels),
        torch.nn.AdaptiveAvgPool2d(output_size=32)
    )
    return seg_head


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    head = get_seg_head(24, out_channels=1)
    x = torch.randn(2, 24, 16, 16)
    y = head(x)
    print(y.shape)

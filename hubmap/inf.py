class Model_pred:
    def __init__(self, models, dl, checkpoints, tta: bool = True, half: bool = False, checkpoints):
        self.models = models
        self.checkpoints = checkpoints
        self.dl = dl
        self.tta = tta
        self.half = half

    def __iter__(self):
        count = 0
        for x in iter(self.dl):
            x = x.to(device)
            if self.half: x = x.half()
            py = None
            for model in self.models:
                p = model(x)
                p = torch.sigmoid(p)
                if py is None:
                    py = p
                else:
                    py += p
                if self.tta:
                    flips = [[-1], [-2], [-2, -1]]
                    for f in flips:
                        xf = torch.flip(x, f)
                        for model in self.models:
                            p = model(xf)
                            p = torch.flip(p, f)
                            py += torch.sigmoid(p).detach()
                    py /= (1 + len(flips))
            py /= len(self.models)

            py = F.upsample(py, scale_factor=reduce, mode="bilinear")
            py = py.permute(0, 2, 3, 1).float().cpu()
            batch_size = len(py)
            for i in range(batch_size):
                yield py[i]
                count += 1

    def __len__(self):
        return len(self.dl.dataset)
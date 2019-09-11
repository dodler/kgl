def weighted_bce(logit_pixel, gt):
    logit = logit_pixel.view(-1)
    truth = gt.view(-1)
    assert (logit.shape == truth.shape)

    loss = F.binary_cross_entropy_with_logits(logit, truth, reduction='none')
    if 0:
        loss = loss.mean()
    if 1:
        pos = (truth > 0.5).float()
        neg = (truth < 0.5).float()
        pos_weight = pos.sum().item() + 1e-12
        neg_weight = neg.sum().item() + 1e-12
        loss = (0.25 * pos * loss / pos_weight + 0.75 * neg * loss / neg_weight).sum()

    return loss


def lovasz_and_dice(pred, gt):
    return lovasz_hinge(pred, gt)


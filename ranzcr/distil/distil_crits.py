import torch
import torch.nn as nn

from ranzcr.distil.distil_utils import pdist
from utils import get_or_default


class HardDarkRank(nn.Module):
    def __init__(self, alpha=3, beta=3, permute_len=4):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.permute_len = permute_len

    def forward(self, student, teacher):
        score_teacher = -1 * self.alpha * pdist(teacher, squared=False).pow(self.beta)
        score_student = -1 * self.alpha * pdist(student, squared=False).pow(self.beta)

        permute_idx = score_teacher.sort(dim=1, descending=True)[1][:, 1:(self.permute_len + 1)]
        ordered_student = torch.gather(score_student, 1, permute_idx)

        log_prob = (ordered_student - torch.stack(
            [torch.logsumexp(ordered_student[:, i:], dim=1) for i in range(permute_idx.size(1))], dim=1)).sum(dim=1)
        loss = (-1 * log_prob).mean()

        return loss


class TopnMSELoss(nn.Module):
    def __init__(self, k):
        self.k = k
        self.mse = nn.MSELoss(reduction='sum')

    def forward(self, student, teacher):
        vals, idx = torch.topk(student, largest=False, k=self.k)

        return self.mse(
            student[:, idx],
            teacher[:, idx]
        )


def get_crit(trn_params):

    crit_name = get_or_default(trn_params, 'crit', 'HardDarkRank')

    if crit_name == 'HardDarkRank':
        dark_alpha = float(get_or_default(trn_params, 'dark_alpha', 2))
        dark_beta = float(get_or_default(trn_params, 'dark_alpha', 3))
        dark_crit = HardDarkRank(alpha=dark_alpha, beta=dark_beta)
        return dark_crit
    elif crit_name == 'TopnMSELoss':
        topk = float(get_or_default(trn_params, 'topk', 5))
        return TopnMSELoss(k=topk)
    else:
        raise Exception(' crit {} is not supported'.format(crit_name))

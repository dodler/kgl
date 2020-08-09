import torch

from isic_melanoma.starter_512.main import MelanomaModule
from isic_melanoma.starter_512.main import parser

args = parser.parse_args()

module = MelanomaModule(args)
module.load_from_checkpoint('/home/lyan/Documents/kaggle/isic_melanoma/lightning_logs/version_4/checkpoints/epoch=36'
                            '.ckpt',)
# module.load_state_dict(torch.load())
# module.load_from_checkpoint('/home/lyan/Documents/kaggle/isic_melanoma/lightning_logs/version_4'
#                             '/checkpoints/epoch=36.ckpt')
# module.load
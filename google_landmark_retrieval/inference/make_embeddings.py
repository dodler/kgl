from torch.utils.data import DataLoader
from tqdm import *
import faiss
import numpy as np
from google_landmark_retrieval.arguments import get_make_index_parser
from google_landmark_retrieval.aug import get_augs
from google_landmark_retrieval.constants import RETRIEVAL_DATA_PATH, IMG_SIZE
from google_landmark_retrieval.resnet_face import resnet_arcface

from google_landmark_retrieval.data import *


parser = get_make_index_parser()
args = parser.parse_args()

print('make embeddings using', args.csv_path)
target_df = pd.read_csv(f'%s{args.csv_path}' % RETRIEVAL_DATA_PATH)

device = 0
model = resnet_arcface().to(device)
state = torch.load(args.resume)
print(state.keys())
model.load_state_dict(state['model_0'], strict=False)
if "val_loss" in state.keys():
    print(f'restoring from state with loss {state["val_loss"]} and acc {state["val_acc"]}')

train_aug, valid_aug = get_augs(img_size=IMG_SIZE)


ds = LandmarkDataset('%s' % RETRIEVAL_DATA_PATH,
                     args.images_path, args.csv_path, valid_aug, test=True)

loader = DataLoader(ds, shuffle=False, num_workers=20, batch_size=args.batch_size)

outputs = []
model.eval()
for data in tqdm(loader):
    data_input, _ = data
    data_input = data_input.to(device)
    with torch.no_grad():
        output = model(data_input).detach().cpu().numpy()
    outputs.append(output)

outputs = np.concatenate(outputs)
np.save(f'{args.csv_path}.embeddings', outputs)
print('embeddings done')
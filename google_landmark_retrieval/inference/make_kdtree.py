from torch.utils.data import DataLoader
import numpy as np
from google_landmark_retrieval.arguments import get_make_index_parser
from google_landmark_retrieval.aug import get_augs
from google_landmark_retrieval.constants import RETRIEVAL_DATA_PATH, IMG_SIZE
from tqdm import *
# tqdm = tqdm_notebook
from google_landmark_retrieval.data import *
from google_landmark_retrieval.resnet_face import resnet_arcface
from sklearn.neighbors import KDTree

parser = get_make_index_parser()
args = parser.parse_args()

print('make tree using', args.csv_path)
target_df = pd.read_csv(f'%s{args.csv_path}' % RETRIEVAL_DATA_PATH)

device = 0
model = resnet_arcface().to(device)
state = torch.load(args.resume)
model.load_state_dict(state['model_0'])

if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)


train_aug, valid_aug = get_augs(img_size=IMG_SIZE)


ds = LandmarkDataset('%s' % RETRIEVAL_DATA_PATH,
                     args.images_path, args.csv_path, valid_aug, test=True)

loader = DataLoader(ds, shuffle=False, num_workers=args.num_workers, batch_size=args.batch_size)

outputs = []
model.eval()
for data in tqdm(loader):
    data_input, _ = data
    data_input = data_input.to(device)
    with torch.no_grad():
        output = model(data_input).detach().cpu().numpy()
        outputs.append(output)

outputs = np.concatenate(outputs)
tree = KDTree(outputs)

np.save(f'{args.csv_path}.embeddings', outputs)
np.save(f'{args.csv_path}.tree', tree)
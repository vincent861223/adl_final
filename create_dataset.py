from torch.utils.data import Dataset, DataLoader
import torch
import csv
from tqdm import tqdm
from scipy.stats import skew
import numpy as np

etc_feature = ['txn_floor', 'marriage_rate', 'land_area', 'town_population', 'born_rate', 'building_area', 'town_population_density', 'lat', 'parking_price', 'village_income_median', 'lon', 'doc_rate', 'master_rate', 'bachelor_rate', 'jobschool_rate']

cuda = True if torch.cuda.is_available() else False
device = 'cuda:0' if cuda else 'cpu'

class ContestDataset(Dataset):
	def __init__(self, trainData_path, train=True):
		self.trainData_path = trainData_path
		self.fields = []
		self.data = []
		tqdm.write('[*] Loading {} data from {}'.format('training' if train else 'testing', trainData_path))
		with open(trainData_path, 'r') as trainData_file:
			lines = csv.reader(trainData_file)
			lines_bar = tqdm(lines, desc='[-] Loading:', dynamic_ncols=True, leave=False)
			for i, line in enumerate(lines_bar):
				#print(line)
				if i == 0:
					self.fields = line
				else:
					idx = line[0]
					
					if train: 
						feature = [skew(float(t)) if t != '' else 0.0 for j, t in enumerate(line[1:-1]) if self.fields[j+1] in etc_feature]
						label = np.log1p(float(line[-1]))
					else:
						feature = [skew(float(t)) if t != '' else 0.0 for j, t in enumerate(line[1:]) if self.fields[j+1] in etc_feature] 
						label = 0.0
					self.data.append({'id': idx, 'feature': feature, 'label': label})
			print(self.fields)
			print(self.data[:5])
	def __getitem__(self, index):
		return self.data[index]

	def __len__(self):
		return len(self.data)

def collate_fn(batch):
	id = [b['id'] for b in batch]
	feature = torch.tensor([b['feature'] for b in batch])
	label = torch.tensor([b['label'] for b in batch]).unsqueeze(1)
	return {'id': id, 'feature': feature, 'label': label}

def create_dataloader(trainData_path, batch_size=16, train=True):
	dataset = ContestDataset(trainData_path, train=train)
	shuffle = True if train else False
	return DataLoader(dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=shuffle)

def test(trainData_path):
	dataloader = create_dataloader(trainData_path, shuffle=False)
	for i, data in enumerate(dataloader):
		if i == 0: 
			print(data['feature'])
			print(data['label'])
	return 


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('trainData_path', type=str, help='Path to training data')
	kwargs = vars(parser.parse_args())
	print(kwargs)
	test(**kwargs)

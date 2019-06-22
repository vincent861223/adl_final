import torch.nn as nn

class LinearModel(nn.Module):
	def __init__(self, input_size):
		super(LinearModel, self).__init__()
		self.input_size = input_size
		self.model = nn.Sequential(
			nn.BatchNorm1d(input_size, momentum=0.5),
			nn.Linear(input_size, input_size//2),
			nn.BatchNorm1d(input_size//2, momentum=0.5),
			#nn.ReLU(),

			nn.Linear(input_size//2, input_size//2),
			nn.BatchNorm1d(input_size//2, momentum=0.5),
			#nn.ReLU(),
			nn.Linear(input_size//2, input_size//4),
			nn.BatchNorm1d(input_size//4, momentum=0.5),
			#nn.ReLU(),
			nn.Linear(input_size//4, 1),
		)
		'''
		
		'''
	def forward(self, x):
		return self.model(x)


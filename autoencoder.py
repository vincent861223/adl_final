import torch.nn as nn

class AutoEncoder(nn.Module):
	def __init__(self, input_size):
		super(AutoEncoder, self).__init__()
		self.input_size = input_size
		self.encoder = nn.Sequential(
			nn.Linear(input_size, input_size//2),
			nn.Tanh(),
			nn.Linear(input_size//2, input_size//4),
			nn.Tanh(),
			nn.Linear(input_size//4, input_size//8),
			nn.Tanh(),
			nn.Linear(input_size//8, input_size//16),
			nn.Tanh(),
			nn.Linear(input_size//16, input_size//32),
		)
		self.decoder = nn.Sequential(
			nn.Linear(input_size//32, input_size//16),
			nn.Tanh(),
			nn.Linear(input_size//16, input_size//8),
			nn.Tanh(),
			nn.Linear(input_size//8, input_size//4),
			nn.Tanh(),
			nn.Linear(input_size//4, input_size//2),
			nn.Tanh(),
			nn.Linear(input_size//2, input_size),
		)
		'''
		
		'''
	def forward(self, x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		return encoded, decoded


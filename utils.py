import torch

def save_model(model, path):
	#print('[*] Save model to {}'.format(path))
	torch.save(model.state_dict(), path)

def load_model(model, path):
	print('[*] Load model from {}'.format(path))
	model.load_state_dict(torch.load(path))

def hit_rate(pred, label):
	with torch.no_grad():
		score = 0
		precision = (pred - label) / label
		for p in precision: 
			if abs(p) <= 0.1: score += 1
		return score / pred.size(0)

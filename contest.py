import argparse
import csv 
import os
from tqdm import tqdm
import torch
from tensorboardX import SummaryWriter

from create_dataset import create_dataloader
from model import LinearModel
#from autoencoder import AutoEncoder
from utils import *

def parse_arguments(parser):
	parser.add_argument('--train', action='store_true')
	parser.add_argument('--test', action='store_true')

	parser.add_argument('--train_data', type=str, default='../dataset/train.csv')
	parser.add_argument('--test_data', type=str, default='../dataset/test.csv')
	parser.add_argument('--test_output', type=str, default='../result.csv')
	parser.add_argument('--ckpt_dir', type=str, default='../checkpoints')
	parser.add_argument('--step', type=int, default=0)

	parser.add_argument('--n_epochs', type=int, default=100)
	parser.add_argument('--batch_size', type=int, default=16)
	parser.add_argument('--input_size', type=int, default=15)
	parser.add_argument('--save_fq', type=int, default=1000)

	parser.add_argument('--lr', type=float, default=1e-5)
	parser.add_argument('--b1', type=float, default=0.9)
	parser.add_argument('--b2', type=float, default=0.99)
	return parser.parse_args()

def train(args):
	os.makedirs(args.ckpt_dir, exist_ok=True)
	writer = SummaryWriter('log')
	dataloader = create_dataloader(args.train_data, args.batch_size, train=True)
	model = LinearModel(args.input_size)
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.b1, args.b2))
	criterion = torch.nn.MSELoss()

	tqdm.write('[*] Start training')
	step = 0
	epoch_bar = tqdm(range(args.n_epochs), desc='[Total Progress]', dynamic_ncols=True, leave=False, position=0)
	for epoch in epoch_bar:
		batch_bar = tqdm(dataloader, desc='[Train epoch {:2}]'.format(epoch), dynamic_ncols=True, leave=False, position=1)
		for i, data in enumerate(batch_bar):
			feature, label = data['feature'], data['label']
			optimizer.zero_grad()
			#encoded, decoded = model(feature)
			pred = model(feature)
			loss = criterion(pred, label)
			loss.backward()
			optimizer.step()

			hitRate = hit_rate(pred, label)
			#hitRate = 0
			if (step % 100 == 0):
				batch_bar.set_description('[Batch {}/{}] loss: {} hit_rate: {}'.format(i, len(dataloader), loss, hitRate))
				writer.add_scalar('Loss', loss, step)
			if (step % args.save_fq == 0):
				save_path = os.path.join(args.ckpt_dir, '{}.ckpt'.format(step))
				save_model(model, save_path)
			step += 1

	
	tqdm.write('[*] Finish training')
	return 

def test(args):
	dataloader = create_dataloader(args.test_data, args.batch_size, train=False)
	model = LinearModel(args.input_size)
	load_model(model, os.path.join(args.ckpt_dir, '{}.ckpt'.format(args.step)))
	output_csv = open(args.test_output, 'w')
	output_writer = csv.writer(output_csv)
	output_writer.writerow(['building_id', 'total_price'])

	tqdm.write('[*] Start testing')
	step = 0
	batch_bar = tqdm(dataloader, desc='[Testing]', dynamic_ncols=True, leave=False)
	for i, data in enumerate(batch_bar):
		id, feature, label = data['id'], data['feature'], data['label']
		pred = model(feature)
		batch_bar.set_description('[Testing] [Batch {}/{}]'.format(i, len(dataloader)))
		for i, p in enumerate(pred): output_writer.writerow([id[i], p.item()])
		#print(pred)
	tqdm.write('[*] Finish testing')
		
	

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	args = parse_arguments(parser)
	print(args)
	if args.train: 
		train(args)
	elif args.test:
		test(args)
	else:
		pass

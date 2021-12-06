import numpy as np
import torch
from tqdm import tqdm # for displaying progress bar
import os
from data import SLUDataset, ASRDataset
from models_test_v2 import PretrainedModel, Model
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau
class Trainer:
	def __init__(self, model, config):
		self.model = model
		self.config = config
		if isinstance(self.model, PretrainedModel):
			self.lr = config.pretraining_lr
			self.checkpoint_path = os.path.join(self.config.folder, "pretraining")
		else:
			self.lr = config.training_lr
			self.checkpoint_path = os.path.join(self.config.folder, "training")
		self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
		self.sheduler = ReduceLROnPlateau(self.optimizer,factor=0.5,threshold=0.02,patience=0,verbose=True)
		self.epoch = 0
		self.df = None

	def load_checkpoint(self):
		if os.path.isfile(os.path.join(self.checkpoint_path, "model_state.pth")):
			try:
				if self.model.is_cuda:
					self.model.load_state_dict(torch.load(os.path.join(self.checkpoint_path, "model_state.pth")))
				else:
					self.model.load_state_dict(torch.load(os.path.join(self.checkpoint_path, "model_state.pth"), map_location="cpu"))
			except:
				print("Could not load previous model; starting from scratch")
		else:
			print("No previous model; starting from scratch")

	def save_checkpoint(self,append_name=""):
		try:
			torch.save(self.model.state_dict(), os.path.join(self.checkpoint_path, append_name + "model_state.pth"))
		except:
			print("Could not save model")

	def log(self, results):
		if self.df is None:
			self.df = pd.DataFrame(columns=[field for field in results])
		self.df.loc[len(self.df)] = results
		self.df.to_csv(os.path.join(self.checkpoint_path, "log.csv"))

	def get_lr(self,optimizer):
		for param_group in optimizer.param_groups:
			return param_group['lr']

	def train(self, dataset, print_interval=100):

		if isinstance(dataset, ASRDataset):
			train_phone_acc = 0
			train_phone_loss = 0
			train_word_acc = 0
			train_word_loss = 0
			num_examples = 0
			self.model.train()
			for idx, batch in enumerate(tqdm(dataset.loader)):
				x,y_phoneme,y_word = batch
				batch_size = len(x)
				num_examples += batch_size
				phoneme_loss, word_loss, phoneme_acc, word_acc = self.model(x,y_phoneme,y_word)
				if self.config.pretraining_type == 1: loss = phoneme_loss
				if self.config.pretraining_type == 2: loss = phoneme_loss + word_loss
				if self.config.pretraining_type == 3: loss = word_loss
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()

				train_phone_loss += phoneme_loss.cpu().data.numpy().item() * batch_size
				train_word_loss += word_loss.cpu().data.numpy().item() * batch_size
				train_phone_acc += phoneme_acc.cpu().data.numpy().item() * batch_size
				train_word_acc += word_acc.cpu().data.numpy().item() * batch_size

				if idx % print_interval == 0:
					print("phoneme loss: " + str(phoneme_loss.cpu().data.numpy().item()))
					print("word loss: " + str(word_loss.cpu().data.numpy().item()))
					print("phoneme acc: " + str(phoneme_acc.cpu().data.numpy().item()))
					print("word acc: " + str(word_acc.cpu().data.numpy().item()))
			train_phone_loss /= num_examples
			train_phone_acc /= num_examples
			train_word_loss /= num_examples
			train_word_acc /= num_examples
			results = {"phone_loss" : train_phone_loss, "phone_acc" : train_phone_acc, "word_loss" : train_word_loss, "word_acc" : train_word_acc, "set": "train"}

			self.log(results)
			self.epoch += 1
			return train_phone_acc, train_phone_loss, train_word_acc, train_word_loss
		else: # SLUDataset
			train_intent_acc = 0
			train_intent_loss = 0
			num_examples = 0
			self.model.train()
			my_lr = self.get_lr(self.optimizer)
			print("current lr ", my_lr)
			#self.model.print_frozen()
			for idx, batch in enumerate(tqdm(dataset.loader)):
				x,y_intent = batch
				batch_size = x.shape[0] #len(x)
				num_examples += batch_size
				intent_loss, intent_acc = self.model(x,y_intent)
			### optimise back backprop
				for param in self.model.parameters(): # insted of self.optimizer.zero_grad()
					param.grad = None
				intent_loss.backward()
#				torch.nn.utils.clip_grad_norm_(self.model.parameters(), 3) # gradient cliping 

				self.optimizer.step()
				train_intent_loss += float(intent_loss)* batch_size
				train_intent_acc += float(intent_acc)* batch_size
				if idx % print_interval == 0:
					#exit()
					print("intent loss: " + str(float(intent_loss)))
					print("intent acc: " + str(float(intent_acc)))

				del intent_loss , intent_acc , param
			train_intent_loss /= num_examples
			train_intent_acc /= num_examples
			self.model.unfreeze_l2()
			self.model.unfreeze_l2()
			self.model.unfreeze_l2()

			results = {"intent_loss" : train_intent_loss, "intent_acc" : train_intent_acc, "set": "train"}
			self.log(results)
			self.epoch += 1
			return train_intent_acc, train_intent_loss

	def test(self, dataset):
		if isinstance(dataset, ASRDataset):
			test_phone_acc = 0
			test_phone_loss = 0
			test_word_acc = 0
			test_word_loss = 0
			num_examples = 0
			self.model.eval()
			for idx, batch in enumerate(dataset.loader):
				x,y_phoneme,y_word = batch
				batch_size = len(x)
				num_examples += batch_size
				phoneme_loss, word_loss, phoneme_acc, word_acc = self.model(x,y_phoneme,y_word)
				test_phone_loss += phoneme_loss.cpu().data.numpy().item() * batch_size
				test_word_loss += word_loss.cpu().data.numpy().item() * batch_size
				test_phone_acc += phoneme_acc.cpu().data.numpy().item() * batch_size
				test_word_acc += word_acc.cpu().data.numpy().item() * batch_size

			test_phone_loss /= num_examples
			test_phone_acc /= num_examples
			test_word_loss /= num_examples
			test_word_acc /= num_examples
			if self.config.pretraining_type == 1: test_loss = test_phone_loss
			if self.config.pretraining_type == 2: test_loss = test_phone_loss + test_word_loss
			if self.config.pretraining_type == 3: test_loss = test_word_loss
#
			self.sheduler.step(test_loss)
			results = {"phone_loss" : test_phone_loss, "phone_acc" : test_phone_acc, "word_loss" : test_word_loss, "word_acc" : test_word_acc,"set": "valid"}
			self.log(results)
			return test_phone_acc, test_phone_loss, test_word_acc, test_word_loss 
		else:
			test_intent_acc = 0
			test_intent_loss = 0
			num_examples = 0
			self.model.eval()

			for idx, batch in enumerate(dataset.loader):
				x,y_intent = batch
				batch_size = len(x)
				num_examples += batch_size
				intent_loss, intent_acc = self.model(x,y_intent)
				test_intent_loss += intent_loss.cpu().data.numpy().item() * batch_size
				test_intent_acc += intent_acc.cpu().data.numpy().item() * batch_size

			self.model.cuda(); self.model.is_cuda = True
			test_intent_loss /= num_examples
			test_intent_acc /= num_examples
			self.sheduler.step(test_intent_loss)
			results = {"intent_loss" : test_intent_loss, "intent_acc" : test_intent_acc, "set": "valid"}
			self.log(results)
			return test_intent_acc, test_intent_loss 

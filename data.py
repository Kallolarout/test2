import torch
import torch.utils.data
import torchaudio
import os, glob
from collections import Counter
import soundfile as sf
import numpy as np
import configparser
import textgrid
import multiprocessing
import json
import pandas as pd
from subprocess import call

class Config:
	def __init__(self):
		self.use_sincnet = True

def read_config(config_file):
	config = Config()
	parser = configparser.ConfigParser()
	parser.read(config_file)

	#[experiment]
	config.seed=int(parser.get("experiment", "seed"))
	config.folder=parser.get("experiment", "folder")
	
	# Make a folder containing experiment information
	if not os.path.isdir(config.folder):
		os.mkdir(config.folder)
		os.mkdir(os.path.join(config.folder, "pretraining"))
		os.mkdir(os.path.join(config.folder, "training"))
	call("cp " + config_file + " " + os.path.join(config.folder, "experiment.cfg"), shell=True)


	#[pretraining]
	config.asr_path=parser.get("pretraining", "asr_path")
	config.pretraining_type=int(parser.get("pretraining", "pretraining_type")) # 2 - phoneme + word pre-training
	if config.pretraining_type == 2: config.starting_unfreezing_index = 1

	config.pretraining_lr=float(parser.get("pretraining", "pretraining_lr"))
	config.pretraining_batch_size=int(parser.get("pretraining", "pretraining_batch_size"))
	config.pretraining_num_epochs=int(parser.get("pretraining", "pretraining_num_epochs"))
	config.pretraining_length_mean=float(parser.get("pretraining", "pretraining_length_mean"))
	config.pretraining_length_var=float(parser.get("pretraining", "pretraining_length_var"))

	#[training]
	config.slu_path=parser.get("training", "slu_path")
	config.unfreezing_type=int(parser.get("training", "unfreezing_type"))
	config.training_lr=float(parser.get("training", "training_lr"))
	config.training_batch_size=int(parser.get("training", "training_batch_size"))
	config.training_num_epochs=int(parser.get("training", "training_num_epochs"))
	config.real_dataset_subset_percentage=float(parser.get("training", "real_dataset_subset_percentage"))
	config.synthetic_dataset_subset_percentage=float(parser.get("training", "synthetic_dataset_subset_percentage"))
	config.real_speaker_subset_percentage=float(parser.get("training", "real_speaker_subset_percentage"))
	config.synthetic_speaker_subset_percentage=float(parser.get("training", "synthetic_speaker_subset_percentage"))
	config.train_wording_path=parser.get("training", "train_wording_path")
	if config.train_wording_path=="None": config.train_wording_path = None
	config.test_wording_path=parser.get("training", "test_wording_path")
	if config.test_wording_path=="None": config.test_wording_path = None
	try:
		config.augment = (parser.get("training", "augment")  == "True")
	except:
		#  file with no augmentation
		config.augment = False

	try:
		config.seq2seq = (parser.get("training", "seq2seq")  == "True")
	except:
		# \ no seq2seq
		config.seq2seq = False

	try:
		config.dataset_upsample_factor = int(parser.get("training", "dataset_upsample_factor"))
	except:
		# old config file
		config.dataset_upsample_factor = 1


	return config

def get_SLU_datasets(config):
	"""
	config: Config object (contains info about model and training)
	"""
	base_path = config.slu_path

	# loading csvs
	train_df = pd.read_csv(os.path.join(base_path, "data","train.csv"))
	valid_df = pd.read_csv(os.path.join(base_path, "data","valid.csv"))
	test_df = pd.read_csv(os.path.join(base_path, "data","test.csv"))
	if "\"Unnamed: 0\"" in list(train_df): train_df = train_df.drop(columns="Unnamed: 0")


	if not config.seq2seq:
		# Get list of slots
		Sy_intent = {"action": {}, "object": {}} #, "location": {}} 

		values_per_slot = []
		for slot in ["action", "object"]: # "location"]:
			slot_values = Counter(train_df[slot])
			for idx,value in enumerate(slot_values):
				Sy_intent[slot][value] = idx
				#print(Sy_intent)
			values_per_slot.append(len(slot_values))
		config.values_per_slot = values_per_slot
		
		config.Sy_intent = Sy_intent

	else: #seq2seq
		import string
		all_chars = "".join(train_df.loc[i]["semantics"] for i in range(len(train_df))) + string.printable # all printable chars; TODO: unicode?
		all_chars = list(set(all_chars))
		Sy_intent = ["<sos>"]
		Sy_intent += all_chars
		Sy_intent.append("<eos>")
		config.Sy_intent = Sy_intent

	# If certain phrases are specified, only use those phrases
	if config.train_wording_path is not None:
		with open(config.train_wording_path, "r") as f:
			train_wordings = [line.strip() for line in f.readlines()]
		train_df = train_df.loc[train_df.transcription.isin(train_wordings)]
		train_df = train_df.set_index(np.arange(len(train_df)))

	if config.test_wording_path is not None:
		with open(config.test_wording_path, "r") as f:
			test_wordings = [line.strip() for line in f.readlines()]
		valid_df = valid_df.loc[valid_df.transcription.isin(test_wordings)]
		valid_df = valid_df.set_index(np.arange(len(valid_df)))
		test_df = test_df.loc[test_df.transcription.isin(test_wordings)]
		test_df = test_df.set_index(np.arange(len(test_df)))

	# Get number of phonemes
	if os.path.isfile(os.path.join(config.folder, "pretraining", "phonemes.txt")):
		Sy_phoneme = []
		with open(os.path.join(config.folder, "pretraining", "phonemes.txt"), "r") as f:
			for line in f.readlines():
				if line.rstrip("\n") != "": Sy_phoneme.append(line.rstrip("\n"))
		config.num_phonemes = len(Sy_phoneme)
	else:
		print("No phoneme file found.")

	# Create dataset objects
	train_dataset = SLUDataset(train_df, base_path, Sy_intent, config)
	valid_dataset = SLUDataset(valid_df, base_path, Sy_intent, config)
	test_dataset = SLUDataset(test_df, base_path, Sy_intent, config)

	return train_dataset, valid_dataset, test_dataset


def rms_energy(x):
	return 10*np.log10((1e-12 + x.dot(x))/len(x))

class SLUDataset(torch.utils.data.Dataset):
	def __init__(self, df, base_path, Sy_intent, config="", upsample_factor=1):
		"""
		df:
		Sy_intent: Dictionary (transcript --> slot values)
		config: Config object (contains info about model and training)
		"""
		self.df = df
		self.base_path = base_path
		self.Sy_intent = Sy_intent
		self.Sy_intent_df = pd.DataFrame.from_dict(Sy_intent, orient ='index') 
		self.upsample_factor = upsample_factor
		self.batch_size = config.training_batch_size

		self.loader = torch.utils.data.DataLoader(self, batch_size=self.batch_size,shuffle=True,drop_last=True, num_workers=6,pin_memory =True, prefetch_factor=60)


	def __len__(self):

		return self.df.shape[0]

	def __getitem__(self, idx):
		wav_path = self.df.loc[idx].path

		x  = torchaudio.load(wav_path)[0].squeeze(0)
		T = 32000 
		if T >  x.shape[0]:
			x_pad_length = T - x.shape[0] # len(x)
			x = torch.nn.functional.pad(x,(x_pad_length,0))
		y_intent =np.zeros(2)  #
		for ind ,slot in enumerate(["action", "object"]):
			value = self.df.loc[idx][slot]
			y_intent[ind] = self.Sy_intent_df.iloc[ind][value]
		return (x, y_intent.astype(np.long))

def one_hot(letters, S):
	"""
	letters : LongTensor of shape (batch size, sequence length)
	S : integer
	Convert batch of integer letter indices to one-hot vectors of dimension S (# of possible letters).
	"""

	out = torch.zeros(letters.shape[0], letters.shape[1], S)
	for i in range(0, letters.shape[0]):
		for t in range(0, letters.shape[1]):
			out[i, t, letters[i,t]] = 1
	return out

class CollateWavsSLU:
	def __init__(self, Sy_intent, seq2seq):

		self.seq2seq = seq2seq
		if self.seq2seq:
			self.EOS = self.Sy_intent.index("<eos>")

	def __call__(self, batch):
		"""
		batch: list of tuples (input wav, intent labels)

		Returns a minibatch of wavs and labels as Tensors.
		"""
		x = []; y_intent = []
		batch_size = len(batch)
		for index in range(batch_size):
			x_,y_intent_ = batch[index]

			x.append(torch.tensor(x_).float())
			y_intent.append(torch.tensor(y_intent_).long())

		# pad all sequences to have same length
		if not self.seq2seq:
			T = 32000 #max([len(x_) for x_ in x])
			for index in range(batch_size):
				x_pad_length = (T - len(x[index]))
				x[index] = torch.nn.functional.pad(x[index], (0,x_pad_length))

			x = torch.stack(x)
			y_intent = torch.stack(y_intent)


			return (x,y_intent)


def get_ASR_datasets(config):
	"""
		Assumes that the data directory contains the following two directories:
			"audio" : wav files (split into train-clean, train-other, ...)
			"text" : alignments for each wav

	config: Config object (contains info about model and training)
	"""
	base_path = config.asr_path

	# Get only files with a label

	train_textgrid_paths = glob.glob(base_path + "/train-clean-100/*/*/*.TextGrid")
	train_wav_paths = [path.replace(".TextGrid", ".wav") for path in train_textgrid_paths]
	test_textgrid_paths = glob.glob(base_path + "/test*/*/*/*.TextGrid")
	test_wav_paths = [path.replace(".TextGrid", ".wav") for path in test_textgrid_paths]
	vaild_textgrid_paths = glob.glob(base_path + "/valid*/*/*/*.TextGrid")
	valid_wav_paths = [path.replace(".TextGrid", ".wav") for path in valid_textgrid_paths]
	# Get list of phonemes and words
	if os.path.isfile(os.path.join(config.folder, "pretraining", "phonemes.txt")) and os.path.isfile(os.path.join(config.folder, "pretraining", "words.txt")):
		Sy_phoneme = []
		with open(os.path.join(config.folder, "pretraining", "phonemes.txt"), "r") as f:
			for line in f.readlines():
				if line.rstrip("\n") != "": Sy_phoneme.append(line.rstrip("\n"))
		config.num_phonemes = len(Sy_phoneme)

		Sy_word = []
		with open(os.path.join(config.folder, "pretraining", "words.txt"), "r") as f:
			for line in f.readlines():
				Sy_word.append(line.rstrip("\n"))

	else:
		print("Getting vocabulary...")
		phoneme_counter = Counter()
		word_counter = Counter()
		for path in valid_textgrid_paths:
			tg = textgrid.TextGrid()
			tg.read(path)
			phoneme_counter.update([phone.mark.rstrip("0123456789") for phone in tg.getList("phones")[0] if phone.mark != ''])
			word_counter.update([word.mark for word in tg.getList("words")[0]]) #if word.mark != ''])

		Sy_phoneme = list(phoneme_counter)
		Sy_word = [w[0] for w in word_counter.most_common(config.vocabulary_size)]
		config.num_phonemes = len(Sy_phoneme)
		with open(os.path.join(config.folder, "pretraining", "phonemes.txt"), "w") as f:
			for phoneme in Sy_phoneme:
				f.write(phoneme + "\n")

		with open(os.path.join(config.folder, "pretraining", "words.txt"), "w") as f:
			for word in Sy_word:
				f.write(word + "\n")

	print("Done.")

	# Create dataset objects
	train_dataset = ASRDataset(train_wav_paths, train_textgrid_paths, Sy_phoneme, Sy_word, config)#
	valid_dataset =  ASRDataset(valid_wav_paths, valid_textgrid_paths, Sy_phoneme, Sy_word, config)
	test_dataset =  ASRDataset(test_wav_paths, test_textgrid_paths, Sy_phoneme, Sy_word, config)

	return train_dataset ,valid_dataset, test_dataset

class ASRDataset(torch.utils.data.Dataset):
	def __init__(self, wav_paths, textgrid_paths, Sy_phoneme, Sy_word, config):
		"""
		wav_paths: list of strings (wav file paths)
		textgrid_paths: list of strings (textgrid for each wav file)
		Sy_phoneme: list of strings (all possible phonemes)
		Sy_word: list of strings (all possible words)
		config: Config object (contains info about model and training)
		"""
		self.wav_paths = wav_paths # list of wav file paths
		self.textgrid_paths = textgrid_paths # list of textgrid file paths
		self.length_mean = config.pretraining_length_mean
		self.length_var = config.pretraining_length_var
		self.Sy_phoneme = Sy_phoneme
		self.Sy_word = Sy_word
		self.phone_downsample_factor = 1
		self.word_downsample_factor = 1		
		self.loader = torch.utils.data.DataLoader(self, batch_size=config.pretraining_batch_size,shuffle=True, collate_fn=CollateWavsASR())

	def __len__(self):
		return len(self.wav_paths)

	def __getitem__(self, idx):

		x, fs = sf.read(self.wav_paths[idx])

		tg = textgrid.TextGrid()

		tg.read(self.textgrid_paths[idx])

		y_phoneme = []
		for phoneme in tg.getList("phones")[0]:
			duration = phoneme.maxTime - phoneme.minTime
			phoneme_index = self.Sy_phoneme.index(phoneme.mark.rstrip("0123456789")) if phoneme.mark.rstrip("0123456789") in self.Sy_phoneme else -1
			if phoneme.mark == '': phoneme_index = -1
			y_phoneme += [phoneme_index] * round(duration * fs)

		y_word = []
		for word in tg.getList("words")[0]:
			duration = word.maxTime - word.minTime
			word_index = self.Sy_word.index(word.mark) if word.mark in self.Sy_word else -1
			# if word.mark == '': word_index = -1
			y_word += [word_index] * round(duration * fs)

		# Cut a snippet of length random_length from the audio
		random_length = round(fs * max(self.length_mean + self.length_var * torch.randn(1).item(), 0.5))
		if len(x) <= random_length:
			start = 0
		else:
			start = torch.randint(low=0, high=len(x)-random_length, size=(1,)).item()
		end = start + random_length

		x = x[start:end]
		y_phoneme = y_phoneme[start:end:self.phone_downsample_factor]
		y_word = y_word[start:end:self.word_downsample_factor]

		return (x, y_phoneme, y_word)

class CollateWavsASR:
	def __call__(self, batch):
		"""
		batch: list of tuples (input wav, phoneme labels, word labels)

		Returns a minibatch of wavs and labels as Tensors.
		"""
		x = []; y_phoneme = []; y_word = []
		batch_size = len(batch)
		for index in range(batch_size):
			x_,y_phoneme_, y_word_ = batch[index]

			x.append(torch.tensor(x_).float())
			y_phoneme.append(torch.tensor(y_phoneme_).long())
			y_word.append(torch.tensor(y_word_).long())

		# pad all sequences to have same length
		T = max([len(x_) for x_ in x])
		U_phoneme = max([len(y_phoneme_) for y_phoneme_ in y_phoneme])
		U_word = max([len(y_word_) for y_word_ in y_word])
		for index in range(batch_size):
			x_pad_length = (T - len(x[index]))
			x[index] = torch.nn.functional.pad(x[index], (0,x_pad_length))

			y_pad_length = (U_phoneme - len(y_phoneme[index]))
			y_phoneme[index] = torch.nn.functional.pad(y_phoneme[index], (0,y_pad_length), value=-1)
			
			y_pad_length = (U_word - len(y_word[index]))
			y_word[index] = torch.nn.functional.pad(y_word[index], (0,y_pad_length), value=-1)

		x = torch.stack(x)
		y_phoneme = torch.stack(y_phoneme)
		y_word = torch.stack(y_word)

		return (x,y_phoneme, y_word)

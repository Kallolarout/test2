import torch
import numpy as np
import sys
import os
import math
def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.contiguous()
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, 
        -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)


def sinc(band,t_right):
    y_right= torch.sin(2*math.pi*band*t_right)/(2*math.pi*band*t_right)
    y_left= flip(y_right,0)

    if t_right.is_cuda: y=torch.cat([y_left,(torch.ones(1)).cuda(),y_right])
    else: y=torch.cat([y_left,(torch.ones(1)),y_right])

    return y

class Downsample(torch.nn.Module):
    """
    Downsamples the input in the time/sequence domain
    """
    def __init__(self, method="none", factor=1, axis=1):
        super(Downsample,self).__init__()
        self.factor = factor
        self.method = method
        self.axis = axis
        methods = ["none", "avg", "max"]
        if self.method not in methods:
            #print("Error: downsampling method must be one of the following: \"none\", \"avg\", \"max\"")
            sys.exit()
            
    def forward(self, x):
        if self.method == "none":
            return x.transpose(self.axis, 0)[::self.factor].transpose(self.axis, 0)
        if self.method == "avg":
            return torch.nn.functional.avg_pool1d(x.transpose(self.axis, 2), kernel_size=self.factor, ceil_mode=True).transpose(self.axis, 2)
        if self.method == "max":
            return torch.nn.functional.max_pool1d(x.transpose(self.axis, 2), kernel_size=self.factor, ceil_mode=True).transpose(self.axis, 2)
class SincConv_fast(torch.nn.Module):
    """Sinc-based convolution
    Parameters
    ----------
    in_channels : `int`
        Number of input channels. Must be 1.
    out_channels : `int`
        Number of filters.
    kernel_size : `int`
        Filter length.
    sample_rate : `int`, optional
        Sample rate. Defaults to 16000.

    """

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, out_channels, kernel_size, sample_rate=16000, in_channels=1,
                 stride=1, padding=0, dilation=1, bias=False, groups=1, min_low_hz=50, min_band_hz=50):

        super(SincConv_fast,self).__init__()

        if in_channels != 1:
            #msg = (f'SincConv only support one input channel '
            #       f'(here, in_channels = {in_channels:d}).')
            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size%2==0:
            self.kernel_size=self.kernel_size+1
            
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')

        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # initialize filterbanks such that they are equally spaced in Mel scale
        low_hz = 30
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        mel = np.linspace(self.to_mel(low_hz),
                          self.to_mel(high_hz),
                          self.out_channels + 1)
        hz = self.to_hz(mel)
        

        # filter lower frequency (out_channels, 1)
        self.low_hz_ = torch.nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))

        # filter frequency band (out_channels, 1)
        self.band_hz_ = torch.nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))

        # Hamming window
        #self.window_ = torch.hamming_window(self.kernel_size)
        n_lin=torch.linspace(0, (self.kernel_size/2)-1, steps=int((self.kernel_size/2))) # computing only half of the window
        self.window_=0.54-0.46*torch.cos(2*math.pi*n_lin/self.kernel_size);


        # (1, kernel_size/2)
        n = (self.kernel_size - 1) / 2.0
        self.n_ = 2*math.pi*torch.arange(-n, 0).view(1, -1) / self.sample_rate # Due to symmetry, I only need half of the time axes

 


    def forward(self, waveforms):
        """
        Parameters
        ----------
        waveforms : `torch.Tensor` (batch_size, 1, n_samples)
            Batch of waveforms.
        Returns
        -------
        features : `torch.Tensor` (batch_size, out_channels, n_samples_out)
            Batch of sinc filters activations.
        """

        self.n_ = self.n_.to(waveforms.device)

        self.window_ = self.window_.to(waveforms.device)

        low = self.min_low_hz  + torch.abs(self.low_hz_)
        
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_),self.min_low_hz,self.sample_rate/2)
        band=(high-low)[:,0]
        
        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        band_pass_left=((torch.sin(f_times_t_high)-torch.sin(f_times_t_low))/(self.n_/2))*self.window_ # Equivalent of Eq.4 of the reference paper (SPEAKER RECOGNITION FROM RAW WAVEFORM WITH SINCNET). I just have expanded the sinc and simplified the terms. This way I avoid several useless computations. 
        band_pass_center = 2*band.view(-1,1)
        band_pass_right= torch.flip(band_pass_left,dims=[1])
        
        
        band_pass=torch.cat([band_pass_left,band_pass_center,band_pass_right],dim=1)

        
        band_pass = band_pass / (2*band[:,None])
        

        self.filters = (band_pass).view(
            self.out_channels, 1, self.kernel_size)

        return torch.nn.functional.conv1d(waveforms, self.filters, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                         bias=None, groups=1)            
class SincLayer(torch.nn.Module):
    """
    Modified from https://github.self.com/mravanelli/SincNet/blob/master/dnn_models.py:sinc_conv
    """
    def __init__(self, N_filt,Filt_dim,fs, stride=1, padding=0, is_cuda=False):
        super(SincLayer,self).__init__()

        # Mel Initialization of the filterbanks
        low_freq_mel = 80
        high_freq_mel = (2595 * np.log10(1 + (fs / 2) / 700))  # Convert Hz to Mel
        mel_points = np.linspace(low_freq_mel, high_freq_mel, N_filt)  # Equally spaced in Mel scale
        f_cos = (700 * (10**(mel_points / 2595) - 1)) # Convert Mel to Hz
        b1=np.roll(f_cos,1)
        b2=np.roll(f_cos,-1)
        b1[0]=30
        b2[-1]=(fs/2)-100

        self.freq_scale=fs*1.0
        self.filt_b1 = torch.nn.Parameter(torch.from_numpy(b1/self.freq_scale))
        self.filt_band = torch.nn.Parameter(torch.from_numpy((b2-b1)/self.freq_scale))
        
        self.min_freq = torch.nn.Parameter(torch.tensor([50.0]))
        self.min_band = torch.nn.Parameter(torch.tensor([50.0]))
        
        self.N_filt=N_filt
        self.Filt_dim=Filt_dim
        self.fs=fs
        self.stride=stride
        self.padding=padding
        self.is_cuda = is_cuda

    def forward(self, x):
        self.is_cuda = next(self.parameters()).is_cuda
        filters=torch.zeros((self.N_filt,self.Filt_dim)) #.cuda()
        if self.is_cuda: filters = filters.cuda()
        N=self.Filt_dim
        t_right=(torch.linspace(1, (N-1)/2, steps=int((N-1)/2))/self.fs) #.cuda()
        if self.is_cuda: t_right = t_right.cuda()

        min_freq=self.min_freq#50.0;
        min_band=self.min_band#50.0;

        filt_beg_freq=torch.abs(self.filt_b1)+min_freq/self.freq_scale
        filt_end_freq=filt_beg_freq+(torch.abs(self.filt_band)+min_band/self.freq_scale)

        n=torch.linspace(0, N, steps=N)

        # Filter window (hamming)
        window=0.54-0.46*torch.cos(2*math.pi*n/N);
        window=window.float() #.cuda()
        if self.is_cuda: window = window.cuda()

        for i in range(self.N_filt):
            low_pass1 = 2*filt_beg_freq[i].float()*sinc(filt_beg_freq[i].float()*self.freq_scale,t_right)
            low_pass2 = 2*filt_end_freq[i].float()*sinc(filt_end_freq[i].float()*self.freq_scale,t_right)
            band_pass=(low_pass2-low_pass1)

            band_pass=band_pass/torch.max(band_pass)
            if self.is_cuda: band_pass = band_pass.cuda()

            filters[i,:]=band_pass*window

            out=torch.nn.functional.conv1d(x, filters.view(self.N_filt,1,self.Filt_dim), stride=self.stride, padding=self.padding)# 0)#

        return out


class FinalPool(torch.nn.Module):
    def __init__(self):
        super(FinalPool, self).__init__()

    def forward(self, input):
        """
        input : Tensor of shape (batch size, T, Cin)
        
        Outputs a Tensor of shape (batch size, Cin).
        """

        return input.max(dim=1)[0]

class NCL2NLC(torch.nn.Module):
    def __init__(self):
        super(NCL2NLC, self).__init__()

    def forward(self, input):
        """
        input : Tensor of shape (batch size, T, Cin)
        
        Outputs a Tensor of shape (batch size, Cin, T).
        """

        return input.transpose(1,2)

class RNNSelect(torch.nn.Module):
    def __init__(self):
        super(RNNSelect, self).__init__()

    def forward(self, input):
        """
        input : tuple of stuff
        
        Outputs a Tensor of shape 
        """

        return input[0] 

class LayerNorm(torch.nn.Module):
    def __init__(self, dim,index=1, eps=1e-6):
        super(LayerNorm,self).__init__()
        self.gamma = torch.nn.Parameter(torch.ones(dim))
        self.beta = torch.nn.Parameter(torch.zeros(dim))
        self.eps = eps
        self.index = index

    def forward(self, x):
        mean = x.mean(self.index, keepdim=True)
        std = x.std(self.index, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class Abs(torch.nn.Module):
    def __init__(self):
        super(Abs, self).__init__()

    def forward(self, input):
        return torch.abs(input) 

class PretrainedModel(torch.nn.Module):
    """
    Model pre-trained to recognize phonemes and words.
    """
    def __init__(self, config):
        super(PretrainedModel, self).__init__()
        self.phoneme_layers = []
        self.word_layers = []
        self.is_cuda = torch.cuda.is_available()
        

        # CNN
        num_conv_layers = 3

        self.Sinc_layer = SincConv_fast(out_channels=160, kernel_size=401,stride=160, padding=401//2, min_low_hz=50, min_band_hz=100)

        self.abs_layer = Abs()

        self.PReLU_layer = torch.nn.PReLU()
        self.Norm_1 = LayerNorm(200,index=-1)

        self.Conv1d_layer1 = torch.nn.Conv1d(160, 60, kernel_size=5, stride=2, padding=5//2)

        self.PReLU_layer1 = torch.nn.PReLU()
        self.Norm_2 = LayerNorm(100,index=-1)

        self.Conv1d_layer2 = torch.nn.Conv1d(60, 60, kernel_size=5, stride=2, padding=5//2)

        #activation
        self.PReLU_layer2 = torch.nn.PReLU()
        self.Norm_3 = LayerNorm(50,index=-1)
        # reshape output of CNN to be suitable for RNN (batch size, T, Cin)
        self.ncl_layer = NCL2NLC()
		
                
        # phoneme RNN
        
        num_rnn_layers = 2
        out_dim = 60
        self.GRU_layer1 = torch.nn.GRU(input_size=60, hidden_size=64, batch_first=True, bidirectional=False)
        # grab hidden states of RNN for each timestep
        self.RNNSelect_layer1 = RNNSelect()
        self.Dropout_rnn_layer1 = torch.nn.Dropout(p=0.1)  
        self.Downsample_rnn_layer1 = Downsample(method="none", factor=1, axis=1)

        out_dim =64 #hidden_size
        self.phoneme_linear = torch.nn.Linear(out_dim, 42)
        
        self.pretraining_type = config.pretraining_type
            
        #### word nn
        out_dim =64
        self.word_downsample_layer1 = Downsample(method="avg", factor=2, axis=1) 
        self.word_gru_layer1 = torch.nn.GRU(input_size=out_dim, hidden_size=64, batch_first=True, bidirectional=False)
        out_dim = 64
        self.word_rnnselect_layer1 = RNNSelect()
        self.word_dropout_layer1 = torch.nn.Dropout(p=0.1)
        self.word_downsample_layer2 = Downsample(method="avg", factor=2, axis=1)        

        self.word_linear = torch.nn.Linear(out_dim, 10000)
        if self.is_cuda:
            self.cuda()
        
        
    def forward(self, x, y_phoneme, y_word):
        """
        x : Tensor of shape (batch size, T)
        y_phoneme : LongTensor of shape (batch size, T')
        y_word : LongTensor of shape (batch size, T'')

        Compute loss for y_word and y_phoneme for each x in the batch.
        """
        if self.is_cuda:
            x = x.cuda()
            y_phoneme = y_phoneme.cuda()
            y_word = y_word.cuda()

        out = x.unsqueeze(1)
        out = self.Sinc_layer(out)
        out = self.abs_layer(out)
        out = self.PReLU_layer(out)
        out = self.Norm_1(out)
        out = self.Conv1d_layer1(out)
        out = self.PReLU_layer1(out)
        out = self.Conv1d_layer2(out)
        out = self.PReLU_layer2(out)
        out = self.Norm_3(out)
        out = self.ncl_layer(out)
        out = self.GRU_layer1(out)
        out = self.RNNSelect_layer1(out)
        out = self.Dropout_rnn_layer1(out)
        out = self.Downsample_rnn_layer1(out)

        phoneme_logits = self.phoneme_linear(out)
        phoneme_out = phoneme_logits
        phoneme_logits = phoneme_logits.view(phoneme_logits.shape[0]*phoneme_logits.shape[1], -1)
        y_phoneme = y_phoneme.view(-1)
        phoneme_loss = torch.nn.functional.cross_entropy(phoneme_logits, y_phoneme, ignore_index=-1)
                
        valid_phoneme_indices = y_phoneme!=-1
        phoneme_acc = (phoneme_logits.max(1)[1][valid_phoneme_indices] == y_phoneme[valid_phoneme_indices]).float().mean()

        # avoid computing 
        if self.pretraining_type == 1:
            word_loss = torch.tensor([0.])
            word_acc = torch.tensor([0.])
        else:
            out = self.word_downsample_layer1(out)
            out = self.word_gru_layer1(out)
            out = self.word_rnnselect_layer1(out)
            out = self.word_dropout_layer1(out)
            out = self.word_downsample_layer2(out)
            word_logits = self.word_linear(out)
            word_logits = word_logits.view(word_logits.shape[0]*word_logits.shape[1], -1)
            y_word = y_word.view(-1)

            word_loss = torch.nn.functional.cross_entropy(word_logits, y_word, ignore_index=-1)
            valid_word_indices = y_word!=-1
            word_acc = (word_logits.max(1)[1][valid_word_indices] == y_word[valid_word_indices]).float().mean()

        return phoneme_loss, word_loss, phoneme_acc, word_acc
        

    def compute_posteriors(self, x):
        self.is_cuda = next(self.parameters()).is_cuda
        if self.is_cuda:
            x = x.cuda()

        out = x.unsqueeze(1)
        for layer in self.phoneme_layers:
            out = layer(out)
        phoneme_logits = self.phoneme_linear(out)

        for layer in self.word_layers:
            out = layer(out)
        word_logits = self.word_linear(out)

        return phoneme_logits, word_logits

    def compute_features(self, x):

        if self.is_cuda:
            x = x.cuda()

        out = x.unsqueeze(1)
        out = self.Sinc_layer(out)
        out = self.abs_layer(out)
        out = self.PReLU_layer(out)
        out = self.Norm_1(out)
        out = self.Conv1d_layer1(out)

        out = self.PReLU_layer1(out)
        out = self.Norm_2(out)

        out = self.Conv1d_layer2(out)
        out = self.PReLU_layer2(out)
        out = self.Norm_3(out)

        out = self.ncl_layer(out)
        out,_ = self.GRU_layer1(out)
        out = self.Dropout_rnn_layer1(out)
        out = self.Downsample_rnn_layer1(out)
        
        #word_nn
        out = self.word_downsample_layer1(out)
        out,_ = self.word_gru_layer1(out)
        out = self.word_dropout_layer1(out)
        out = self.word_downsample_layer2(out)
        return out


class attention(torch.nn.Module):
        def __init__(self,hidden_dim = 64):
            super(attention,self).__init__()

            self.fc1 = torch.nn.Linear(hidden_dim,hidden_dim)
            self.fc2 = torch.nn.Linear(hidden_dim,hidden_dim)

            self.act = torch.nn.PReLU()


        def forward(self,x):

            xfirst =x[:,-1,:]

            query = self.fc1(xfirst)
            query = query.unsqueeze(1)
            attscores = torch.bmm(query,x.transpose(1, 2))
            attscores = torch.nn.functional.softmax(attscores, dim=-1)
            attvector = torch.bmm(attscores, x)
            attvector = attvector.squeeze(1)
            x = self.act(self.fc2(attvector))
            del xfirst , query , attscores , attvector
            return x

class Model(torch.nn.Module):
    """
    End-to-end SLU model.
    """
    def __init__(self, config):
        super(Model, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        self.Sy_intent = config.Sy_intent
        pretrained_model = PretrainedModel(config)
        if config.pretraining_type != 0:
            pretrained_model_path = os.path.join(config.folder, "pretraining", "model_state.pth")
            if self.is_cuda:
                pretrained_model.load_state_dict(torch.load(pretrained_model_path))
            else:
                pretrained_model.load_state_dict(torch.load(pretrained_model_path, map_location="cpu"))
        self.pretrained_model = pretrained_model
        self.unfreezing_type = config.unfreezing_type

        if  self.unfreezing_type  == 0 or self.unfreezing_type  == 2:
            for parameter in self.pretrained_model.parameters():
                parameter.requires_grad = False  # freezing pretrained_model layers 
        elif self.unfreezing_type  == 1:
            print("NO layer frozen")

        self.intent_layers = []
        out_dim = 64

        self.values_per_slot = config.values_per_slot
        self.num_values_total = sum(self.values_per_slot)

        ############## 1
        self.intent_gru_layer1 = torch.nn.GRU(input_size=64, hidden_size=64, batch_first=True, bidirectional=False)

        self.intent_Norm_1 = LayerNorm(64,index=-2)
        out_dim = 64

        self.intent_rnnselect_layer1 = RNNSelect()     
        # dropout
        self.intent_dropout_layer1 = torch.nn.Dropout(p=0.3)
        # downsample
        self.intent_downsample_layer1 = Downsample(method="none", factor=2, axis=1)

        self.attention = attention(out_dim)
        self.intent_linear_layer_1 = torch.nn.Linear(out_dim, 128)
        self.intent_dropout_layer2 = torch.nn.Dropout(p=0.3)		
        self.intent_linear_layer = torch.nn.Linear(128, self.num_values_total)

        if self.is_cuda:
            self.cuda()

    def forward(self, x, y_intent):
        """
        x : Tensor of shape (batch size, T)
        y_intent : LongTensor of shape (batch size, num_slots)
        """
        if self.is_cuda:
            y_intent = y_intent.cuda()
            device = "cuda:0"
        else:
            device = 'cpu'
        out = self.pretrained_model.compute_features(x)


        out ,_= self.intent_gru_layer1(out)
        out = self.intent_Norm_1(out)
        out = self.intent_dropout_layer1(out)
        out = self.intent_downsample_layer1(out)
        out = self.attention(out)
        out = self.intent_linear_layer_1(out)
        out = self.intent_dropout_layer2(out)
        out = self.intent_linear_layer(out)
            
        intent_logits = out
        intent_loss = 0.0
        start_idx = 0
        predicted_intent = []
        predicted_intent_prob = []
        for slot in range(len(self.values_per_slot)):

            end_idx = start_idx + self.values_per_slot[slot]
            subset = intent_logits[:, start_idx:end_idx]
            predicted_intent.append(subset.detach().max(1)[1])
            
            intent_loss += torch.nn.functional.cross_entropy(subset, y_intent[:, slot])
            start_idx = end_idx

        predicted_intent = torch.stack(predicted_intent, dim=1)
        intent_acc = (predicted_intent == y_intent).prod(1).float().mean() # all slots must be correct
        del subset ,predicted_intent , out ,intent_logits ,x
        return intent_loss, intent_acc



    def compute_intents(self,x):
        if self.is_cuda:
            x = x.cuda()
            
        out = self.intent_gru_layer1(x)
        out = self.intent_rnnselect_layer1(out)
        out = self.intent_dropout_layer1(out)
        out = self.intent_downsample_layer1(out)
        out = self.attention(out)
        out = self.intent_linear_layer_1(out)
        out = self.intent_dropout_layer2(out)
        out = self.intent_linear_layer(out)

        return out
            
            
    def predict_intents(self, x):
        torch.set_printoptions(precision=10)
        out = self.pretrained_model.compute_features(x)
        out = self.compute_intents(out)
        device = "cuda:0" if self.cuda else "cpu"
        intent_logits = out
        start_idx = 0
        predicted_intent_prob = []
        predicted_intent = []
        for slot in range(len(self.values_per_slot)):
            end_idx = start_idx + self.values_per_slot[slot]
            subset = intent_logits[:, start_idx:end_idx]
            subset = torch.nn.functional.softmax(subset)
            np.savetxt(str(slot)+"_intent_subset.txt",subset.cpu().detach().numpy().T,fmt="%s")
            [ probability,index] = subset.max(1)

            predicted_intent.append(index)
            predicted_intent_prob.append(probability)

            if predicted_intent_prob[slot] < 0.80:
                predicted_intent[slot] = torch.tensor([0],device= device) ## 0 is none in sy_intent
            start_idx = end_idx
        predicted_intent = torch.stack(predicted_intent, dim=1)
        #exit()
        return intent_logits, predicted_intent


    def decode_intents(self, x):
        _, predicted_intent = self.predict_intents(x)

        intents = []
        for prediction in predicted_intent:
            intent = []
            for idx, slot in enumerate(self.Sy_intent):
                for value in self.Sy_intent[slot]:
                    if prediction[idx].item() == self.Sy_intent[slot][value]:
                        intent.append(value)
            intents.append(intent)
        return intents

    def unfreeze_l2(self):

        if self.unfreezing_type  == 2:
            try:
                self.prelist.pop(-1).requires_grad = True
                print("unfreezed few layers")
            except:
                print("all layers unfreezed")


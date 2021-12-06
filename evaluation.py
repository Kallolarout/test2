import data
import soundfile as sf
import os
import torch , torchaudio
import warnings
import glob
from shutil import copyfile
from tqdm import tqdm
warnings.filterwarnings("ignore")

 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

### reading commands and respective slots
rdata = open('new_53_cmds_act_obj_loc.txt')
lines = rdata.readlines()
csvdict = {}
for line in lines:
    prt = line.split(',', 1)
    csvdict[prt[0]] = prt[1]
####################################################################################
path = "experiments/arg_cmds_653spkrs" + "/"   # specify config path
mode_nm =  "model_state" # specify modelname
import models_test_v2_infer
config = data.read_config(path + "experiment.cfg"); _,_,_=data.get_SLU_datasets(config)
model = models_test_v2_infer.Model(config).eval()
model.load_state_dict(torch.load(path  +"training/"+mode_nm+ ".pth", map_location=device))


file_paths = glob.glob("/root/data_common10tb/kallola/database/speech_commands/*/*.wav")


file = open("detected.csv","w")
file.write("S.no"+","+"Audio in file"+","+"detected as "+","+"prob 1"+","+"prob 2"+","+"filename"+"\n")

file_neg = open("not_detected.csv","w")
file_neg.write("Audio in file"+","+"detected as "+","+"prob 1"+","+"prob 2"+","+"filename"+"\n")

count = 0 
for filename in tqdm(file_paths):#filenames:#
    detected = False
    signal, _ = sf.read(filename,always_2d=True)
    signal = signal[:,0]
    signal = torch.tensor(signal, device=device).float() 
    #print(len(signal))
    if len(signal) <= 32000:
        T = 32000 
        x_pad_length = (T - len(signal))
        signal = torch.nn.functional.pad(signal,(x_pad_length,0)) #(x_pad_length//2,x_pad_length//2)) #(0,x_pad_length))
        signal = signal.unsqueeze(0)
        estimate = model.decode_intents(signal)
        x , [prob1 , prob2] = estimate
        #print(estimate,"--------",filename.split("\\")[-1])#type(estimate))
 
        for key , value in csvdict.items():
 
            if x[0][0] + "," + x[0][1] + "," + "none" == value.strip("\n"):
                command_detected = key  
                #print(command_detected)
                detected = True
                count = count + 1
                break
        if not detected:
            
            command_detected = "none"
            file_neg.write(''.join(filter(str.isalpha, filename.split("\\")[-1]))[:-3]+","+command_detected+","+str(prob1[0])+","+str(prob2[0])+","+filename.split("\\")[-1]+"\n")

        else :
            file.write(str(count)+","+filename+","+command_detected+","+str(prob1[0])+","+str(prob2[0])+","+filename.split("\\")[-1]+"\n")

    else:
       print(filename, " is longer than 2sec")

file.close()
file_neg.close()



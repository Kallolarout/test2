File structure: 
main.py  --> launches pretrainig and training 
data.py --> has pytorch dataloader 
training.py  --> code for back propagatoin over dataloader 
model_test_v2.py -->  slu models code 

Folder structure:
./experiments contain configuration files and respective folders
./experiments/configuration_name contains related pretrainig and training models , logs...etc 

commands:
python main.py --restart --train --config_path=./experiments/arg_cmds_500spkrs.cfg
(arg_cmds_500spkrs.cfg is configuration file)

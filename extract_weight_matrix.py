import data
import models_test_v2
import soundfile as sf
import torch


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


path = "experiments/arg_cmds_500spkrs/"    #config path
config = data.read_config(path + "experiment.cfg"); _,_,_=data.get_SLU_datasets(config)
model = models_test_v2.Model(config).eval()
mode_nm = "model_state"   #model name

model.load_state_dict(torch.load(path  +"training/"+mode_nm+".pth", map_location=device))

torch.set_printoptions(profile="full")

layer_name_dict = {}
layer_weight_dict = {}
for param_tensor in model.state_dict():
    #print(param_tensor, "\t", model.state_dict()[param_tensor].shape)
    x = param_tensor
    layer_name = x.replace(".","_")
    print(layer_name, "\t", model.state_dict()[param_tensor].shape)
    layer_name_dict[layer_name] = list(model.state_dict()[param_tensor].shape)
    layer_weight_dict [layer_name] = model.state_dict()[param_tensor].cpu()





def weight3d_write_fun(file_to_write,weight_mat,name_of_file,dim0,dim1,dim2):
    tmp_fl = open(file_to_write+'.cpp','a')
    tmp_fl.write('float '+name_of_file+'['+str(dim0)+']['+str(dim1)+']['+str(dim2)+'] = { \n')
    for i in weight_mat:
        tmp_fl.write('{')
        for j in i:
            tmp_fl.write('{')
            for k in j:
                tmp_fl.write(str(k.numpy()))
                tmp_fl.write(',')
            tmp_fl.write(' },'+'\n')
        tmp_fl.write(' },'+'\n')
    tmp_fl.write(' };\n\n')
    tmp_fl.close()




def weight2d_write_fun(file_to_write,weight_mat,name_of_file,dim0,dim1):
    tmp_fl = open(file_to_write+'.cpp','a')
    tmp_fl.write('float '+name_of_file+'['+str(dim0)+']['+str(dim1)+'] = { \n')
    for i in weight_mat:
        tmp_fl.write('{')
        for j in i:
            tmp_fl.write(str(j.numpy()))
            tmp_fl.write(',')
        tmp_fl.write(' },'+'\n')
    tmp_fl.write(' };\n\n')
    tmp_fl.close()

def bias1d_write_fun(file_to_write,bias_vec,name_of_file,dim0):
    tmp_fl = open(file_to_write+'.cpp','a')
    tmp_fl.write('float '+name_of_file+'[1]'+str(dim0)+' = { \n {')
    for i in bias_vec:
        tmp_fl.write(str(i.numpy()))
        tmp_fl.write(',')
    tmp_fl.write('} };\n\n' )


wt_file = mode_nm
for k,v in layer_name_dict.items():
    #v = v.cpu()
    print(k)
    print(len(v))
    if len(v) ==1:
        print("one 1d vector")
        if k in layer_weight_dict.keys():
            print(k)
            bias_vec = layer_weight_dict[k]
            bias1d_write_fun(path+wt_file,bias_vec, k ,v)
    elif len(v) ==2:
        print("2D vector")
        if k in layer_weight_dict.keys():
            print(k)
            bias_vec = layer_weight_dict[k]
            weight2d_write_fun(path+wt_file,bias_vec,k,bias_vec.shape[0] ,bias_vec.shape[1])

    elif len(v) ==3:
        print("3D vector")
        if k in layer_weight_dict.keys():
            print(k)
            bias_vec = layer_weight_dict[k]
            weight3d_write_fun(path+wt_file,bias_vec,k,bias_vec.shape[0] ,bias_vec.shape[1],bias_vec.shape[2])
    else:
        print("Something went wrong !!")
print(" Done!! ")

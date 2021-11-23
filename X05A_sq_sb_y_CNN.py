#!/usr/bin/env python
# coding: utf-8
# Microsoft VS header 
import os, os.path
# import winsound
from sys import platform
if os.name == 'nt' or platform == 'win32':
    try:
        os.chdir(os.path.dirname(__file__))
        print("Running in Microsoft VS!")
    except:
        print("Not Running in Microsoft VS")
#########################################################################################################
#########################################################################################################
import sys
import time
import torch
import numpy as np
import pandas as pd
import pickle
import argparse
import scipy
import random
import subprocess
#--------------------------------------------------#
from torch import nn
#--------------------------------------------------#
import seaborn as sns
#--------------------------------------------------#
from pathlib import Path
from copy import deepcopy
#--------------------------------------------------#
from datetime import datetime
from models import CNN_dataset, generate_loader, CNN
from data_processing import Get_represented_X_y_data, Get_represented_X_y_data_clf, split_idx, split_seqs_idx_custom, split_seqs_subs_idx_book, Get_X_y_data_selected
from chemical_encoding import Get_ECFPs_encoding, Get_JTVAE_encoding, Get_Morgan_encoding
#--------------------------------------------------#
seed=42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
## Args
Step_code="X05A_"
data_folder = Path("X_DataProcessing/")
embedding_file_list = ["X03_embedding_ESM_1B.p", 
                       "X03_embedding_BERT.p", 
                       "X03_embedding_TAPE.p", 
                       "X03_embedding_ALBERT.p", 
                       "X03_embedding_T5.p", 
                       "X03_embedding_TAPE_FT.p", 
                       "X03_embedding_Xlnet.p"]
embedding_file = embedding_file_list[0] # embedding_file is a dict, {"seq_embeddings":seq_embeddings, "seq_ids":seq_ids, "seq_all_hiddens":seq_all_hiddens}
properties_file= "X00_substrates_properties_list.p" # properties_file is a list, [ SUBSTRATE_name, Y_Property_value, Y_Property_Class_#1, Y_Property_Class_#2, SMILES_str ]
#====================================================================================================#
# Select properties (Y) of the model 
screen_bool = True
classification_threshold_type = 2 # 2: 1e-5, 3: 1e-2
log_value = False ##### !!!!! If value is True, screen_bool will be changed
#====================================================================================================#
# Select substrate encodings
subs_encodings_list = ["ECFP2", "ECFP4", "ECFP6", "JTVAE", "MorganFP"]
subs_encodings = subs_encodings_list[2]
#---------- ECFP
ECFP_type = subs_encodings[-1] if subs_encodings in ["ECFP2", "ECFP4", "ECFP6",] else 2 # 2, 4, 6
#---------- JTVAE
data_folder_2 = Path("X_DataProcessing/X00_enzyme_datasets_processed/")
subs_JTVAE_file="phosphatase_JTVAE_features.p"
#---------- Morgan
data_folder = Path("X_DataProcessing/")
subs_Morgan1024_file = "X00_phosphatase_Morgan1024_features.p"
subs_Morgan2048_file = "X00_phosphatase_Morgan2048_features.p"
#====================================================================================================#
# Data Split Methods
split_type = 2 # 0, 1, 2, 3
# split_type = 0, train/test/split completely randomly selected
# split_type = 1, train/test/split looks at different seq-subs pairs
# split_type = 2, train/test/split looks at different seqs
# split_type = 3, train/test/split looks at different subs
custom_split = False
customized_idx_file = "X02_customized_idx_selected.p"
#====================================================================================================#
# Prediction NN settings
epoch_num=50
batch_size=256
learning_rate=0.0001
NN_type_list=["Reg", "Clf"]
NN_type=NN_type_list[0]
#====================================================================================================#
hid_dim = 256   # 256
kernal_1 = 3    # 5
out_dim = 1     # 2
kernal_2 = 3    # 3
last_hid = 1024  # 1024
dropout = 0.    # 0
#--------------------------------------------------#
"""model = Encoder(
    d_model= NN_input_dim,
    d_k = d_k,
    n_heads= n_heads,
    d_v = d_v,
    out_dim = 1,
    sub_dim = X_subs_encodings_dim,
    d_ff = last_hid
)"""
#====================================================================================================#
if log_value==True:
    screen_bool = True
if NN_type=="Clf": ##### !!!!! If value is "Clf", log_value will be changed
    screen_bool = False # Actually Useless
    log_value==False
#====================================================================================================#
# Results
results_folder = Path("X_DataProcessing/" + Step_code +"intermediate_results/")
i_o_put_file_1 = "X04B_all_ecfps" + str(ECFP_type) + ".p"
i_o_put_file_2 = "X04B_all_cmpds_ecfps" + str(ECFP_type) + "_dict.p"
output_file_3 = Step_code + "_all_X_y.p"
output_file_header = Step_code + "_result_"
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
# Create Temp Folder for Saving Results
print(">>>>> Creating temporary subfolder and clear past empty folders! <<<<<")
now = datetime.now()
d_t_string = now.strftime("%m%d-%H%M%S")
#====================================================================================================#
results_folder_contents = os.listdir(results_folder)
for item in results_folder_contents:
    if os.path.isdir(results_folder / item):
        try:
            os.rmdir(results_folder / item)
            print("Remove empty folder " + item + "!")
        except:
            print("Found Non-empty folder " + item + "!")
embedding_code=embedding_file.replace("X03_embedding_", "")
embedding_code=embedding_code.replace(".p", "")
temp_folder_name = Step_code + d_t_string + "_" + embedding_code.replace("_","") + "_" + subs_encodings + "_" + NN_type + "_Split" + str(split_type) + "_screen" + str(screen_bool) + "_log" + str(log_value) + "_threshold" + str(classification_threshold_type)
results_sub_folder=Path("X_DataProcessing/" + Step_code + "intermediate_results/" + temp_folder_name +"/")
if not os.path.exists(results_sub_folder):
    os.makedirs(results_sub_folder)
print(">>>>> Temporary subfolder created! <<<<<")
#########################################################################################################
#########################################################################################################
class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # If you want the output to be visible immediately
    def flush(self) :
        for f in self.files:
            f.flush()
#--------------------------------------------------#
orig_stdout = sys.stdout
f = open(results_sub_folder / 'print_out.txt', 'w')
sys.stdout = Tee(sys.stdout, f)
print("="*50)
#--------------------------------------------------#
print("embedding_file: ", embedding_file)
#--------------------------------------------------#
print("log_value: ", log_value," --- Use log values of Y.")
print("screen_bool: ", screen_bool, " --- Whether zeros shall be removed")
print("classification_threshold_type: ", classification_threshold_type, " --- 2: 1e-5, 3: 1e-2")
#--------------------------------------------------#
print("subs_encodings: ", subs_encodings)
print("ECFP_type: ", ECFP_type)
#--------------------------------------------------#
print("split_type: ", split_type)
print("custom_split: ", custom_split)
#--------------------------------------------------#
print("epoch_num: ", epoch_num)
print("batch_size: ", batch_size)
print("learning_rate: ", learning_rate)
print("NN_type: ", NN_type)
#--------------------------------------------------#
print("-"*50)
for i in ['d_k', 'n_heads', 'out_dim', 'd_v', 'last_hid', 'dropout']:
    print(i, ": ", locals()[i])
print("-"*50)
#########################################################################################################
#########################################################################################################
# Get Input files
# Get Sequence Embeddings from X03 pickles.
with open( data_folder / embedding_file, 'rb') as seqs_embeddings:
    seqs_embeddings_pkl = pickle.load(seqs_embeddings)
X_seqs_all_hiddens_list = seqs_embeddings_pkl['seq_all_hiddens'] # new: X_seqs_all_hiddens_list, old: seq_all_hiddens_list
#====================================================================================================#
# Get subs_properties_list.
with open( data_folder / properties_file, 'rb') as subs_properties:
    subs_properties_list = pickle.load(subs_properties) # [[one_substrate, y_prpty_reg, y_prpty_cls, y_prpty_cls2, SMILES], [], [], ...[] ]
#====================================================================================================#
with open( data_folder / customized_idx_file, 'rb') as customized_idx_list:
    customized_idx_list = pickle.load(customized_idx_list)
#====================================================================================================#
# Encode Substrates.
#====================================================================================================#
if NN_type=="Reg":
    X_seqs_all_hiddens, X_subs_representations, y_data, seqs_subs_idx_book = Get_represented_X_y_data(X_seqs_all_hiddens_list, subs_properties_list, screen_bool, classification_threshold_type)
if NN_type=="Clf":
    X_seqs_all_hiddens, X_subs_representations, y_data, seqs_subs_idx_book = Get_represented_X_y_data_clf(X_seqs_all_hiddens_list, subs_properties_list, classification_threshold_type)
#====================================================================================================#
#subs_encodings_list = ["ECFP2", "ECFP4", "ECFP6", "JTVAE", "MorganFP"]
if subs_encodings == "ECFP2" or subs_encodings == "ECFP4"  or subs_encodings == "ECFP6" :
    all_smiles_list=[]
    for one_list_prpt in subs_properties_list:
        all_smiles_list.append(one_list_prpt[-1])
    #print(all_smiles_list)
    #--------------------------------------------------#
    #all_ecfps,all_smiles_ecfps_dict=generate_all_smiles_ecfps_list_dict(all_smiles_list,ecfp_type="ECFP4",iteration_number=1)
    #pickle.dump(all_ecfps, open(data_folder / i_o_put_file_1,"wb") )
    #pickle.dump(all_smiles_ecfps_dict, open(data_folder / i_o_put_file_2,"wb"))
    #====================================================================================================#
    with open( data_folder / i_o_put_file_1, 'rb') as all_ecfps:
        all_ecfps = pickle.load(all_ecfps)
    with open( data_folder / i_o_put_file_2, 'rb') as all_smiles_ecfps_dict:
        all_smiles_ecfps_dict = pickle.load(all_smiles_ecfps_dict)
    X_subs_encodings = Get_ECFPs_encoding(X_subs_representations, all_ecfps, all_smiles_ecfps_dict)
if subs_encodings == "JTVAE":
    with open( data_folder_2 / subs_JTVAE_file, 'rb') as subs_JTVAE_info:
        subs_SMILES_JTVAE_dict = pickle.load(subs_JTVAE_info)
    X_subs_encodings = Get_JTVAE_encoding(X_subs_representations, subs_SMILES_JTVAE_dict)
if subs_encodings == "MorganFP":
    with open( data_folder / subs_Morgan1024_file, 'rb') as subs_Morgan1024_info:
        subs_SMILES_Morgan1024_dict = pickle.load(subs_Morgan1024_info)
    X_subs_encodings = Get_Morgan_encoding(X_subs_representations, subs_SMILES_Morgan1024_dict)
#====================================================================================================#
print("len(X_seqs_all_hiddens): ", len(X_seqs_all_hiddens), ", len(X_subs_encodings): ", len(X_subs_encodings), ", len(y_data): ", len(y_data) )
#====================================================================================================#
# Get size of some interests
X_seqs_all_hiddens_dim = [ max([ X_seqs_all_hiddens_list[i].shape[0] for i in range(len(X_seqs_all_hiddens_list)) ]), X_seqs_all_hiddens_list[0].shape[1], ]
X_subs_encodings_dim = len(X_subs_encodings[0])
X_seqs_num = len(X_seqs_all_hiddens_list)
X_subs_num = len(subs_properties_list)
print("seqs, subs dimensions: ", X_seqs_all_hiddens_dim, ", ", X_subs_encodings_dim)
print("seqs, subs counts: ", X_seqs_num, ", ", X_subs_num)

seqs_max_len = max([  X_seqs_all_hiddens_list[i].shape[0] for i in range(len(X_seqs_all_hiddens_list))  ])
print("seqs_max_len: ", seqs_max_len)
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
# Get Separate SEQS index and SUBS index.
# Get Customized SEQS index if needed.
#====================================================================================================#
if custom_split == True:
    tr_idx_seqs, ts_idx_seqs, va_idx_seqs = split_seqs_idx_custom(X_seqs_num, customized_idx_list, valid_test_split=0.5, random_state=seed)
else:
    tr_idx_seqs, ts_idx_seqs, va_idx_seqs = split_idx(X_seqs_num, train_split=0.8, test_split=0.1, random_state=seed)
    tr_idx_subs, ts_idx_subs, va_idx_subs = split_idx(X_subs_num, train_split=0.8, test_split=0.1, random_state=seed)

#====================================================================================================#
print("len(tr_idx_seqs): ", len(tr_idx_seqs))
print("len(ts_idx_seqs): ", len(ts_idx_seqs))
print("len(va_idx_seqs): ", len(va_idx_seqs))
#########################################################################################################
#########################################################################################################
# Get splitted index of the entire combined dataset.
X_train_idx, X_test_idx, X_valid_idx = split_seqs_subs_idx_book(tr_idx_seqs, ts_idx_seqs, va_idx_seqs, tr_idx_subs, ts_idx_subs, va_idx_subs, seqs_subs_idx_book, split_type)
dataset_size = len(seqs_subs_idx_book)
print("dataset_size: ", dataset_size)
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
# Get splitted data of the combined dataset through the index.
#====================================================================================================#
X_tr_seqs, X_tr_subs, y_tr = Get_X_y_data_selected(X_train_idx, X_seqs_all_hiddens, X_subs_encodings, y_data, log_value)
X_ts_seqs, X_ts_subs, y_ts = Get_X_y_data_selected(X_test_idx, X_seqs_all_hiddens, X_subs_encodings, y_data, log_value)
X_va_seqs, X_va_subs, y_va = Get_X_y_data_selected(X_valid_idx, X_seqs_all_hiddens, X_subs_encodings, y_data, log_value)
#print("Done getting X_data and y_data!")
print("X_tr_seqs_dimension: ", len(X_tr_seqs), ", X_tr_subs_dimension: ", len(X_tr_subs), ", y_tr_dimension: ", y_tr.shape )
print("X_ts_seqs_dimension: ", len(X_ts_seqs), ", X_ts_subs_dimension: ", len(X_ts_subs), ", y_ts_dimension: ", y_ts.shape )
print("X_va_seqs_dimension: ", len(X_va_seqs), ", X_va_subs_dimension: ", len(X_va_subs), ", y_va_dimension: ", y_va.shape )
#########################################################################################################
#########################################################################################################
train_loader, valid_loader, test_loader = generate_loader(CNN_dataset, X_tr_seqs, X_tr_subs, y_tr, batch_size, X_va_seqs, X_va_subs, y_va, X_ts_seqs, X_ts_subs, y_ts)
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
NN_input_dim=X_seqs_all_hiddens_dim[1]
print("NN_input_dim: ", NN_input_dim)
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#====================================================================================================#
model = CNN(
            in_dim = NN_input_dim,
            hid_dim = hid_dim,
            kernal_1 = kernal_1,
            out_dim = out_dim, #2
            kernal_2 = kernal_2,
            max_len = seqs_max_len,
            sub_dim = X_subs_encodings_dim,
            last_hid = last_hid, #256
            dropout = 0.
            )
#########################################################################################################
#########################################################################################################
model.double()
model.cuda()
#--------------------------------------------------#
print("#"*50)
print(model)
#model.float()
#print( summary( model,[(seqs_max_len, NN_input_dim), (X_subs_encodings_dim, )] )  )
#model.double()
print("#"*50)
#--------------------------------------------------#
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
#optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
criterion = nn.MSELoss()
#====================================================================================================#
for epoch in range(epoch_num): 
    model.train()
    #====================================================================================================#
    count_x=0
    for one_seqsubs_ppt_group in train_loader:
        len_train_loader=len(train_loader)
        count_x+=1
        #--------------------------------------------------#
        seq_rep, subs_rep, target = one_seqsubs_ppt_group["embedding"], one_seqsubs_ppt_group["substrate"], one_seqsubs_ppt_group["target"]
        seq_rep, subs_rep, target = seq_rep.double().cuda(), subs_rep.double().cuda(), target.double().cuda()
        output, _ = model(seq_rep, subs_rep)
        loss = criterion(output,target.view(-1,1))
        if ((count_x) % 10) == 0:
            print(str(count_x)+"/"+str(len_train_loader)+"->", end=" ")
        optimizer.zero_grad()
        loss.backward()
        #nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
        
        optimizer.step()
    #====================================================================================================#
    model.eval()
    y_pred_valid = []
    y_real_valid = []
    #--------------------------------------------------#
    for one_seqsubs_ppt_group in valid_loader:
        seq_rep, subs_rep, target = one_seqsubs_ppt_group["embedding"], one_seqsubs_ppt_group["substrate"], one_seqsubs_ppt_group["target"]
        seq_rep, subs_rep = seq_rep.double().cuda(), subs_rep.double().cuda()
        output, _ = model(seq_rep, subs_rep)
        output = output.cpu().detach().numpy().reshape(-1)
        target = target.numpy()
        y_pred_valid.append(output)
        y_real_valid.append(target)
    y_pred_valid = np.concatenate(y_pred_valid)
    y_real_valid = np.concatenate(y_real_valid)
    slope, intercept, r_value_va, p_value, std_err = scipy.stats.linregress(y_pred_valid, y_real_valid)
    #====================================================================================================#
    y_pred = []
    y_real = []
    #--------------------------------------------------#
    for one_seqsubs_ppt_group in test_loader:
        seq_rep, subs_rep, target = one_seqsubs_ppt_group["embedding"], one_seqsubs_ppt_group["substrate"], one_seqsubs_ppt_group["target"]
        seq_rep, subs_rep = seq_rep.double().cuda(), subs_rep.double().cuda()
        output, _ = model(seq_rep, subs_rep)
        output = output.cpu().detach().numpy().reshape(-1)
        target = target.numpy()
        y_pred.append(output)
        y_real.append(target)
    y_pred = np.concatenate(y_pred)
    y_real = np.concatenate(y_real)
    #--------------------------------------------------#
    if log_value == False:
        y_pred[y_pred<0]=0
    #--------------------------------------------------#
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(y_pred, y_real)
    print()
    print("epoch: {} | vali_r_value: {} | loss: {} | test_r_value: {} ".format( str((epoch+1)+1000).replace("1","",1) , np.round(r_value_va,4), loss, np.round(r_value, 4)))
    #====================================================================================================#
    if ((epoch+1) % 1) == 0:
        _, _, r_value, _ , _ = scipy.stats.linregress(y_pred, y_real)
        pred_vs_actual_df = pd.DataFrame(np.ones(len(y_pred)))
        pred_vs_actual_df["actual"] = y_real
        pred_vs_actual_df["predicted"] = y_pred
        pred_vs_actual_df.drop(columns=0, inplace=True)
        pred_vs_actual_df.head()
        #--------------------------------------------------#
        sns.set_theme(style="darkgrid")
        y_interval=max(np.concatenate((y_pred, y_real),axis=0))-min(np.concatenate((y_pred, y_real),axis=0))
        x_y_range=(min(np.concatenate((y_pred, y_real),axis=0))-0.1*y_interval, max(np.concatenate((y_pred, y_real),axis=0))+0.1*y_interval)
        g = sns.jointplot(x="actual", y="predicted", data=pred_vs_actual_df,
                            kind="reg", truncate=False,
                            xlim=x_y_range, ylim=x_y_range,
                            color="blue",height=7)

        g.fig.suptitle("Predictions vs. Actual Values, R = " + str(round(r_value,3)) + ", Epoch: " + str(epoch+1) , fontsize=18, fontweight='bold')
        g.fig.tight_layout()
        g.fig.subplots_adjust(top=0.95)
        g.ax_joint.text(0.4,0.6,"", fontsize=12)
        g.ax_marg_x.set_axis_off()
        g.ax_marg_y.set_axis_off()
        g.ax_joint.set_xlabel('Actual Values',fontsize=18 ,fontweight='bold')
        g.ax_joint.set_ylabel('Predictions',fontsize=18 ,fontweight='bold')
        g.savefig(results_sub_folder / (output_file_header + "epoch_" + str(epoch+1)) )
    #====================================================================================================#
        if log_value == False and screen_bool==True:

            y_real = np.delete(y_real, np.where(y_pred == 0.0))
            y_pred = np.delete(y_pred, np.where(y_pred == 0.0))

            y_real = np.log10(y_real)
            y_pred = np.log10(y_pred)

            log_pred_vs_actual_df = pd.DataFrame(np.ones(len(y_pred)))
            log_pred_vs_actual_df["log(actual)"] = y_real
            log_pred_vs_actual_df["log(predicted)"] = y_pred
            log_pred_vs_actual_df.drop(columns=0, inplace=True)

            y_interval = max(np.concatenate((y_pred, y_real),axis=0))-min(np.concatenate((y_pred, y_real),axis=0))
            x_y_range = (min(np.concatenate((y_pred, y_real),axis=0))-0.1*y_interval, max(np.concatenate((y_pred, y_real),axis=0))+0.1*y_interval)
            g = sns.jointplot(x="log(actual)", y="log(predicted)", data=log_pred_vs_actual_df,
                                kind="reg", truncate=False,
                                xlim=x_y_range, ylim=x_y_range,
                                color="blue",height=7)

            g.fig.suptitle("Predictions vs. Actual Values, R = " + str(round(r_value,3)) + ", Epoch: " + str(epoch+1) , fontsize=18, fontweight='bold')
            g.fig.tight_layout()
            g.fig.subplots_adjust(top=0.95)
            g.ax_joint.text(0.4,0.6,"", fontsize=12)
            g.ax_marg_x.set_axis_off()
            g.ax_marg_y.set_axis_off()
            g.ax_joint.set_xlabel('Log(Actual Values)',fontsize=18 ,fontweight='bold')
            g.ax_joint.set_ylabel('Log(Predictions)',fontsize=18 ,fontweight='bold')
            g.savefig(results_sub_folder / (output_file_header + "_log_plot, epoch_" + str(epoch+1)) )
#########################################################################################################
#########################################################################################################
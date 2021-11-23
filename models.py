import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import scipy
import torch
from torch import nn
from torch.utils import data
from torch.nn.utils.weight_norm import weight_norm
from typing import Union, List, Tuple, Sequence, Dict, Any, Optional, Collection

class ATT_dataset(data.Dataset):
    def __init__(self, embedding, substrate, label):
        super().__init__()
        self.embedding = embedding
        self.substrate = substrate
        self.label = label

    def __len__(self):
        return len(self.embedding)

    def __getitem__(self, idx):
        return self.embedding[idx], self.substrate[idx], self.label[idx]

    def collate_fn(self, batch:List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        embedding, substrate, target = zip(*batch)
        batch_size = len(embedding)
        emb_dim = embedding[0].shape[1]
        max_len = np.max([s.shape[0] for s in embedding],0)
        arra = np.full([batch_size,max_len,emb_dim], 0.0)
        seq_mask = []
        for arr, seq in zip(arra, embedding):
            padding_len = max_len - len(seq)
            seq_mask.append(np.concatenate((np.ones(len(seq)),np.zeros(padding_len))).reshape(-1,max_len))
            arrslice = tuple(slice(dim) for dim in seq.shape)
            arr[arrslice] = seq
        seq_mask = np.concatenate(seq_mask, axis=0)        
        return {'embedding': torch.from_numpy(arra), 'mask': torch.from_numpy(seq_mask), 'substrate': torch.tensor(list(substrate)), 'target': torch.tensor(list(target))}

def ATT_loader(dataset_class,training_embedding,training_substrate,training_target,batch_size,validation_embedding,validation_substrate,validation_target,test_embedding,test_substrate,test_target):
    
    emb_train = dataset_class(list(training_embedding),list(training_substrate),training_target)
    emb_validation = dataset_class(list(validation_embedding),list(validation_substrate),validation_target)
    emb_test = dataset_class(list(test_embedding),list(test_substrate),test_target)
    trainloader = data.DataLoader(emb_train,batch_size,True,collate_fn=emb_train.collate_fn)
    validation_loader = data.DataLoader(emb_validation,batch_size,False,collate_fn=emb_validation.collate_fn)
    test_loader = data.DataLoader(emb_test,batch_size,False,collate_fn=emb_test.collate_fn)
    return trainloader, validation_loader, test_loader

class CNN_dataset(data.Dataset):
    def __init__(self, embeding, substrate, target, max_len):
        super().__init__()
        self.embedding = embeding
        self.substrate = substrate
        self.target = target
        self.max_len = max_len

    def __len__(self):
        return len(self.embedding)

    def __getitem__(self, idx):
        return self.embedding[idx], self.substrate[idx], self.target[idx]

    def collate_fn(self, batch:List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        embedding, substrate, target = zip(*batch)
        batch_size = len(embedding)
        emb_dim = embedding[0].shape[1]
        arra = np.full([batch_size,self.max_len,emb_dim], 0.0)
        for arr, seq in zip(arra, embedding):
            arrslice = tuple(slice(dim) for dim in seq.shape)
            arr[arrslice] = seq        
        return {'embedding': torch.from_numpy(arra), 'substrate': torch.tensor(list(substrate)), 'target': torch.tensor(list(target))}

def generate_loader(dataset_class,training_embedding,training_substrate,training_target,max_len,batch_size,validation_embedding,validation_substrate,validation_target,test_embedding,test_substrate,test_target):
    
    emb_train = dataset_class(list(training_embedding),list(training_substrate),training_target,max_len)
    emb_validation = dataset_class(list(validation_embedding),list(validation_substrate),validation_target,max_len)
    emb_test = dataset_class(list(test_embedding),list(test_substrate),test_target,max_len)
    train_loader = data.DataLoader(emb_train,batch_size,True,collate_fn=emb_train.collate_fn)
    validation_loader = data.DataLoader(emb_validation,batch_size,False,collate_fn=emb_validation.collate_fn)
    test_loader = data.DataLoader(emb_test,batch_size,False,collate_fn=emb_test.collate_fn)
    return train_loader, validation_loader, test_loader

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1,-2))
        scores.masked_fill_(attn_mask,-1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V) # [batch_size, n_heads, len_q, d_v]
        return context, attn

class MultiHeadAttentionwithonekey(nn.Module):
    def __init__(self,d_model,d_k,n_heads,d_v,out_dim):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.out_dim = out_dim
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, out_dim, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        Q = self.W_Q(input_Q).view(input_Q.size(0),-1, self.n_heads, self.d_k).transpose(1,2)
        K = self.W_K(input_K).view(input_Q.size(0),-1, self.n_heads, self.d_k).transpose(1,2)
        V = self.W_V(input_V).view(input_V.size(0),-1, self.n_heads, self.d_v).transpose(1,2)
        attn_mask = attn_mask.unsqueeze(1).repeat(1,self.n_heads,1,1)
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(input_Q.size(0), -1, self.n_heads * self.d_v)
        output = self.fc(context) # [batch_size, len_q, out_dim]
        return output, attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self,d_model,sub_dim,d_ff):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model+sub_dim, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, 1)
            )

    def forward(self, inputs, substrate):
        '''
        inputs: [batch_size, src_len, out_dim]
        '''
        inputs = torch.cat((torch.flatten(inputs, start_dim=1),substrate),1)
        output = self.fc(inputs)
        return output

class EncoderLayer(nn.Module):
    def __init__(self, d_model,d_k,n_heads,d_v,out_dim,sub_dim,d_ff): #out_dim = 1, n_head = 4, d_k = 256
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttentionwithonekey(d_model,d_k,n_heads,d_v,out_dim)
        self.pos_ffn = PoswiseFeedForwardNet(d_model,sub_dim,d_ff)

    def forward(self, enc_inputs, enc_self_attn_mask, input_mask, substrate):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        '''
        # enc_outputs: [batch_size, src_len, 1], attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        pos_weights = nn.Softmax(dim=1)(enc_outputs.masked_fill_(input_mask.unsqueeze(2).data.eq(0), -1e9)).permute(0,2,1) # [ batch_size, 1, src_len]
        enc_outputs = torch.matmul(pos_weights,enc_inputs)
        enc_outputs = self.pos_ffn(enc_outputs,substrate) # enc_outputs: [batch_size, d_model]
        return enc_outputs, pos_weights

class Encoder(nn.Module):
    def __init__(self,d_model,d_k,n_heads,d_v,out_dim,sub_dim,d_ff):
        super(Encoder, self).__init__()
        self.layers = EncoderLayer(d_model,d_k,n_heads,d_v,out_dim,sub_dim,d_ff)

    def get_attn_pad_mask(self, seq_mask):
        batch_size, len_q = seq_mask.size()
        _, len_k = seq_mask.size()
        # eq(zero) is PAD token
        pad_attn_mask = seq_mask.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], True is masked
        return pad_attn_mask.expand(batch_size, len_q, len_k)        

    def forward(self, enc_inputs, input_mask, substrate):
        '''
        enc_inputs: [batch_size, src_len, embedding_dim]
        input_mask: [batch_size, src_len]
        '''

        enc_self_attn_mask = self.get_attn_pad_mask(input_mask) # [batch_size, src_len, src_len]
        # enc_outputs: [batch_size, src_len, out_dim], enc_self_attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs, enc_self_attn = self.layers(enc_inputs, enc_self_attn_mask, input_mask, substrate)
        return enc_outputs, enc_self_attn

class Weight_linear(nn.Module):
    def __init__(self, d_model,d_k,sub_dim, d_ff, dropout):
        super().__init__()
        self.weight_h = weight_norm(nn.Linear(d_model, d_k),dim=None)
        self.weight = weight_norm(nn.Linear(d_k, 1),dim=None)
        self.hidden = weight_norm(nn.Linear(d_model+sub_dim, d_ff),dim=None)
        self.dropout = nn.Dropout(p=dropout)
        self.feedforward = weight_norm(nn.Linear(d_ff, 1),dim=None)

    def forward(self, enc_inputs, input_mask, substrate):
        '''
        enc_inputs: [batch_size, src_len, embedding_dim]
        input_mask: [batch_size, src_len]
        '''
        position_weight = nn.functional.relu(self.weight_h(enc_inputs))
        position_weight = nn.Softmax(dim=1)(self.weight(position_weight).masked_fill_(input_mask.unsqueeze(2).data.eq(0), -1e9)).permute(0,2,1) #[batch_size, 1, src_len]
        enc_outputs = torch.matmul(position_weight,enc_inputs) #[batch_size, 1, d_model]
        enc_outputs = torch.cat((torch.flatten(enc_outputs, start_dim=1),substrate),1) #[batch_size, d_model+sub_dim]
        enc_outputs = nn.functional.relu(self.hidden(enc_outputs))
        #print(enc_outputs.size())
        enc_outputs = self.dropout(enc_outputs)
        enc_outputs = self.feedforward(enc_outputs)
        return enc_outputs, position_weight
     

class CNN(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hid_dim: int,
                 kernal_1: int,
                 out_dim: int,
                 kernal_2: int,
                 max_len: int,
                 sub_dim: int,
                 last_hid: int,
                 dropout: float = 0.
                 ):
        super().__init__()
        self.norm = nn.BatchNorm1d(in_dim)
        self.conv1 = nn.Conv1d(in_dim, hid_dim, kernal_1, padding=int((kernal_1-1)/2))
        self.dropout1 = nn.Dropout(dropout, inplace=True)
        #--------------------------------------------------#
        self.conv2_1 = nn.Conv1d(hid_dim, out_dim, kernal_2, padding=int((kernal_2-1)/2))
        self.dropout2_1 = nn.Dropout(dropout, inplace=True)
        #--------------------------------------------------#
        self.conv2_2 = nn.Conv1d(hid_dim, hid_dim, kernal_2, padding=int((kernal_2-1)/2))
        self.dropout2_2 = nn.Dropout(dropout, inplace=True)
        #--------------------------------------------------#
        self.fc_early = nn.Linear(max_len*hid_dim+sub_dim,1)
        #--------------------------------------------------#
        self.conv3 = nn.Conv1d(hid_dim, out_dim, kernal_2, padding=int((kernal_2-1)/2))
        self.dropout3 = nn.Dropout(dropout, inplace=True)
        #self.pooling = nn.MaxPool1d(3, stride=3,padding=1)
        #--------------------------------------------------#
        self.fc_1 = nn.Linear(int(2*max_len*out_dim+sub_dim),last_hid)
        self.fc_2 = nn.Linear(last_hid,last_hid)
        self.fc_3 = nn.Linear(last_hid,1)
        self.cls = nn.Sigmoid()

    def forward(self, enc_inputs, substrate):
        #--------------------------------------------------#
        output = enc_inputs.transpose(1, 2)
        output = nn.functional.relu(self.conv1(self.norm(output)))
        output = self.dropout1(output)
        #--------------------------------------------------#
        output_1 = nn.functional.relu(self.conv2_1(output))
        output_1 = self.dropout2_1(output_1)
        #--------------------------------------------------#
        output_2 = nn.functional.relu(self.conv2_2(output)) + output
        output_2 = self.dropout2_2(output_2)
        #--------------------------------------------------#
        single_conv = torch.cat( (torch.flatten(output_2,1),substrate) ,1)
        single_conv = self.cls(self.fc_early(single_conv))
        #--------------------------------------------------#
        output_2 = nn.functional.relu(self.conv3(output_2))
        output_2 = self.dropout3(output_2)
        #--------------------------------------------------#
        output = torch.cat((output_1,output_2),1)
        #--------------------------------------------------#
        #output = self.pooling(output)
        #--------------------------------------------------#
        output = torch.cat( (torch.flatten(output,1), substrate) ,1)
        #--------------------------------------------------#
        output = self.fc_1(output)
        output = nn.functional.relu(output)
        output = self.fc_2(output)
        output = nn.functional.relu(output)
        output = self.fc_3(output)
        return output, single_conv
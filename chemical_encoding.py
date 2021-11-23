import numpy as np
def list_smiles_to_ecfp_through_dict(smiles_list, all_smiles_ecfps_dict):
    ecfp_list=[]
    for one_smiles in smiles_list:
        ecfp_list=ecfp_list + all_smiles_ecfps_dict[one_smiles]
    return ecfp_list
#====================================================================================================#
def smiles_to_ECFP_vec( smiles_x, all_ecfps, all_smiles_ecfps_dict):
    dimension=len(all_ecfps)
    Xi=[0]*dimension
    Xi_ecfp_list=list_smiles_to_ecfp_through_dict( [smiles_x, ] ,all_smiles_ecfps_dict)
    for one_ecfp in Xi_ecfp_list:
        Xi[all_ecfps.index(one_ecfp)]=Xi_ecfp_list.count(one_ecfp)
    return np.array(Xi)
#====================================================================================================#
def Get_ECFPs_encoding(X_subs_representations, all_ecfps, all_smiles_ecfps_dict):
    X_subs_encodings=[]
    for one_smiles in X_subs_representations:
        one_subs_encoding = smiles_to_ECFP_vec(one_smiles, all_ecfps, all_smiles_ecfps_dict) # substrate_encoding
        X_subs_encodings.append(one_subs_encoding)
    return X_subs_encodings
#====================================================================================================#
def Get_JTVAE_encoding(X_subs_representations, subs_SMILES_JTVAE_dict):
    X_subs_encodings=[]
    for one_smiles in X_subs_representations:
        one_subs_encoding = subs_SMILES_JTVAE_dict[one_smiles] # substrate_encoding
        X_subs_encodings.append(one_subs_encoding)
    return X_subs_encodings
#====================================================================================================#
def Get_Morgan_encoding(X_subs_representations, subs_SMILES_Morgan1024_dict):
    X_subs_encodings=[]
    for one_smiles in X_subs_representations:
        one_subs_encoding = subs_SMILES_Morgan1024_dict[one_smiles] # substrate_encoding
        X_subs_encodings.append(one_subs_encoding)
    return X_subs_encodings
import sys

sys.path.append('./tools')
sys.path.append('./data')

import numpy as np
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import Guided_Cond_DDPM_Tools as GC_DDPM
import pandas as pd

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

from HullParameterization import Hull_Parameterization as HP


np.set_printoptions(suppress=True) # don't use scientific notation


#Step 1: Load in the data # the Design Paramaters Set
DesVec = np.load('./data/DesVec_82k.npy', allow_pickle=True)
print(DesVec.shape)
# print("DesVec 1: ", DesVec[0,:])
DesVec_neg = np.load('./data/Negative_DesVec_82k.npy', allow_pickle=True)
print(DesVec_neg.shape)
# print("DesVec_neg 1: ", DesVec_neg[0,:])

reference_design = np.load('./data/Par_val.npy', allow_pickle=True)
# print("Peak values: ", Peak_values)

# Now lets clean up X

idx_BBFactors = [33,34,35,36,37]
idx_BB = 31

idx_SBFactors = [38,39,40,41,42,43,44]
idx_SB = 32

for i in range(0,len(DesVec)):
    
    DesVec[i,idx_BBFactors] = DesVec[i,idx_BB] * DesVec[i,idx_BBFactors] 
    DesVec[i,idx_SBFactors] = DesVec[i,idx_SB] * DesVec[i,idx_SBFactors]



Y = np.load('./data/GeometricMeasures.npy', allow_pickle=True) # 12 Geometry preporties 
print("Volume: ", Y.shape)

LenRatios = np.load('./data/Length_Ratios.npy', allow_pickle=True) 


X_LIMITS = np.load('./data/X_LIMITS.npy') # Range of each parameters

print(X_LIMITS.shape)

X_lower_lim = [X_LIMITS[:,0].tolist()]                   
X_upper_lim = [X_LIMITS[:,1].tolist()]

#Set up Conditioning Vectors:
num_WL_Steps = 101

VolVec = np.log10(Y[:,1*num_WL_Steps:2*num_WL_Steps]) # volume 
idx = np.where(np.isnan(VolVec))
print(idx)

VolVec[idx] = -6.0 #fix nan to dummy value

print(VolVec.shape)

DdVec = DesVec[:,4] # Depth 
BOAVec = np.amax(LenRatios[:,1:3], axis=1) # Beam
print(BOAVec.shape) 

Cw = np.load('./data/Cw_82k.npy', allow_pickle=True)
print("Cw:", Cw.shape)
print("Cw 1: ", Cw[1])

# Set up the file for architecting the network, diffusion parameters, and training

DDPM_C_Dict = {
        'xdim' : len(DesVec[0]) -1,             # Dimension of parametric design vector
        'datalength': len(DesVec),           # number of samples
        'X_LL' : X_lower_lim,           # lower limits of parametric design vector variables
        'X_UL' : X_upper_lim,
        'ydim': 0,                       # Number of objectives
        'cdim': 6,                      # number of conditioning inputs
        'gamma' : 0.7,                  # 0.2 weight of feasibility guidance for guided sampling
        'lambda': [0.7,0.3],                 # weight of drag  guidance for guided sampling
        #'lambdas': [1,1,1,1,1,1,1],     # dummy variable for performance guided sampling
        'tdim': 128,                    # 128 dimension of latent variable
        # 'net': [1024,1024,1024,       1024],
        'net': [512,512,512,512],   # network architecture
        'batch_size': 1024,             # batch size
        'Training_Epochs': 10000,      # number of training epochs
        'Diffusion_Timesteps': 1000,    # number of diffusion timesteps
        'lr' : 0.00025,                 # learning rate
        'weight_decay': 0.0,            # weight decay  
        'device_name': 'cuda:0'}        # gpu device name

Classify_Dict = {
        'xdim' : len(DesVec[0])-1,
        'cdim': 1,
        'tdim': 128,
        'net': [64,64,64],
        'Training_Epochs': 150000,
        'device_name': 'cuda:0'}

nodes = 512
Drag_Reg_Dict = {
        'xdim' : len(DesVec[0])-1,              # Dimension of parametric design vector
        'ydim': 1,                              # trains regression model for each objective
        'tdim': nodes,                            # dimension of latent variable
        'net': [nodes,nodes,nodes,nodes],                       # network architecture        
        'Training_Epochs': 30000,  #30000             # number of training epochs
        'batch_size': 1024,                       # batch size
        'Model_Label': 'Regressor_CT',         # labels for regressors       
        'lr' : 0.001,                          # learning rate
        'weight_decay': 0.0,                   # weight decay
        'device_name': 'cuda:0'} 

nodes = 256
LOA_wBulb_Reg_Dict = {
        'xdim' : len(DesVec[0])-1,              # Dimension of parametric design vector
        'ydim': 1,                              # trains regression model for each objective
        'tdim': nodes,                            # dimension of latent variable
        'net': [nodes,nodes,nodes],                       # network architecture        
        'Training_Epochs': 150000,               # number of training epochs
        'batch_size': 1024,                       # batch size
        'Model_Label': 'Regressor_LOA_wBulb',         # labels for regressors
                    
        'lr' : 0.001,                          # learning rate
        'weight_decay': 0.0,                   # weight decay
        'device_name': 'cuda:0'}   

WL_Reg_Dict = {
        "xdim": len(DesVec[0]),
        "ydim": 1, 
        "tdim": 512, 
        "net": [512, 512, 512], 
        "Training_Epochs": 30000, 
        "batch_size": 1024, 
        "Model_Label": 
        "Regressor_WL", 
        "lr": 0.001, 
        "weight_decay": 0.0, 
        "device_name": "cuda:0"}

Vol_Reg_Dict = {
                "xdim": len(DesVec[0]), 
                "ydim": 1, 
                "tdim": 512, 
                "net": [512, 512, 512], 
                "Training_Epochs": 30000, 
                "batch_size": 1024, 
                "Model_Label": "Regressor_WL", 
                "lr": 0.001, 
                "weight_decay": 0.0, 
                "device_name": "cuda:0"}

T2 = GC_DDPM.GuidedDiffusionEnv(DDPM_C_Dict,
                Classify_Dict,
                Drag_Reg_Dict,
                LOA_wBulb_Reg_Dict,
                WL_Reg_Dict,
                Vol_Reg_Dict,
                X= DesVec[:,1:],
                X_neg= DesVec_neg,
                VolVec = VolVec, 
                BOAVec = BOAVec, 
                DdVec = DdVec,
                Cw = Cw)

'''
===================================================
train classifier
===================================================
'''

classifier_path = './TrainedModels/Constraint_Classifier_150000Epochs.pth' 

T2.load_trained_classifier_model(classifier_path)

'''
===================================================
Load Regression Models
==================================================
'''
PATHS = ['./TrainedModels/Regressor_CT.pth',
        './TrainedModels/Regressor_LOA_wBulb.pth',
        './TrainedModels/Regressor_WL.pth',
        './TrainedModels/Regressor_Vol.pth']

T2.load_trained_Drag_regression_models(PATHS)

                   
diffusion_cw_path = './TrainedModels/CShipGen_Cw_8M_BD_diffusion.pth'
T2.load_trained_diffusion_model(diffusion_cw_path)

#Sample from the Model:
num_samples = 512

# Wave drag condtion 
Ships = np.array([[333, 42.624, 11.28, 29.064, 0.0003, 0.43], #Nimitz Class Carrier [LOA(m), BOA(m), T(m), Dd(m), Ct, Fn] Ct: 0.003826
                  ])

Labels = ['333', 'Kayak', 'Neo-Panamax Container Ship', 'NSC', 'ROPAX ferry']

# Run the Loop on the other samples:
Study_Label = 'Study_' + str(0) + '_' + Labels[0] + '_CT_' + str(Ships[0, 4]) + '_Fn_' + str(Ships[0, 5])

print('Generating Hulls')

LOA = Ships[0,0] #in meters
BoL = Ships[0,1]/LOA #beam to length ratio
ToD = Ships[0,2]/Ships[0,3] #Draft to depth ratio
DoL = Ships[0,3]/LOA #Depth to length ratio
Ct = Ships[0, 4] # Wave drag coefficient

Fn = Ships[0,5]  #  12.86 #m/s  = 25 knots

dim_d = np.array([[ToD, Fn, LOA, Ct, DoL, BoL]]) #Drag_conditioning is [ToD, Fn(m/s), LOA (m), Ct]
# num_sample = 512
drag_cond = np.repeat(dim_d, num_samples, axis=0) #reapeat 512 
#print(cond.shape)

# dim_g = np.array([[ToD, BoL, DoL, Ct]])            # Condition Vector
print('Condition: ',drag_cond.shape )

# Parameters of known design (44 parameters, LOA given in CshipGen)
x_gt = torch.tensor([4.96634266e-01  ,4.51955760e-01  ,1.09615953e-01,
8.75783026e-02,  5.71478128e-01,  1.43758851e-01,  1.31093639e-01,
    3.29564095e+00,  3.16418976e-01,  -3.14146566e-01,  3.54170799e-01,
4.29022789e-01,  1.32593811e-01,  6.56113803e-01,  1.69787741e+00,
-1.84923196e+00,  3.88996124e-01, -3.26323986e-01,  1.01175034e+01,
0.00000000e+00,  0.00000000e+00, -2.73985386e-01,  4.10849154e-01,
2.80871749e-01, -2.13681006e+00,  1.77291155e+00,  1.66059315e+00,
7.07736760e-02,  4.26586211e-01,  2.07895815e-01,  0.00000000e+00,
0.00000000e+00,  0.00000000e+00,  0.00000000e+00 , 0.00000000e+00,
0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
0.00000000e+00]).to('cuda:0')

'''
===================================================
Configurate RaPaint
==================================================
'''
#----------------------------------
# print('GT: ', x_gt)
# Mask configuration
# 0-5 :Principle Dimensions 
# 6-9: Midship Cross Section 
# 10-18: Bow Geometry 
# 19-29: Stern Geometry 
# 30-43: Bulb Geometries
#----------------------------------
keep_mask = torch.zeros(x_gt.shape).to('cuda:0')
keep_mask[9:10] = 1
Enable_mask = 1     # Enable RePaint

# print("mask:", keep_mask)
# print("temp: ", x_gt*keep_mask)

# normalization value of reference design values:
x_gt_unorm = x_gt
x_gt =x_gt.cpu().numpy()
x_gt = x = 2.0*(x_gt-T2.data_norm.X_LL_Scaled)/(T2.data_norm.X_UL_Scaled- T2.data_norm.X_LL_Scaled) - 1.0
x_gt = T2.data_norm.normalizer.transform(x)     #  transform_Data(self,X) in Guided_Cond_DDPM_Tools.py
x_gt = torch.from_numpy(x_gt.astype('float32')).to('cuda:0')

'''
===================================================
Main Loop
==================================================
'''
results = []
result = []
for Pa_ind in range(20,21):
    # Reset the mask and configure new one
    keep_mask = torch.zeros(keep_mask.shape).to('cuda:0')
    if Pa_ind == 0:
        keep_mask[6] = 1
        keep_mask[7] = 1
        keep_mask[8] = 0
        keep_mask[9] = 0
        casename = "Fix 6 7"

    elif Pa_ind == 1:
        keep_mask[6] = 1
        keep_mask[7] = 0
        keep_mask[8] = 1
        keep_mask[9] = 0
        casename = "Fix 6 8"

    elif Pa_ind == 2:
        keep_mask[6] = 1
        keep_mask[7] = 0
        keep_mask[8] = 0
        keep_mask[9] = 1
        casename = "Fix 6 9"

    elif Pa_ind == 3:
        keep_mask[6] = 0
        keep_mask[7] = 1
        keep_mask[8] = 1
        keep_mask[9] = 0
        casename = "Fix 7 8"

    elif Pa_ind == 4:
        keep_mask[6] = 0
        keep_mask[7] = 1
        keep_mask[8] = 0
        keep_mask[9] = 1
        casename = "Fix 7 9"

    elif Pa_ind == 5:
        keep_mask[6] = 0
        keep_mask[7] = 0
        keep_mask[8] = 1
        keep_mask[9] = 1
        casename = "Fix 8 9"

    elif Pa_ind == 6:
        keep_mask[6] = 1
        keep_mask[7] = 1
        keep_mask[8] = 1
        keep_mask[9] = 0
        casename = "Fix 6 7 8"

    elif Pa_ind == 7:
        keep_mask[6] = 1
        keep_mask[7] = 0
        keep_mask[8] = 1
        keep_mask[9] = 1
        casename = "Fix 6 8 9"

    elif Pa_ind == 8:
        keep_mask[6] = 1
        keep_mask[7] = 1
        keep_mask[8] = 0
        keep_mask[9] = 1
        casename = "Fix 6 7 9"
    
    elif Pa_ind == 9:
        keep_mask[6] = 0
        keep_mask[7] = 1
        keep_mask[8] = 1
        keep_mask[9] = 1
        casename = "Fix 7 8 9"
    
    else:
        keep_mask[6] = 1
        keep_mask[7] = 1
        keep_mask[8] = 1
        keep_mask[9] = 1
        casename = "Fix 6 7 8 9"

    print("mask: ", keep_mask)
    print("reference: ", x_gt*keep_mask)

    for HL in range(1):
        # Gen Samples:  Those functions comes from Guided_Cond_DDPM_Tools.py
        print("Gen-no-guidance: ")
        X_gen_cond, unnorm_cond_only = T2.gen_cond_samples(drag_cond, x_gt = x_gt, keep_mask = keep_mask, Repaint = Enable_mask)    # Conditinal Generation

        print("Gen-guidance: ")
        X_gen, unnorm = T2.gen_low_drag_samples(drag_cond, x_gt = x_gt, keep_mask = keep_mask, Repaint = Enable_mask)               # Guidance Conditional Generation

        print(X_gen_cond.shape)

        print("LOA: ", LOA)
        print("X_shape:", unnorm.shape)

        UorF = 1 # Use Speed or Fn as condition 1: Fn, 0: Speed
        print("condition: ", dim_d[0])
        Rt_guidance, CT = T2.Predict_Drag(unnorm, drag_cond, UorF = UorF)    # Drag_conditioning is [ToD, Fn(m/s), LOA (m), Ct]
        Drag_Guidance = np.mean(Rt_guidance)
        Ct_mean = np.mean(CT)

        Rt_no_guidance, CT_no = T2.Predict_Drag(unnorm_cond_only, drag_cond, UorF = UorF)    # Drag_conditioning is [ToD, Fn(m/s), LOA (m), Ct]
        Drag_no_Guidance = np.mean(Rt_no_guidance)
        Ct_no_mean = np.mean(CT_no)

        print('-----------------------------------------------------')
        print('Predicted Mean Drag Coefficient Ct of Guidance samples:', Ct_mean)
        print('Predicted Mean Drag of Guidance samples: ' + str(Drag_Guidance) + ' N')
        print('Minimum Drag of Guidance samples: ' + str(np.amin(Rt_guidance)) + ' N')
        print('-----------------------------------------------------')
        print('Predicted Mean Drag Coefficient Ct of no Guidance samples:', Ct_no_mean)
        print('Predicted Mean Drag of no Guidance samples: ' + str(Drag_no_Guidance) + ' N')
        print('Minimum Drag of no Guidance samples: ' + str(np.amin(Rt_no_guidance)) + ' N')

        '''
        ===================================================
        Feasibility Checking - Result of Guidance Conditional Model
        ==================================================
        '''
        print('Checking Feasibility of Samples')

        x_samples = X_gen
        LOAs = np.repeat(LOA , x_samples.shape[0])
        x_samples = np.hstack((LOAs[:,np.newaxis], x_samples))

        for i in range(0,len(x_samples)):
            
            x_samples[i,idx_BB] = (x_samples[i,idx_BB] + 0.5) // 1 #int rounds to 1 or 0
            x_samples[i,idx_SB] = (x_samples[i,idx_SB] + 0.5) // 1 #int rounds to 1 or 0
            
            
            x_samples[i,idx_BBFactors] = x_samples[i,idx_BB] * x_samples[i,idx_BBFactors] 
            x_samples[i,idx_SBFactors] = x_samples[i,idx_SB] * x_samples[i,idx_SBFactors]

        #Check the constraint violations for the sampled designs
        constraints = []
        sum_violation = []
        cons = []
        valid_idx = []
        # check guidance result
        for i in tqdm(range(0,len(x_samples))):
            hull = HP(x_samples[i])
            constraints.append(hull.input_Constraints())
            cons.append(constraints[i] > 0)
            if sum(cons[i]) == 0:
                valid_idx.append(i)
                #hull.Calc_VolumeProperties(NUM_WL = 101, PointsPerWL = 1000)
            sum_violation.append(sum(cons[i]))

        print("Fesible design with guidance: ", len(valid_idx))

        # Calculate the BOA and Dd / under guidance 
        sample_BOA = []
        sample_Dd = []
        idx_to_remove = []

        for i in tqdm(range(0,len(valid_idx))): # calculate the performance of the generated samples 
            hull = HP(x_samples[valid_idx[i]]) 
            # print(i)
            try:
                Z = hull.Calc_VolumeProperties(NUM_WL = 101, PointsPerWL = 1000)
                BOA = max(hull.Calc_Max_Beam_midship(), hull.Calc_Max_Beam_PC())
                sample_BOA.append(BOA)
                sample_Dd.append(hull.Dd)
            except:
                print('error at hull {}'.format(i))
                idx_to_remove.append(i)
                continue

        #Remove the samples that failed to calculate volume properties
        valid_idx = np.delete(valid_idx, idx_to_remove)
        print(len(valid_idx))

        '''
        ===================================================
        Feasibility Checking - Result of Conditional Model
        ==================================================
        '''
        x_samples = X_gen_cond
        LOAs = np.repeat(LOA , x_samples.shape[0])
        x_samples = np.hstack((LOAs[:,np.newaxis], x_samples))

        for i in range(0,len(x_samples)):
            
            x_samples[i,idx_BB] = (x_samples[i,idx_BB] + 0.5) // 1 #int rounds to 1 or 0
            x_samples[i,idx_SB] = (x_samples[i,idx_SB] + 0.5) // 1 #int rounds to 1 or 0
            
            
            x_samples[i,idx_BBFactors] = x_samples[i,idx_BB] * x_samples[i,idx_BBFactors] 
            x_samples[i,idx_SBFactors] = x_samples[i,idx_SB] * x_samples[i,idx_SBFactors]

        #Check the constraint violations for the sampled designs
        constraints_1 = []
        sum_violation_1 = []
        cons_1 = []
        valid_idx_1 = []
        # check guidance result
        for i in tqdm(range(0,len(x_samples))):
            hull = HP(x_samples[i])
            constraints_1.append(hull.input_Constraints())
            cons_1.append(constraints_1[i] > 0)
            if sum(cons_1[i]) == 0:
                valid_idx_1.append(i)
                #hull.Calc_VolumeProperties(NUM_WL = 101, PointsPerWL = 1000)
            sum_violation_1.append(sum(cons_1[i]))

        print("Fesible design with no guidance: ", len(valid_idx_1))

        # Calculate the BOA and Dd / without guidance 
        sample_BOA_ng = []
        sample_Dd_ng = []
        idx_to_remove_ng = []

        for i in tqdm(range(0,len(valid_idx_1))): # calculate the performance of the generated samples 
            hull = HP(x_samples[valid_idx_1[i]]) 
            # print(i)
            try:
                Z = hull.Calc_VolumeProperties(NUM_WL = 101, PointsPerWL = 1000)
                BOA = max(hull.Calc_Max_Beam_midship(), hull.Calc_Max_Beam_PC())
                sample_BOA_ng.append(BOA)
                sample_Dd_ng.append(hull.Dd)
            except:
                print('error at hull {}'.format(i))
                idx_to_remove_ng.append(i)
                continue

        #Remove the samples that failed to calculate volume properties
        valid_idx_1 = np.delete(valid_idx_1, idx_to_remove_ng)
        print(len(valid_idx_1)) 

        '''
        ===================================================
        Error Calculation / Performance Evaluation
        ==================================================
        '''
        # The error of result of guidance conditional model
        print('Caclculating Error in Samples:')

        if len(valid_idx_1) == 0:
            sample_RT = 0.00000001
            sample_Ct = 0.00000001
        else:
            sample_RT = Rt_guidance[valid_idx]
            sample_Ct = CT[valid_idx]
        
        sample_BOA = np.array(sample_BOA)   # Beam 
        sample_Dd = np.array(sample_Dd)     # Depth

        CTMAPE = np.mean(np.abs(sample_Ct - Ships[0,4])/sample_Ct)*100
        print('CT with guidance MAPE: {}%'.format(CTMAPE))

        BOAMEAP = np.mean(np.abs(sample_BOA - Ships[0,1])/Ships[0,1])*100
        print('Beam MEAP: {}%'.format(BOAMEAP))

        DdMEAP = np.mean(np.abs(sample_Dd - Ships[0,3])/Ships[0,3])*100
        print('Depth MEAP: {}%'.format(DdMEAP))

        # The error of result of condition model
        if len(valid_idx_1) == 0:
            sample_RT_no = 0.00000001
            sample_no_Ct = 0.00000001
        else:
            sample_RT_no = Rt_no_guidance[valid_idx_1]
            sample_no_Ct = CT_no[valid_idx_1]

        sample_BOA_no = np.array(sample_BOA_ng)   # Beam 
        sample_Dd_no = np.array(sample_Dd_ng)     # Depth
        CTMAPE_no = np.mean(np.abs(sample_no_Ct - Ships[0,4])/sample_no_Ct)*100
        print('CT without guidance MAPE: {}%'.format(CTMAPE_no))

        BOAMEAP_no = np.mean(np.abs(sample_BOA_no - Ships[0,1])/Ships[0,1])*100
        print('Beam MEAP: {}%'.format(BOAMEAP_no))

        DdMEAP_no = np.mean(np.abs(sample_Dd_no - Ships[0,3])/Ships[0,3])*100
        print('Depth MEAP: {}%'.format(DdMEAP_no))

        print("Expect value: ", Ct )
        print("Mean of Ct with guidance: ",np.mean(sample_Ct))
        print("Variance of Ct with guidance: ", np.var(CT))
        print("Error to expect value: ", np.abs(Ct - np.mean(sample_Ct)) )

        print("Mean of Ct without guidance: ", np.mean(CT_no))
        print("Variance of Ct without guidance: ", np.var(CT_no))
        print("Error to expect value: ", np.abs(Ct - Ct_no_mean) )  

        '''
        ===================================================
        Visualization
        ==================================================
        '''
        bin_num = 20
        fig, axs = plt.subplots(1, 2, figsize=(12, 5)) 
        counts, edges = np.histogram(sample_Ct, bins=bin_num)
        print("Expect value: ", Ct )
        print("Mean of Ct with guidance: ",np.mean(sample_Ct))
        print("Variance of Ct with guidance: ", np.var(CT))
        print("Error to expect value: ", np.abs(Ct - np.mean(sample_Ct)) )

        counts_2, edges_2 = np.histogram(CT_no, bins = bin_num)
        print("Mean of Ct without guidance: ", np.mean(CT_no))
        print("Variance of Ct without guidance: ", np.var(CT_no))
        print("Error to expect value: ", np.abs(Ct - Ct_no_mean) )

        axs[0].bar(edges[:-1], counts, width=np.diff(edges), edgecolor='black', align='edge')
        axs[0].axvline(x=Ct, color='red', linestyle='--', label=f'Expect value: {Ct}')
        axs[0].axvline(x=np.mean(sample_Ct), color='blue', linestyle='--', label=f'Mean: {Ct_mean}')
        axs[0].set_title('Ct with guidance')
        axs[0].set_xlabel('Value')
        axs[0].set_ylabel('Density')
        axs[0].legend()

        axs[1].bar(edges_2[:-1], counts_2, width=np.diff(edges_2), edgecolor='black', align='edge')
        axs[1].axvline(x=Ct, color='red', linestyle='--', label=f'Expect value: {Ct}')
        axs[1].axvline(x=np.mean(CT_no), color='blue', linestyle='--', label=f'Mean: {Ct_no_mean}')
        axs[1].set_title('Ct without guidance')
        axs[1].set_xlabel('Value')
        axs[1].set_ylabel('Density')
        axs[1].legend()

        plt.tight_layout()
        Address = "./result/image/"
        plt.savefig( Address + casename , bbox_inches='tight', dpi=300)  # 保存为 JPEG

        '''
        ===================================================
        Result Collection and Output
        ==================================================
        '''
        np.save(Address + casename + '_Conditioning_Only_RT_DesVec.npy',X_gen_cond[valid_idx_1])
        np.save(Address + casename + '_Drag_Guidance_DesVec.npy',X_gen[valid_idx])
        np.save(Address + casename + '_Ct_pred.npy',CT[valid_idx])
        result_no_guidance = {
                "Type of generation" : "Condition only",
                "Fixed Parameter indicator": Pa_ind,
                "Drag Coefficient Mean": Ct_no_mean,
                "Drag Coefficient (Ct) Variance": np.var(CT_no),
                "Drag Resistance (Rt) Mean (N)": Drag_no_Guidance,
                "CT MAPE (%)": CTMAPE_no,
                "Beam MAPE (%)": BOAMEAP_no,
                "Depth MAPE (%)": DdMEAP_no,
                "Number_of_Feasible_Results_(512_total)": len(valid_idx_1),
                "Error": np.abs(Ct - Ct_no_mean),
                "HL": HL,
                "casename": casename
                }

        result_guidance = {
                "Type of generation" : "Condition + Guidance",
                "Fixed Parameter indicator": Pa_ind,
                "Drag Coefficient Mean": Ct_mean,
                "Drag Coefficient (Ct) Variance": np.var(CT),
                "Drag Resistance (Rt) Mean (N)": Drag_Guidance,
                "CT MAPE (%)": CTMAPE,
                "Beam MAPE (%)": BOAMEAP,
                "Depth MAPE (%)": DdMEAP,
                "Number_of_Feasible_Results_(512_total)": len(valid_idx),
                "Error": np.abs(Ct - np.mean(sample_Ct)),
                "HL": HL,
                "casename": casename
                }

        results.append(result_no_guidance)
        results.append(result_guidance)

        df = pd.DataFrame(results)
        output_file = "H_V_Combinations.xlsx"
        df.to_excel(Address + output_file, index=False)


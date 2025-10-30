import sys

sys.path.append('./tools')
sys.path.append('./data')

import os
import numpy as np
from tqdm import tqdm
import torch
import Guided_Cond_DDPM_Tools as GC_DDPM
import pandas as pd
from gen_tools import design_generation
import itertools

def main():
        np.set_printoptions(suppress=True) # don't use scientific notation
        #Step 1: Load in the data # the Design Paramaters Set
        DesVec = np.load('./data/DesVec_82k.npy', allow_pickle=True)

        DesVec_neg = np.load('./data/Negative_DesVec_82k.npy', allow_pickle=True)

        idx_BBFactors = [33,34,35,36,37]
        idx_BB = 31

        idx_SBFactors = [38,39,40,41,42,43,44]
        idx_SB = 32

        for i in range(0,len(DesVec)):
                DesVec[i,idx_BBFactors] = DesVec[i,idx_BB] * DesVec[i,idx_BBFactors] 
                DesVec[i,idx_SBFactors] = DesVec[i,idx_SB] * DesVec[i,idx_SBFactors]
        Y = np.load('./data/GeometricMeasures.npy', allow_pickle=True) # 12 Geometry preporties 
        LenRatios = np.load('./data/Length_Ratios.npy', allow_pickle=True) 
        X_LIMITS = np.load('./data/X_LIMITS.npy') # Range of each parameters
        X_lower_lim = [X_LIMITS[:,0].tolist()]                   
        X_upper_lim = [X_LIMITS[:,1].tolist()]
        #Set up Conditioning Vectors:
        num_WL_Steps = 101
        VolVec = np.log10(Y[:,1*num_WL_Steps:2*num_WL_Steps]) # volume 
        idx = np.where(np.isnan(VolVec))
        VolVec[idx] = -6.0 #fix nan to dummy value
        DdVec = DesVec[:,4] # Depth 
        BOAVec = np.amax(LenRatios[:,1:3], axis=1) # Beam
        # print(BOAVec.shape) 
        Cw = np.load('./data/Cw_82k.npy', allow_pickle=True)
        DDPM_C_Dict = {
                'xdim' : len(DesVec[0]) -1,             # Dimension of parametric design vector
                'datalength': len(DesVec),           # number of samples
                'X_LL' : X_lower_lim,           # lower limits of parametric design vector variables
                'X_UL' : X_upper_lim,
                'ydim': 0,                       # Number of objectives
                'cdim': 6,                      # number of conditioning inputs
                'gamma' : 0.7,                  # 0.2 weight of feasibility guidance for guided sampling
                'lambda': [0.7],                 # weight of drag  guidance for guided sampling
                'tdim': 128,                    # 128 dimension of latent variable
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
        # print('Generating Hulls')
        LOA = Ships[0,0] #in meters
        BoL = Ships[0,1]/LOA #beam to length ratio
        ToD = Ships[0,2]/Ships[0,3] #Draft to depth ratio
        DoL = Ships[0,3]/LOA #Depth to length ratio
        Ct = Ships[0, 4] # Wave drag coefficient
        Fn = Ships[0,5]  #  12.86 #m/s  = 25 knots
        dim_d = np.array([[ToD, Fn, LOA, Ct, DoL, BoL]]) #Drag_conditioning is [ToD, Fn(m/s), LOA (m), Ct]
        drag_cond = np.repeat(dim_d, num_samples, axis=0) #reapeat 512 
   
        # x_gt = torch.tensor(reference_design, dtype=torch.float32).to('cuda:0')
        x_gt = torch.tensor([4.96634266e-01  ,4.51955760e-01  , BoL,
        DoL,  5.71478128e-01,  1.43758851e-01,  1.31093639e-01,
        3.29564095e+00,  3.16418976e-01,  -3.14146566e-01,  3.54170799e-01,
        4.29022789e-01,  1.32593811e-01,  6.56113803e-01,  1.69787741e+00,
        -1.84923196e+00,  3.88996124e-01, -3.26323986e-01,  1.01175034e+01,
        0.00000000e+00,  0.00000000e+00, -2.73985386e-01,  4.10849154e-01,
        2.80871749e-01, -2.13681006e+00,  1.77291155e+00,  1.66059315e+00,
        7.07736760e-02,  4.26586211e-01,  2.07895815e-01,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00 , 0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00], dtype=torch.float32)
        '''
        ===================================================
        Configurate RaPaint
        ==================================================
        '''
        #----------------------------------
        # # print('GT: ', x_gt)
        # Mask configuration
        # 0-5 :Principle Dimensions 
        # 6-9: Midship Cross Section 
        # 10-18: Bow Geometry 
        # 19-29: Stern Geometry 
        # 30-43: Bulb Geometries
        #----------------------------------
        start_ind = [ 6, 10, 19, 30]
        end_ind = [10, 19, 30, 44]
        keep_mask = torch.zeros(x_gt.shape).to('cuda:0')
        keep_mask[9:10] = 1
        Enable_mask = 1     # Enable RePaint

        # # print("mask:", keep_mask)
        # # print("temp: ", x_gt*keep_mask)

        # normalization value of reference design values:
        x_gt = x = 2.0*(x_gt-T2.data_norm.X_LL_Scaled)/(T2.data_norm.X_UL_Scaled- T2.data_norm.X_LL_Scaled) - 1.0
        print('x_gt',x_gt)
        x_gt = x_gt.to(torch.float32).to('cuda:0')

        '''
        ===================================================
        Main Loop
        ==================================================
        '''
        results = []
        # Task_names = ["Single_Parameter", "Single_Component", "Mul_Parameter", " Original"]
        Task_names = ["Single_Parameter"]
        base_path = './results/Sensitive_test'
        test_rounds = 10 #20
        test_P = 6
        gamma = [0,0.3,0.7, 1]
        lamb = [0, 0.3,0.7, 1]
        U = [20, 30]
        combinations = list(itertools.product(gamma, lamb, U))
        
        for Task_name in Task_names:
                for i in range(test_rounds):
                        result_sub = []
                        for comb in combinations:     
                                print(f"Round {i+1}")
                                round_path = os.path.join(base_path, Task_name, f'round_{i+1}')
                                os.makedirs(round_path, exist_ok=True)  
                                T2.DDPM_Dict['gamma'] = comb[0]
                                T2.DDPM_Dict['lambda'] = comb[1]
                                T2.gamma = comb[0]
                                T2.lam = comb[1]
                                U = comb[2]
                                print(f"gamma: {T2.gamma}, lam: {T2.lam}, U: {U}")
                                if Task_name == "Single_Parameter":
                                        for Pa_ind in range(0, 44):
                                                if Pa_ind != test_P:
                                                        continue
                                                # Reset the mask and configure new one
                                                print(f"Fixing P {Pa_ind}")
                                                keep_mask = torch.zeros(keep_mask.shape).to('cuda:0')
                                                keep_mask[Pa_ind] = 1
                                                case_name = f'Fix_P_{Pa_ind}_gamma_{T2.gamma}_lam_{T2.lam}_U_{U}'
                                                Guide_Gen = True
                                                result_guidance = design_generation(DDPM_Env=T2, drag_cond=drag_cond, x_gt=x_gt, 
                                                                        keep_mask=keep_mask, Enable_mask=Enable_mask, 
                                                                        dim_d=dim_d,Ships =Ships, Pa_ind =Pa_ind, 
                                                                        save_path =round_path , case_name=case_name,
                                                                        Guide_Gen=Guide_Gen, Resample_t =U)
                                                result_guidance["lam"] = T2.lam
                                                result_guidance["gamma"]= T2.gamma
                                                result_guidance["U"]= U
                                                results.append(result_guidance)
                                                result_sub.append(result_guidance)
                                elif Task_name == "Mul_Parameter":
                                        for Pa_ind in range(7, 44):
                                                # Reset the mask and configure new one
                                                print(f"Fixing P 6 to P {Pa_ind}")
                                                keep_mask = torch.zeros(keep_mask.shape).to('cuda:0')
                                                keep_mask[6:Pa_ind+1] = 1
                                                case_name = f'Fix_P6_to_P{Pa_ind}'
                                                Guide_Gen = True
                                                result_guidance = design_generation(DDPM_Env=T2, drag_cond=drag_cond, x_gt=x_gt, 
                                                                        keep_mask=keep_mask, Enable_mask=Enable_mask, 
                                                                        dim_d=dim_d,Ships =Ships, Pa_ind =Pa_ind, 
                                                                        save_path =round_path , case_name=case_name,
                                                                        Guide_Gen=Guide_Gen)
                                                results.append(result_guidance)
                                                result_sub.append(result_guidance)
                                elif Task_name == "Single_Component":
                                        for Com_ind in range(len(start_ind)):
                                                # Reset the mask and configure new one
                                                print(f"Fixing Component {Com_ind}")
                                                keep_mask = torch.zeros(keep_mask.shape).to('cuda:0')
                                                print(f'fix component {Com_ind} from ind {start_ind[Com_ind]} to {end_ind[Com_ind]-1}')
                                                keep_mask[start_ind[Com_ind]:end_ind[Com_ind]] = 1
                                                case_name = f'Fix_Component_{Com_ind}'
                                                Guide_Gen = True
                                                result_guidance = design_generation(DDPM_Env=T2, drag_cond=drag_cond, x_gt=x_gt, 
                                                                        keep_mask=keep_mask, Enable_mask=Enable_mask, 
                                                                        dim_d=dim_d,Ships =Ships, Pa_ind =Com_ind, 
                                                                        save_path =round_path , case_name=case_name,
                                                                        Guide_Gen=Guide_Gen)
                                                results.append(result_guidance)
                                                result_sub.append(result_guidance)
                                elif Task_name == "Original":
                                        Enable_mask = False
                                        case_name = "pre-trained_model"
                                        Pa_ind = 0
                                        Guide_Gen = True
                                        result_guidance = design_generation(DDPM_Env=T2, drag_cond=drag_cond, x_gt=x_gt, 
                                                                        keep_mask=keep_mask, Enable_mask=Enable_mask, 
                                                                        dim_d=dim_d,Ships =Ships, Pa_ind =Pa_ind, 
                                                                        save_path =round_path , case_name=case_name,
                                                                        Guide_Gen=Guide_Gen)
                                        results.append(result_guidance)
                                        result_sub.append(result_guidance)
                        df = pd.DataFrame(result_sub)
                        output_file = round_path+ f"_Summary_table_{Task_name}.xlsx"
                        df.to_excel(output_file, index=False)
                        del result_sub

                df = pd.DataFrame(results)
                save_path = base_path
                output_file = save_path+ f"/Summary_table_{Task_name}.xlsx"
                df.to_excel(output_file, index=False)

if __name__ == '__main__':
        main()


import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from HullParameterization import Hull_Parameterization as HP
import os 
import multiprocessing as mp
import time

idx_BBFactors = [33,34,35,36,37]
idx_BB = 31

idx_SBFactors = [38,39,40,41,42,43,44]
idx_SB = 32

def worker(args):
    hull = HP(args) 
    try:
        _ = hull.Calc_VolumeProperties(NUM_WL = 101, PointsPerWL = 1000)    # must run this code before the code following!
        BOA = max(hull.Calc_Max_Beam_midship(), hull.Calc_Max_Beam_PC())
        Dd = hull.Dd
        val = True
    except:
        BOA = 0
        Dd = 0
        val = False
    return BOA, Dd, val

def design_generation(DDPM_Env, drag_cond, x_gt, keep_mask, Enable_mask, 
                      dim_d,Ships, Pa_ind, save_path, case_name,Guide_Gen, Resample_t=20):
    LOA = Ships[0,0] #in meters
    Ct = Ships[0, 4] # Expected Wave drag coefficient
    start_time = time.time()
    if Guide_Gen:
        print("Gen-guidance: ")
        X_gen, unnorm = DDPM_Env.gen_low_drag_samples(drag_cond, x_gt = x_gt, keep_mask = keep_mask, Repaint = Enable_mask, Resample_t=Resample_t)               # Guidance Conditional Generation
    else:
        print("Gen-no-guidance: ")
        X_gen, unnorm = DDPM_Env.gen_cond_samples(drag_cond, x_gt = x_gt, keep_mask = keep_mask, Repaint = Enable_mask)    # Conditinal Generation
    # print("LOA: ", LOA)
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Elapsed time: {elapsed:.4f} seconds")
    print("X_shape:", unnorm.shape)

    UorF = 1 # Use Speed or Fn as condition 1: Fn, 0: Speed
    print("condition: ", dim_d[0])
    Rt_guidance, CT = DDPM_Env.Predict_Drag(unnorm, drag_cond, UorF = UorF)    # Drag_conditioning is [ToD, Fn(m/s), LOA (m), Ct]
    Drag_Guidance = np.mean(Rt_guidance)
    Ct_mean = np.mean(CT)
    print('-----------------------------------------------------')
    print('Predicted Mean Drag Coefficient Ct of Guidance samples:', Ct_mean)
    print('Predicted Mean Drag of Guidance samples: ' + str(Drag_Guidance) + ' N')
    print('Minimum Drag of Guidance samples: ' + str(np.amin(Rt_guidance)) + ' N')

    '''
    ==================================================
    Feasibility Checking - Result of Guidance Conditional Model
    ==================================================
    '''
    print('Checking Feasibility of Samples')

    x_samples = X_gen
    LOAs = np.repeat(LOA , x_samples.shape[0])
    x_samples = np.hstack((LOAs[:,np.newaxis], x_samples))
    X_gen = x_samples
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
    mul_task = True
    if mul_task:
        args_list = x_samples[valid_idx] 
        # Set up pool
        n_workers = 0
        n_workers = n_workers or mp.cpu_count() - 2
        print(f"Using {n_workers} parallel workers")
        with mp.Pool(processes=n_workers) as pool:
            results = list(tqdm(pool.imap(worker, args_list), total=len(args_list), desc="Validate designs"))
        # Unpack results
        if results:
            BOA_list, Dd_list, val_list = zip(*results)
        else:
            BOA_list, Dd_list, val_list = [], [], []
        sample_BOA, sample_Dd, val = np.array(BOA_list), np.array(Dd_list), np.array(val_list)
        valid_idx = np.delete(valid_idx, np.where(val == 0)[0])
    else: 
        for i in tqdm(range(0,len(valid_idx))): # calculate the performance of the generated samples 
            hull = HP(x_samples[valid_idx[i]]) 
            try:
                _ = hull.Calc_VolumeProperties(NUM_WL = 101, PointsPerWL = 1000)    # must run this code before the code following!
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
    Error Calculation / Performance Evaluation
    ==================================================
    '''
    # The error of result of guidance conditional model
    print('Caclculating Error in Samples:')

    if len(valid_idx) == 0:
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

    print("Expect value: ", Ct )
    print("Mean of Ct with guidance: ",np.mean(sample_Ct))
    print("Variance of Ct with guidance: ", np.var(CT))
    print("Error to expect value: ", np.abs(Ct - np.mean(sample_Ct)) )

    '''
    ===================================================
    Visualization
    ==================================================
    '''
    bin_num = 20
    plt.figsize=(12, 5) 
    counts, edges = np.histogram(sample_Ct, bins=bin_num)
    # print("Expect value: ", Ct )
    # print("Mean of Ct with guidance: ",np.mean(sample_Ct))
    # print("Variance of Ct with guidance: ", np.var(CT))
    # print("Error to expect value: ", np.abs(Ct - np.mean(sample_Ct)) )

    plt.bar(edges[:-1], counts, width=np.diff(edges), edgecolor='black', align='edge')
    plt.axvline(x=Ct, color='red', linestyle='--', label=f'Expect value: {Ct}')
    plt.axvline(x=np.mean(sample_Ct), color='blue', linestyle='--', label=f'Mean: {Ct_mean}')
    plt.title('Ct with guidance')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    
    img_path = os.path.join(save_path,"image")
    os.makedirs(img_path, exist_ok=True)  
    file_name = os.path.join(img_path, f"{case_name}.png")
    plt.savefig( file_name, bbox_inches='tight', dpi=300)  
    '''
    ===================================================
    Result Collection and Output
    ==================================================
    '''
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
            "Elapsed": elapsed
            }
    file_name = os.path.join(save_path, f'{case_name}_Drag_Guidance_DesVec.npy')
    np.save(file_name, X_gen[valid_idx])
    file_name = os.path.join(save_path, f'{case_name}_Ct_pred.npy')
    np.save(file_name, CT[valid_idx])
    return  result_guidance

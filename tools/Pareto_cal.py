import numpy as np
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm 
import multiprocessing as mp
import pandas as pd
import os 
sys.path.append('./tools')
sys.path.append('./data')
from HullParameterization import Hull_Parameterization as HP

def worker(args):
    hull = HP(args) 
    try:
        Z = hull.Calc_VolumeProperties(NUM_WL = 101, PointsPerWL = 1000)    # must run this code before the code following!
        vol = HP.interp(hull.Volumes, Z, 11.28)
    except:
        vol = 0
    return vol

def pareto_front(X, Y, maximize_x=False, maximize_y=False):

    data = np.stack([X, Y], axis=1)
    is_efficient = np.ones(data.shape[0], dtype=bool)
    for i, c in enumerate(data):
        if is_efficient[i]:
            if maximize_x:
                worse_x = data[:, 0] <= c[0]
            else:
                worse_x = data[:, 0] >= c[0]
            if maximize_y:
                worse_y = data[:, 1] <= c[1]
            else:
                worse_y = data[:, 1] >= c[1]
            is_efficient[worse_x & worse_y] = False
            is_efficient[i] = True
    return np.where(is_efficient)[0]

def get_component(i, start_ind, end_ind):
    components = ["Principal Dimension", "Midship Cross Section", "Bow", "Stern", "Bulb"]
    for name, start, end in zip(components, start_ind, end_ind):
        if start <= i < end:
            return name
    return "Out of range"

def main():
    save_img = True
    group = False
    selected = True
    start_ind = [ 0, 6, 10, 19, 30]
    end_ind = [6, 10, 19, 30, 43]
    Component_list = ["Midship Cross Section", "Bow", "Stern", "Bulb"]
    pre_trained_lines = []
    pre_trained_lables = []
    ptr_trained_Ct = []
    pre_trained_vol = []
    for task in ["Original",'Single_Parameter', 'Single_Component', 'Mul_Parameter']:
        i = 0
        print(task + " in progress")
        
        if pre_trained_lines != []:
            pareto_lines = pre_trained_lines.copy()
            lables = pre_trained_lables.copy()
            Ct_record = ptr_trained_Ct.copy()
            vol_record = pre_trained_vol.copy()
        else:
            pareto_lines = []
            Ct_record = []
            vol_record = []
            lables = []
        while(True):
            if task == 'Single_Parameter':
                Num_case = 44
                gen_dev_path = f'./results_2/Single_Parameter/round_3/Fix_P_{i}_Drag_Guidance_DesVec.npy'
                Ct_path = f'results_2/Single_Parameter/round_3/Fix_P_{i}_Ct_pred.npy'
                img_name = f'Pareto_Ct_V_Fix_P_{i}.png'
                volume_path = f'./results_2/Single_Parameter/round_3/Fix_P_{i}_Volume.npy'
            elif task == 'Single_Component':
                Num_case = 4
                if i < len(Component_list): 
                    comp = Component_list[i]
                else: 
                    comp = ''
                gen_dev_path = f'results_2/Single_Component/round_3/Fix_Component_{i}_Drag_Guidance_DesVec.npy'
                Ct_path = f'results_2/Single_Component/round_3/Fix_Component_{i}_Ct_pred.npy'
                img_name = f'Pareto_Ct_V_Fix_{comp}.png'
                volume_path = f'./results_2/Single_Component/round_3/Fix_Component_{i}_Volume.npy'
            elif task == 'Mul_Parameter':
                Num_case = 36
                gen_dev_path = f'results_2/Mul_Parameter/round_4/Fix_P6_to_P{6+i+1}_Drag_Guidance_DesVec.npy'
                Ct_path = f'results_2/Mul_Parameter/round_4/Fix_P6_to_P{6+i+1}_Ct_pred.npy'
                img_name = f'Pareto_Ct_V_Fix_P6_to_P{6+i+1}.png'
                volume_path = f'./results_2/Mul_Parameter/round_4/Fix_P6_to_P{6+i+1}_Volume.npy'
            elif task == "Original":
                Num_case =1 
                gen_dev_path = r'./results_2/Original/round_2/pre-trained_model_Drag_Guidance_DesVec.npy'
                Ct_path = r'results_2/Original/round_2/pre-trained_model_Ct_pred.npy'
                img_name = f'Pareto_Ct_V_pre-trained_model.png'
                volume_path = r'./results_2/Original/round_2/pre-trained_model_Volume.npy'
            if i >= Num_case:
                break 
            else:
                i += 1
            if group: 
                lables.append(get_component(i-1, start_ind, end_ind))
            else:
                lables.append(img_name[12:-4])
            print(img_name+ " in progress")
            save_path = os.path.join('Pareto', task)
            os.makedirs(save_path, exist_ok=True)
            img_path = os.path.join(save_path, img_name)
            # if os.path.exists(img_path):
            #     continue
            # data load 
            x_samples = np.load(gen_dev_path)
            if x_samples.shape[0] < 10:
                continue 
            # print(f"size of x_sample: {x_samples.shape}")
            idx_BBFactors = [33,34,35,36,37]
            idx_BB = 31
            idx_SBFactors = [38,39,40,41,42,43,44]
            idx_SB = 32
            for ind in range(0,len(x_samples)):
                x_samples[ind,idx_BBFactors] = x_samples[ind,idx_BB] * x_samples[ind,idx_BBFactors] 
                x_samples[ind,idx_SBFactors] = x_samples[ind,idx_SB] * x_samples[ind,idx_SBFactors]
                
            Ct = np.load(Ct_path)
            Ct = np.squeeze(Ct)
            # Ct = np.abs(Ct - 0.0003) / Ct * 100
            # print("Ct_raw min:", Ct.min(), "max:", Ct.max(), "mean:", Ct.mean())
            
            # ========== 计算体积 ==========
            if os.path.exists(volume_path):
                sample_vol = np.load(volume_path)
            else:
                args_list = x_samples  
                sample_vol = []
                # Set up pool
                n_workers = 0
                n_workers = n_workers or mp.cpu_count() - 2
                # print(f"Using {n_workers} parallel workers")
                with mp.Pool(processes=n_workers) as pool:
                    results = list(tqdm(pool.imap(worker, args_list), total=len(args_list), desc="Calculate vol"))
                sample_vol = np.array(results) 
                if not os.path.exists(volume_path):
                    np.save(volume_path, sample_vol)
            # Ct 越小越好，Vol 越大越好 → 即 minimize Ct, maximize Volume
            pareto_idx = pareto_front(Ct, sample_vol, maximize_x=False, maximize_y=True)
            Ct_record.append(Ct)
            vol_record.append(sample_vol)
            pareto_lines.append((Ct[pareto_idx], sample_vol[pareto_idx]))
            
            if task == "Original":
                pre_trained_lines.append((Ct[pareto_idx], sample_vol[pareto_idx]))
                ptr_trained_Ct.append(Ct)
                pre_trained_vol .append(sample_vol)
                pre_trained_lables.append(img_name[12:-4])
            print(f"Pareto front has {len(pareto_idx)} points.")
            if save_img:
                plt.figure(figsize=(8, 5))
                plt.scatter(Ct, sample_vol, s=15, alpha=0.5, label='All samples')
                plt.scatter(Ct[pareto_idx], sample_vol[pareto_idx], color='red', s=40, label='Pareto front')
                sorted_idx = np.argsort(Ct[pareto_idx])
                plt.plot(Ct[pareto_idx][sorted_idx], sample_vol[pareto_idx][sorted_idx], color='red', linewidth=1.5)

                plt.xlabel("MAPE in Resistance Coefficient (Ct)")
                plt.ylabel("Displacement Volume(m³)")
                plt.title("Pareto Front between Ct and Volume")
                plt.grid(True, linestyle='--', alpha=0.5)
                plt.legend()
                plt.tight_layout()
                plt.savefig(img_path)
                plt.close("all")
                plt.clf
                # plt.show()
        
        if i ==Num_case:
            plt.figure(figsize=(10, 6)) 
            if task == "Single_Parameter":
                if selected: 
                    select_indices = [0 ,31]
                    pareto_lines = [pareto_lines[i] for i in select_indices]
                    # print("Selected indices:", len(pareto_lines))
                    lables = [lables[i] for i in select_indices] 
                    Ct_record = [Ct_record[i] for i in select_indices]
                    vol_record = [vol_record[i] for i in select_indices]
                if group:
                    unique_labels = list(dict.fromkeys(lables))  
                    colors_map = plt.cm.turbo(np.linspace(0, 1, len(unique_labels)))
                    label_color_dict = {lab: col for lab, col in zip(unique_labels, colors_map)}
                    colors = np.array([label_color_dict[lab] for lab in lables])
                    seen = set()
                    lables_unique = []
                    for lab in lables:
                        if lab not in seen:
                            lables_unique.append(lab)
                            seen.add(lab)
                        else:
                            lables_unique.append(None)
                    lables = lables_unique

            N = len(pareto_lines) 
            # print(f"N: {N}")
            # colors = plt.cm.turbo(np.linspace(0, 1, N))
            colors = plt.cm.tab10(np.linspace(0, 1, N)) 
            for j, (Ct_pareto, vol_pareto) in enumerate(pareto_lines):
                sorted_idx = np.argsort(Ct_pareto)
                Ct = Ct_record[j]
                sample_vol = vol_record[j]
                if lables[j] == 'pre-trained_model':
                    plt.plot(Ct_pareto[sorted_idx], vol_pareto[sorted_idx], marker='*', color=colors[j],
                            linestyle='-', linewidth=5, zorder=10)
                    plt.scatter(Ct, sample_vol, s=15, alpha=0.5, label=lables[j]+"_samples")
                else:
                    # plt.plot(Ct_pareto[sorted_idx], vol_pareto[sorted_idx], marker='o', 
                    #         linestyle='-', linewidth=1.5, label=None)
                    plt.plot(Ct_pareto[sorted_idx], vol_pareto[sorted_idx], marker='o', 
                                linestyle='-', linewidth=1.5,    color=colors[j])
                    plt.scatter(Ct, sample_vol, s=15, alpha=0.5, label = "Fix single parameter samples", color=colors[j])
            plt.xlabel("Resistance Coefficient (Ct)")
            plt.ylabel("Displacement Volume(m³)")
            plt.title("Pareto Front between Ct and Volume")
            plt.grid(True, linestyle='--', alpha=0.5)
                
            if task != "Single_Component":
                location = 'lower right'
            else:
                location = 'upper left'
            plt.legend(ncol=1,loc=location, fontsize=10)
            plt.tight_layout()
            summary_img_name = f'Pareto_Ct_V_{task}_All_Cases.png'  
            summary_img_path = os.path.join(save_path, summary_img_name)
            print("Saving summary Pareto front image to:", summary_img_path)
            plt.savefig(summary_img_path)       
            # plt.show( )
            plt.close("all")
            plt.clf
        
    
if __name__ == "__main__":
    main()
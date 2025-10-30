import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os 
import prd_score as prd 

def gaussian_kernel(x, y, sigma=1.0):
    xx = np.sum(x**2, axis=1).reshape(-1, 1)
    yy = np.sum(y**2, axis=1).reshape(1, -1)
    xy = np.dot(x, y.T)
    dist = xx + yy - 2 * xy
    return np.exp(-dist / (2 * sigma**2))

def compute_mmd(X, Y, sigma=1.0):
    Kxx = gaussian_kernel(X, X, sigma)
    Kyy = gaussian_kernel(Y, Y, sigma)
    Kxy = gaussian_kernel(X, Y, sigma)
    mmd = np.mean(Kxx) + np.mean(Kyy) - 2 * np.mean(Kxy)
    return np.sqrt(mmd)

def main():
    save_prd = True
    save_mmd = True 
    DesVec = np.load('./data/DesVec_82k.npy', allow_pickle=True)
    idx_BBFactors = [33,34,35,36,37]
    idx_BB = 31
    idx_SBFactors = [38,39,40,41,42,43,44]
    idx_SB = 32
    for i in range(0,len(DesVec)):
            DesVec[i,idx_BBFactors] = DesVec[i,idx_BB] * DesVec[i,idx_BBFactors] 
            DesVec[i,idx_SBFactors] = DesVec[i,idx_SB] * DesVec[i,idx_SBFactors]
    DesVec = DesVec[:, 1:]  # remove LOA 
    print(f"Size of DesVec : {DesVec.shape}")
    X = DesVec
    X = X[np.random.choice(len(X), 512,replace=False)]
    X_mmd = X
    
    result_path = './prd'
    os.makedirs(result_path, exist_ok=True)  
    img_path = os.path.join(result_path, 'img')
    os.makedirs(img_path, exist_ok=True)  
    
    prd_result= []
    mmd_result= []
    fig1_data = []
    fig2_data = []
    gen_dev_path = r'results_2\Original\round_1\pre-trained_model_Drag_Guidance_DesVec.npy'
    Y = np.load(gen_dev_path)[:,1:] # remove LOA 
    
    
    title = "PRD Curve of Pre-trained Model's Output"
    save_address = os.path.join(img_path, f"{title}.png")
    if save_mmd:
        mmd = compute_mmd(X_mmd, Y)
        result = {'mmd': mmd,
                  'task': title}
        mmd_result.append(result)
        print(title +" MMD:", mmd)
    if save_prd:
        precision, recall = pre_trained_prd = prd.compute_prd_from_embedding(X, Y)        
        prd_result.append((np.array(precision), np.array(recall)))
        fig1_data.append((np.array(precision), np.array(recall)))
        fig2_data.append((np.array(precision), np.array(recall)))
    
    label = ['Pre-trained Model']
    for i in range(44):
        print(f"Processing Parameter {i}")
        title = f"PRD Curve of Fixing Parameter {i}"
        save_address = os.path.join(img_path, f"{title}.png")
        gen_dev_path = f'./results_II/Single_Parameter/round_1/Fix_P_{i}_Drag_Guidance_DesVec.npy'
        Y = np.load(gen_dev_path)[:,1:] # remove LOA 
        Y = np.delete(Y, i, axis=1)
        X_reg = np.delete(X, i, axis=1)
        X_mmd_reg = np.delete(X_mmd, i, axis=1)
        if save_mmd:
            mmd = compute_mmd(X_mmd_reg, Y)
            result = {'mmd': mmd,
                  'task': title}
            mmd_result.append(result)
            print(title +" MMD:", mmd)
        if save_prd:
            if len(Y)!=512:
                if abs(len(Y)-len(X_reg))>50:
                    continue
                else:
                    X_reg = X_reg[np.random.choice(len(X_reg), len(Y),replace=False)]
            legend = ['Pre-trained Model', f'Fixing P {i}']
            precision, recall = result  = prd.compute_prd_from_embedding(X_reg, Y)
            prd_result.append((np.array(precision), np.array(recall)))
            fig1_data.append((np.array(precision), np.array(recall)))
            label.append(f'Fixing P {i}')
            prd.plot([pre_trained_prd, result],labels=legend,out_path=save_address, legend_loc='upper right')
    
    
    label_a= ['Pre-trained Model'] 
    Component_list = ["Midship Cross Section", "Bow", "Stern", "Bulb"]
    start_ind = [ 6, 10, 19, 30]
    end_ind = [10, 19, 30, 44]
    for i in range(4):
        print(f"Processing Component {Component_list[i]}")
        title = f"PRD Curve of Fixing {Component_list[i]}"
        save_address = os.path.join(img_path, f"{title}.png")
        gen_dev_path = f'results_II/Single_Component/round_1/Fix_Component_{i}_Drag_Guidance_DesVec.npy'
        Y = np.load(gen_dev_path)[:,1:] # remove LOA 
        Y = np.delete(Y, range(start_ind[i], end_ind[i]), axis=1)
        X_reg = np.delete(X, range(start_ind[i], end_ind[i]), axis=1)
        X_mmd_reg = np.delete(X_mmd, range(start_ind[i], end_ind[i]), axis=1)
        if save_mmd:
            mmd = compute_mmd(X_mmd_reg, Y)
            result = {'mmd': mmd,
                      'task': title}
            mmd_result.append(result)
            print(title +" MMD:", mmd)
        if save_prd:
            if len(Y)!=512:
                X_reg = X_reg[np.random.choice(len(X_reg), len(Y),replace=False)]
            legend = ['Pre-trained Model', f'Fixing {Component_list[i]}']
            precision, recall = result  = prd.compute_prd_from_embedding(X_reg, Y)
            prd.plot([pre_trained_prd, result], labels=legend, out_path=save_address, legend_loc='upper right')
            prd_result.append((np.array(precision), np.array(recall)))
            fig2_data.append((np.array(precision), np.array(recall)))
            label_a.append(f'Fixing {Component_list[i]}')

    for i in range(7,44):
        print(f"Processing Fixing P 6 to P {i}")
        title = f"PRD Curve of Fixing {i-6+1} Parameters"
        save_address = os.path.join(img_path, f"{title}.png")
        gen_dev_path = f'results_II/Mul_Parameter/round_6/Fix_P6_to_P{i}_Drag_Guidance_DesVec.npy'
        Y = np.load(gen_dev_path)[:,1:] # remove LOA 
        Y = np.delete(Y, range(6,i), axis=1)
        X_reg = np.delete(X, range(6,i), axis=1)
        X_mmd_reg = np.delete(X_mmd, range(6,i), axis=1)
        if save_mmd:
            mmd = compute_mmd(X_mmd_reg, Y)
            result = {'mmd': mmd,
                  'task': title}
            mmd_result.append(result)
            print(title +" MMD:", mmd)
        if save_prd:
            if len(Y)!=512:
                continue
            legend = ['Pre-trained Model', f'Fixing P 6 to P {i}']
            precision, recall = result   = prd.compute_prd_from_embedding(X_reg, Y)
            prd.plot([pre_trained_prd, result], labels=legend, out_path=save_address, legend_loc='upper right')
            prd_result.append((np.array(precision), np.array(recall)))
            
    if save_prd:
        save_address = os.path.join(img_path, "Fixing Single Parameter.png")
        prd.plot(fig1_data,labels=label, out_path=save_address,legend_loc='upper right')
        save_address = os.path.join(img_path, "Fixing Single Component.png")
        prd.plot(fig2_data, labels=label_a, out_path=save_address, legend_loc='upper right')
        file_name = os.path.join(result_path, 'PRD.npy')
        np.save(file_name, prd_result)
        
    if save_mmd:
        file_name = os.path.join(result_path, 'MMD.npy')
        np.save(file_name, mmd_result)
    print("Done")

if __name__ == "__main__":
    main()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

single_p_path = r'.\results_II\Summary_table_Single_Parameter.xlsx'
single_c_path = r'.\results_II\Summary_table_Single_Component.xlsx'
Pre_train_path= r'.\results_II\Summary_table_Original.xlsx'

single_p = False 
if single_p:
    df = pd.read_excel(single_p_path)
    headers = df.columns.tolist()
    P_ind = df['Fixed Parameter indicator']
    Ct_MAPE = df['CT MAPE (%)']
    Feasible_rate = df['Number_of_Feasible_Results_(512_total)']/512
    P_uni = np.unique(P_ind)
    print(f'all P: {P_uni}')
    P_Ct_MPAE_var = []
    P_Ct_MPAE_sd = []
    P_fr_std =[]
    for p in P_uni:
        P_Ct_MPAE = Ct_MAPE.values[np.where(P_ind == p)[0]] 
        P_Ct_MPAE = P_Ct_MPAE[np.where(P_Ct_MPAE<100)]
        P_Feasible_rate = Feasible_rate[np.where(P_ind == p)[0]] 
        
        vars = np.var(P_Ct_MPAE)
        std = np.std(P_Ct_MPAE, ddof=1)  
        fr_std = np.std(P_Feasible_rate, ddof=1)  
        
        P_fr_std.append(fr_std)
        P_Ct_MPAE_var.append(vars)
        P_Ct_MPAE_sd.append(std)
        print(f'P {p}: var-{vars} std-{std} fr_std{fr_std}')
        
    plt.figure(figsize=(10, 4))
    P_labels = [str(p) for p in P_uni]
    print(f'P_labels: {P_labels}')
    x = np.arange(len(P_uni))             # 参数索引
    width = 0.4  
    bars1 = plt.bar(x - width/2, P_Ct_MPAE_sd, width, label='Std. of Ct MAPE', color='skyblue', alpha=0.9)
    bars2 = plt.bar(x + width/2, P_fr_std, width, label='Std. of Feasibility Rate', color='salmon', alpha=0.9)
    
    plt.xlabel("Parameter Index (P)",fontsize=12)
    plt.ylabel("Standard deviation",fontsize=12)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.xticks(x, P_labels, rotation=45, ha='right',fontsize=10)  # rotation可调整倾斜角度
    # plt.yscale('log')
    plt.title("Standard Deviation of Ct MAPE and Feasibility Rate for Each Parameter",fontsize=12)
    plt.legend()
    plt.legend(fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    
single_c = True
if single_c: 
    df = pd.read_excel(single_c_path)
    headers = df.columns.tolist()
    P_ind = df['Fixed Parameter indicator']
    Ct_MAPE = df['CT MAPE (%)']
    Feasible_rate = df['Number_of_Feasible_Results_(512_total)']/512
    P_uni = np.unique(P_ind)
    print(f'all P: {P_uni}')
    P_Ct_MPAE_var = []
    P_Ct_MPAE_sd = []
    P_fr_std =[]
    for p in P_uni:
        P_Ct_MPAE = Ct_MAPE.values[np.where(P_ind == p)[0]] 
        P_Ct_MPAE = P_Ct_MPAE[np.where(P_Ct_MPAE<100)]
        P_Feasible_rate = Feasible_rate[np.where(P_ind == p)[0]] 
        
        vars = np.var(P_Ct_MPAE)
        std = np.std(P_Ct_MPAE, ddof=1)  
        fr_std = np.std(P_Feasible_rate, ddof=1)  
        
        P_fr_std.append(fr_std)
        P_Ct_MPAE_var.append(vars)
        P_Ct_MPAE_sd.append(std)
        print(f'P {p}: var-{vars} std-{std} fr_std{fr_std}')
        
    plt.figure(figsize=(8, 6))
    P_labels = ["Component"+str(p) for p in P_uni]
    P_labels = ["Midship Cross Section", "Bow", "Stern", "Bulb"]
    print(f'P_labels: {P_labels}')
    x = np.arange(len(P_uni))             # 参数索引
    width = 0.4  
    bars1 = plt.bar(x - width/2, P_Ct_MPAE_sd, width, label='Std. of Ct MAPE', color='skyblue', alpha=0.9)
    bars2 = plt.bar(x + width/2, P_fr_std, width, label='Std. of Feasibility Rate', color='salmon', alpha=0.9)
    for bar in bars1:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.0001, f'{height:.3f}', 
                ha='center', va='bottom', fontsize=10, color='blue')

    for bar in bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.0001, f'{height:.3f}', 
                ha='center', va='bottom', fontsize=10, color='darkred')
    # plt.plot(P_uni, P_Ct_MPAE_sd, marker='o', linestyle='-', linewidth=1.5, label = 'standard deviation of Ct MAPE')
    # plt.plot(P_uni, P_fr_std, marker='o', linestyle='-', linewidth=1.5,label = 'standard deviation of Feasibility Rate')
    plt.xlabel("Design components",fontsize=12)
    plt.ylabel("Standard deviation",fontsize=12)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.xticks(x, P_labels, rotation=0, ha='center',fontsize=10 )  # rotation可调整倾斜角度
    # plt.yscale('log')
    plt.title("Standard Deviation of Ct MAPE and Feasibility Rate for Each Components",fontsize=12)
    plt.legend()
    plt.legend(fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
    
Pre_train = True
if Pre_train:
    df = pd.read_excel(Pre_train_path)
    df = df[:30]
    headers = df.columns.tolist()
    P_ind = df['Fixed Parameter indicator']
    Ct_MAPE = df['CT MAPE (%)']
    Feasible_rate = df['Number_of_Feasible_Results_(512_total)']/512
    P_Ct_MPAE_mean = np.mean(Ct_MAPE)
    print(f'Pre-trained model Ct MAPE mean: {P_Ct_MPAE_mean}')
    P_Ct_MPAE_var = np.var(Ct_MAPE, ddof=1)
    P_Ct_MPAE_sd = np.std(Ct_MAPE, ddof=1)  
    P_fr_std = np.std(Feasible_rate, ddof=1)  
    print(f'var-{P_Ct_MPAE_var} std-{P_Ct_MPAE_sd} fr_std{P_fr_std}')
        
    plt.figure(figsize=(10, 6))
    P_labels = [str(p) for p in range(1, len(Ct_MAPE)+1)]
    # print(f'P_labels: {P_labels}')
    x = np.arange(len(P_labels))            # 参数索引
    width = 0.4  
    bars1 = plt.bar(x - width/2, Ct_MAPE, width, label=f'Std. ={P_Ct_MPAE_sd:.3f}', color='skyblue', alpha=0.9)
    plt.xlabel("Rounds",fontsize=12)
    plt.ylabel("Ct MAPE(%)",fontsize=12)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.xticks(x, P_labels, rotation=0, ha='center',fontsize=10 )  # rotation可调整倾斜角度
    plt.title("Ct MAPE of Pre-trained Model's Output",fontsize=12)
    plt.legend()
    plt.legend(fontsize=14,loc='upper right')
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    plt.figure(figsize=(10, 6))
    P_labels = [str(p) for p in range(len(Ct_MAPE))]
    # print(f'P_labels: {P_labels}')
    x = np.arange(len(P_labels))             # 参数索引
    width = 0.4  
    bars1 = plt.bar(x - width/2, Feasible_rate, width, label=f'Std. ={P_fr_std:.3f}', color='skyblue', alpha=0.9)
    plt.xlabel("Rounds",fontsize=12)
    plt.ylabel("Feasibility rate",fontsize=12)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.xticks(x, P_labels, rotation=0, ha='center',fontsize=10 )  # rotation可调整倾斜角度
    plt.title("Feasibility Rate of Pre-trained Model's Output",fontsize=12)
    plt.legend()
    plt.legend(fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    plt.show()
    
Pre_train_test = False
if Pre_train_test:
    df = pd.read_excel(Pre_train_path)
    headers = df.columns.tolist()
    P_ind = df['Fixed Parameter indicator']
    Ct_MAPE = df['CT MAPE (%)']
    Feasible_rate = df['Number_of_Feasible_Results_(512_total)']/512
    var_list= []
    x_labels = []
    for i in range(2,len(Ct_MAPE)+1):
        Ct_var = np.var(Ct_MAPE[:i])
        var_list.append(Ct_var)
        x_labels.append(str(i))
    
    plt.figure(figsize=(10, 4))
    x = np.arange(len(var_list)) 
    plt.plot(x, var_list, marker='o', linestyle='-', linewidth=1.5, label = 'standard deviation of Ct MAPE')
    # ✅ 计算最后一个值的 ±5%
    last_val = var_list[-1]
    upper = last_val * 1.05
    lower = last_val * 0.95

    # ✅ 画两条水平线
    plt.axhline(y=upper, color='red', linestyle='--', linewidth=1.2, label='+5%')
    plt.axhline(y=lower, color='red', linestyle='--', linewidth=1.2, label='-5%')
    plt.axhline(y=last_val, color='green', linestyle='--', linewidth=1.2, label='mean')

    plt.xlabel("Rounds",fontsize=12)
    plt.ylabel("Standard deviation",fontsize=12)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.xticks(x, x_labels, rotation=0, ha='center',fontsize=10 )  # rotation可调整倾斜角度
    # plt.yscale('log')
    plt.title(" Sampling round vs Std.",fontsize=12)
    plt.legend()
    plt.legend(fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
        
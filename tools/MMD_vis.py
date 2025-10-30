import numpy as np
import matplotlib.pyplot as plt

data_path = 'prd\MMD.npy'
mmd = np.load(data_path, allow_pickle=True)
mmd_pre_trained = mmd[0]['mmd']

sig_p_task  = [f"PRD Curve of Fixing Parameter {i}" for i in range(44)]
index_list = []
sig_P_mmd_list = [] 
for i, mmd_i in enumerate(mmd):
    if mmd_i['task'] in sig_p_task:
        index_list.append(i)
        sig_P_mmd_list.append(mmd_i['mmd'])
        
# print("Significant MMD indices:", index_list)
plt.figure(figsize=(10, 6))
plt.bar(range(len(sig_P_mmd_list)), sig_P_mmd_list, alpha=0.5, color='steelblue')
plt.axhline(y=mmd_pre_trained, color='red', linestyle='--', linewidth=1.2, 
            label=f'Pre-trained Model MMD {mmd_pre_trained:.4f}')
plt.xticks(range(len(sig_P_mmd_list)), [str(i) for i in range(44)], rotation=45)
plt.xlabel('Fixed Parameter Index', fontsize=12)
plt.ylabel('MMD Value', fontsize=12)
plt.title('MMD Values for Fixed Parameters', fontsize=14)
plt.legend(fontsize=14)
plt.grid(True)
plt.tight_layout()
# plt.show()

Component_list =["Midship Cross Section", "Bow", "Stern", "Bulb"]
sig_c_task = [f"PRD Curve of Fixing {Component_list[i]}" for i in range(len(Component_list))]
index_list = []
sig_C_mmd_list = [] 
for i, mmd_i in enumerate(mmd):
    if mmd_i['task'] in sig_c_task:
        index_list.append(i)
        sig_C_mmd_list.append(mmd_i['mmd'])
print("Significant Component MMD indices:", index_list)
plt.figure(figsize=(10, 6))
# print("Significant Component MMD values:", sig_C_mmd_list)
plt.bar(range(len(sig_C_mmd_list)), sig_C_mmd_list, alpha=0.5, color='steelblue')
plt.axhline(y=mmd_pre_trained, color='red', linestyle='--', linewidth=1.2,
            label=f'Pre-trained Model MMD {mmd_pre_trained:.4f}')
plt.xticks(range(len(sig_C_mmd_list)), Component_list)
plt.xlabel('Fixed Components', fontsize=12)
plt.ylabel('MMD Value', fontsize=12)
plt.title('MMD Values for Fixed Component', fontsize=14)
plt.legend(fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.show()

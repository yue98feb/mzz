import os
import subprocess


# 获取当前目录下的所有.py文件
# py_files = [file for file in os.listdir() if file.endswith('.py')]

# 排除当前脚本文件，如果当前文件名是 'run_scripts.py'
# py_files.remove('script.py')
# py_files.remove('proposed(1).py')
# py_files.remove('1031.py')

# 得到参数列表
arg_list_1 = []
# pattern_list = ['FEDAVG','FEDSGD','SIGNSGD','baseline4','baseline5','PROPOSED']
pattern_list = ['baseline4','baseline5','PROPOSED']
num_list = [10,15,20]
# optimizer_list = ['sgd','adam']


# for w in range(len(wer_list)):
#     for i in range(len(count)):
#         arguments = ['--wer', f'{wer_list[w]}', '--Tmax', f'{2.0}','--Emax', f'{1.5}','--num_clients','10','--num_epoch','200','--count_py',f'{count[i]}']
#         arg_list_1.append(arguments)

for k in range(len(pattern_list)):
    arguments = ['--pattern',f'{pattern_list[k]}']
    arg_list_1.append(arguments)
    
        
file = './src_RES/main.py'

# arg_list_2 = []
# a = ['--pattern', 'FEDSGD']
# b = ['--pattern', 'SIGNSGD']
# c = ['--pattern', 'FEDAVG']
# arg_list_2 = [a,b,c]

# # 遍历所有的.py文件并运行它们
for ind in range(len(arg_list_1)):
    print(f"Running {file} with argument {arg_list_1[ind]}")
    subprocess.run(['python', file] + arg_list_1[ind])



# wer_list = [0.004, 0.005,0.0055,0.006, 0.008, 0.01]
# T_maxs = [4.0, 3.5, 3.0, 2.5, 2.0]
# E_maxs = [2.5, 2.3, 2.0]
# arg_list_2 = []
# for i in range(len(wer_list)):
#     for j in range(len(T_maxs)):
#         for k in range(len(E_maxs)):
#             arguments = ['--wer', f'{wer_list[i]}', '--Tmax', f'{T_maxs[j]}','--Emax', f'{E_maxs[k]}','--num_clients','10','--num_epoch','1000']
#             arg_list_2.append(arguments)

# file = 'proposed(1).py'
# for ind in range(len(arg_list_2)):
#     print(f"Running {file} with argument {arg_list_2[ind]}")
#     subprocess.run(['python', file] + arg_list_2[ind])

# file = 'avg_fresh.py'
# for ind in range(len(arg_list_1)):
#     print(f"Running {file} with argument {arg_list_1[ind]}")
#     subprocess.run(['python', file] + arg_list_1[ind])
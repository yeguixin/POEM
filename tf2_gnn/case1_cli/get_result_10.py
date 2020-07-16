import re
import numpy as np
import sys

result_log = sys.argv[1] # 这里存放的是十倍交叉的结果
platform = sys.argv[2] # 这里存放的是平台 amd 或者NVIDIA


file_path = "/dev/shm/zjq/case1_data/data_csv/data_orgin/"
print("path=", file_path+platform)
print(f"result in {result_log}")


numb = []
with open(result_log,"r") as fd:
# with open("result.log","r") as fd:
    for log in fd.readlines():
        result = re.findall(r'KAccuracy.*',log) 
        if result:
            numb.append(eval(re.findall(r'\d+.\d+',result[0])[0]))
#print("分benchmark的结果是:", numb)

# 得到训练结果

numb = []
with open(result_log,"r") as fd:
    log = fd.read().replace("\n","")
    # r = re.findall(r"pkl.*?KAccuracy",log)
    r = re.findall(r"RGIN_GraphBinaryClassifica.*?KAccuracy",log)
    for i in r:
        # print(i, "\n")
        for j in re.findall(r"\[(.*?)\]",i):
            a = j.replace(". ",",").replace(".",",")
            numb.extend(list(eval(a)))
# print(np.mean(numb))
numb = np.array(numb)
# print(numb)
print(len(numb))


# 将训练结果按照顺序排列
import json 
index = []
for i in range(10):
    with open(f"{file_path}/{platform}_index/{i}.json", "r") as fd:
        index.extend(json.loads(fd.read())[1])
# print(index)
print(len(index))

result = [0]*680
for i in range(len(result)):
    result[index[i]] = numb[i]
# print(result)
# 
# 
# 保存训练结果,计算加速比
import pandas as pd
data = pd.read_csv(f"{file_path}{platform}_680.csv") # 加载csv文件


# 计算加速比
# # runtimes of baseline mapping (CPU on AMD, GPU on NVIDIA)
zero_r_dev = "runtime_cpu" if platform == "amd" else "runtime_gpu"
zer_r_runtimes = data[zero_r_dev]
# print('\n索引对应的运行时间\n',zer_r_runtimes)

# # speedups of predictions
runtimes = data[['runtime_cpu', 'runtime_gpu']].values
# print("\n左CPU, 右GPU\n", runtimes)
p_runtimes = [r[p_] for p_, r in zip(result, runtimes)]
# print("\n也不知道是啥时间\n",p_runtimes)
p_speedup = zer_r_runtimes / p_runtimes
print("p_speedup:",np.mean(p_speedup))

data.insert(13,"prediction", result)
data.insert(10,"p_speedup",p_speedup)
data.drop(["src"], axis=1, inplace=True)
data.drop(["seq"], axis=1, inplace=True)
# data.to_csv(f"result_{platform}.csv", index=False)

# 计算准确率
import numpy as np
prediction = np.array(data["prediction"])
oracle = np.array(data["oracle"])
accuracy = np.array(prediction==oracle).astype(int)
print("accuracy:",accuracy.sum()/len(accuracy))


import pickle
import numpy as np
import matplotlib.pyplot as plt

noniid_exp_name=['noniid_1.0','noniid_0.75','noniid_0.5','noniid_0.25','noniid_1class','noniid_2class']
noniid_exp=[]
for file in noniid_exp_name:
    with open(f'{file}.txt', 'rb') as pf:
        test_acc = pickle.load(pf)[1]
    noniid_exp.append(test_acc)

files = noniid_exp_name
accuracy = noniid_exp
for i, test_acc in enumerate(accuracy):
    iid_fraction = None
    arr = files[i].split(';')
    if arr[-1] == '-1':
      iid_fraction = arr[0]
    else:
      iid_fraction = arr[-1]+" class"
    label = 'iid_fraction=' + iid_fraction
    plt.plot(range(0,4), [t[1] for t in accuracy[i]], label=label)
plt.legend()
plt.xlabel('Rounds')
plt.ylabel('Test Accuracy')
plt.savefig('noniid.png')
#!/usr/bin/python3

import matplotlib.pyplot as plt

def main():
    gpu_list = []
    cpu_list = []
    try:
        while True:
            strs = input().split()
            if strs[0] == 'Gpu':
                gpu_list.append(strs[3])
            elif strs[0] == 'Cpu':
                cpu_list.append(strs[3])
    except EOFError:
        pass
    
    x = [i for i in range(1, 21)]
    plt.plot(x, gpu_list)
    plt.plot(x, cpu_list)
    plt.show()



if __name__ == "__main__":
    pass

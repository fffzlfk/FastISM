#!/usr/bin/python3

import matplotlib.pyplot as plt

def main():
    gpu_list = []
    cpu_list = []
    try:
        while True:
            strs = input().split()
            if 'res' not in strs:
                continue
            if strs[0] == 'Gpu':
                gpu_list.append(float(strs[3]))
            elif strs[0] == 'Cpu':
                cpu_list.append(float(strs[3]))
    except EOFError:
        pass
   
    print(gpu_list, cpu_list) 
    x = [i for i in range(1, 21)]
    # plt.plot(gpu_list, x)
    plt.plot(x, cpu_list)
    plt.show()



if __name__ == "__main__":
    main()

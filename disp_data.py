import matplotlib.pyplot as plt
import numpy as np

# returns an array for each type of value, with heading at the start
# e.g. data = [["t", 0, 4000 ...], ["optim_avg", -200, -50 ...]]
# just in case i wanna use them as labels for the graph
def getdata(lines):
    headers = lines.pop(0).split(",")
    
    data = [[x] for x in headers]
    for line in lines:
        split = line.split(",")
        for i in range(len(data)):
            data[i].append(float(split[i]))
    return data

def loadfile(filename):
    with open(filename) as file:
        lines = file.readlines()
    return getdata(lines)

def plot_optim_avg(avgs, width, height, cell):
    i = 1
    plt.subplot(width, height, cell)
    # plt.yscale("log")
    plt.title("Optimal Episode Averages")
    for data in avgs:
        plt.plot(data[0][1:], smooth(data[1][1:]), label = i, alpha=0.7)
        i += 1
    plt.legend()

def plot_eps_len(eps, width, height, cell):
    i = 1
    plt.subplot(width, height, cell)
    # plt.yscale("log")
    plt.title("Training Episode Lengths")
    for data in eps:
        plt.plot(smooth(data[1][1:]), label = str(i), alpha=0.7)
        # plt.plot(data[0][1:], data[1][1:], label = i)
        i += 1
    plt.legend()

def plot_eps_rew(eps, width, height, cell):
    i = 1
    plt.subplot(width, height, cell)
    # plt.yscale("log")
    plt.title("Training Episode Rewards")
    for data in eps:
        plt.plot(smooth(data[2][1:]), label = str(i), alpha=0.7)
        # plt.plot(data[0][1:], data[1][1:], label = i)
        i += 1
    plt.legend()

def loadbatch(base, id, end):
    files = [base + x + end for x in id]
    data = []
    for file in files:
        data.append(loadfile(file))
    return data

def smooth(data):
    npdata = np.array(data)
    radius = 10
    npdata = np.pad(npdata, radius, "edge")
    return np.convolve(npdata, np.ones(radius*2+1)/(radius*2+1), "valid")
    # return np.convolve(npdata, np., "valid")


# literally just because it's easier then writing them all out
# replace if you want, or just put the paths directly in the files variable
filebase = "tests\\17046"
part = "_optim"
ext = ".csv"
ids = ["07687", "11254", "11315", "16907", "20618", "79666", "84972"]
width = 2
height = 2
avgs = loadbatch(filebase, ids, "_optim.csv")
plot_optim_avg(avgs, width, height, 1)
eps = loadbatch(filebase, ids, "_eps.csv")
plot_eps_len(eps, width, height, 2)
plot_eps_rew(eps, width, height, 4)

i = 1
for file in [filebase + id + "_info.txt" for id in ids]:
    with open(file) as text:
        print()
        print(i)
        print(text.read())
        i += 1

plt.show()
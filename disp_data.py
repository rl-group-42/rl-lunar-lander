import matplotlib.pyplot as plt

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

def plot_optim_avg(files):
    for file in files:
        data = loadfile(file)
        plt.plot(data[0][1:], data[1][1:], label = file)

files = []

# literally just because it's easier then writing them all out
# replace if you want, or just put the paths directly in the files variable
filebase = "tests\\17046"
part = "_optim"
ext = ".csv"
for num in ["07687", "11254", "11315", "16907", "20618"]:
    files.append(filebase + num + part + ext)

plot_optim_avg(files)

plt.legend()
plt.show()
import numpy as np

train = list(open("train.txt"))
test = list(open("test.txt"))

with open('x3d_train.txt', 'w+') as f:
    normal_files = []
    for line in train:
        parts = line.strip().split()
        path = parts[0]
        label = parts[1]
        filename = path.split('/')[-1]
        if label == "normal":
            normal_files.append((filename, label))
        else:
            newline = 'X3D_Videos/' + filename[:-4] + '.npy ' + label + '\n'
            f.write(newline)
    for filename, label in normal_files:
        newline = 'X3D_Videos/' + filename[:-4] + '.npy ' + label + '\n'
        f.write(newline)

with open('x3d_test.txt', 'w+') as f:
    normal_files = []
    for line in test:
        parts = line.strip().split()
        path = parts[0]
        label = parts[1]
        filename = path.split('/')[-1]
        if label == "normal":
            normal_files.append((filename, label))
        else:
            newline = 'X3D_Videos/' + filename[:-4] + '.npy ' + label + '\n'
            f.write(newline)
    for filename, label in normal_files:
        newline = 'X3D_Videos/' + filename[:-4] + '.npy ' + label + '\n'
        f.write(newline)

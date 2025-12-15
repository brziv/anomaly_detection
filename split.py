import random

with open('dataset.txt', 'r') as f:
    lines = f.readlines()

# Separate into accident and normal
accident_lines = [line for line in lines if ' accident ' in line]
normal_lines = [line for line in lines if ' normal ' in line]

# Shuffle each
random.shuffle(accident_lines)
random.shuffle(normal_lines)

# Split 80-20
train_acc = accident_lines[:int(0.8 * len(accident_lines))]
test_acc = accident_lines[int(0.8 * len(accident_lines)):]

train_norm = normal_lines[:int(0.8 * len(normal_lines))]
test_norm = normal_lines[int(0.8 * len(normal_lines)):]

# Combine
train_lines = train_acc + train_norm
test_lines = test_acc + test_norm

# Shuffle the combined to mix classes
random.shuffle(train_lines)
random.shuffle(test_lines)

# Remove the last number from each line
def remove_last_number(lines):
    return [ ' '.join(line.split()[:-1]) + '\n' for line in lines ]

train_lines = remove_last_number(train_lines)
test_lines = remove_last_number(test_lines)

with open('train.txt', 'w') as f:
    f.writelines(train_lines)

with open('test.txt', 'w') as f:
    f.writelines(test_lines)

print(f"Stratified split: Train - {len(train_acc)} accident, {len(train_norm)} normal")
print(f"Test - {len(test_acc)} accident, {len(test_norm)} normal")
from DecisionTree import *
from clean_dict import *

DT = DecisionTree(600)

train_path = 'data/adult.data'
test_path = 'data/adult.test'

train = clean_data(train_path, False)
test = clean_data(test_path, True)
print(DT.build_tree(train))

correct = 0
num_data = 0
for data in test:
    num_data += 1
    predict_label = DT.classify(data)
    if predict_label == data[-1]:
        correct += 1
print(float(correct) / num_data)
print(correct)
print(num_data)
DT.print_tree()
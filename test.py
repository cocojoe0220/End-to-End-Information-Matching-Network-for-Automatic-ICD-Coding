f = open('data/train.csv', 'r', encoding='utf-8')
lines = f.readlines()
data = []
labels_set = []
for line in lines:
    line = line.replace('\n','').split(',')
    data.append(line[0])
    labels = []
    for label in line[1].split('/'):
        labels.append(label)
    labels_set.append(labels)

accuracy_set = []
error_rate_set = []
for index in range(1, len(data)):
    sentence = data[index]
    predict = {'单一血管的操作', '单根导管冠状动脉造影', '经皮冠状动脉球囊扩张成形术', '经皮冠状动脉药物洗脱支架置入术', '置入一个血管支架'}
    labels = set(labels_set[index])
    diff_between_predict_labels = predict.difference(labels)
    diff_between_labels_predict = labels.difference(predict)
    accuracy = (len(labels) - len(diff_between_labels_predict)) / len(labels)
    error_rate = len(diff_between_predict_labels) / len(predict)
    accuracy_set.append(accuracy)
    error_rate_set.append(error_rate)

sum = 0
for accuracy in accuracy_set:
    sum = sum + accuracy
print('avg accuracy : ' + str(sum / len(accuracy_set)))

sum = 0
for error_rate in error_rate_set:
    sum = sum + error_rate
print('avg error_rate : ' + str(sum / len(error_rate_set)))








print(ls)
print(labels_set[6])

print(set(ls).difference(set(labels_set[6])))
print(set(labels_set[6]).difference(set(ls)))

p = (len(labels_set[6]) - len(set(labels_set[6]).difference(set(ls)))) / len(labels_set[6])
e = len(set(ls).difference(set(labels_set[6]))) / len(set(ls))
print(p)
print(e)
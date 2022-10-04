from openpyxl import load_workbook
from openpyxl import Workbook

excel_name = 'data.xlsx'


wb = load_workbook(excel_name)
sheet_name = 'sheet1'
sheet = wb[sheet_name]

id_text_map = {}
id_ops_map = {}
op_set = set()

for row in sheet:
    id = row[0].value
    text = row[2].value
    text = text.replace('\r', '').replace('\n', '').replace(' ', '').replace(',', '，')
    op = row[3].value.replace('/', '')
    id_text_map[id] = text
    op_set.add(op)
    if id  not in id_ops_map.keys():
        id_ops_map[id] = [op]
    else:
        id_ops_map[id].append(op)
'''
f1 = open('../data_preprocessing/train.csv', 'w', encoding='utf-8')
f1.write('content,label' + "\n")

for id in id_text_map.keys():
    text = id_text_map[id]
    ops = id_ops_map[id]
    op_text = ''
    for op in ops:
        op_text += '/' + op
    op_text = op_text[1:]
    text += ',' + op_text
    f1.write(text + "\n")

'''
for op in op_set:
    print(op)
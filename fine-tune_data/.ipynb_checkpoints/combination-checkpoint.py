import json

# 读取第一个JSON文件
with open('contents_docid.json', 'r') as file:
    data1 = json.load(file)

# 读取第二个JSON文件
with open('all_title.json', 'r') as file:
    data2 = json.load(file)

# 确保两个JSON文件的数据长度相同
combined_data = dict()
if len(data1) == len(data2):
    # 合并数据
    for i in range(len(data1)):
        combined_data[data1[i]] = data2[i]
    # combined_data = [{data1[i], data2[i]} for i in range(len(data1))]

    # 将合并后的数据写入新的JSON文件
    with open('docid_title_pair.json', 'w') as file:
        json.dump(combined_data, file, indent=4)
else:
    print("Error: The JSON files do not contain the same number of entries.")

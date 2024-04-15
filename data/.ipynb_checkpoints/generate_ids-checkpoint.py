import json
import numpy as np
from sklearn.cluster import KMeans
import time

with open('annotations/coco_train.json', 'r') as coco_train_json:
    coco_train = json.load(coco_train_json)
with open('annotations/coco_val.json', 'r') as coco_val_json:
    coco_val = json.load(coco_val_json)
with open('annotations/coco_test.json', 'r') as coco_test_json:
    coco_test = json.load(coco_test_json)
    
# image_list is used to store the image_file_name. Its length equals to the numbers of images.
# caption_list is used to store the captions of images. Its length equals to the numbers of total captions. 
# image_caption is used to store the projection from image_file_name to its captions. In this key, the key is the file_name of images and value is the row number of caption. 

image_caption = dict()
row = 0
image_list = list()
caption_list = list()
for temp in coco_train:
    image = temp['image']
    caption = temp['caption']
    caption_list.append(caption)
    if image not in image_caption.keys():
        image_caption[image] = [row]
        image_list.append(image)
    else:
        image_caption[image].append(row)
    row += 1

# Different from training dataset, in the validation and testing sets, each item consists of the file_name of images and their caption list.
for temp in coco_val:
    image = temp['image']
    temp_caption_list = temp['caption']
    image_list.append(image)
    caption_list.extend(temp_caption_list)
    if image not in image_caption.keys():
        image_caption[image] = list(range(row, row+len(temp_caption_list)))
    row += len(temp_caption_list)

for temp in coco_test:
    image = temp['image']
    temp_caption_list = temp['caption']
    image_list.append(image)
    caption_list.extend(temp_caption_list)
    if image not in image_caption.keys():
        image_caption[image] = list(range(row, row+len(temp_caption_list)))
    row += len(temp_caption_list)
    
caption_embed = np.load('caption_embeddings.npy')

# this part aims to implement the image embedding calculation by aggregating the embeddings of its captions. The code is adopted on the training set. 
# Here, we list all the images in training set. For each image, its row (i.e., index) in the image_list corresponds to the row of its caption in caption_list. 
# Therefore, we can use the index to extract the embedding vector from caption_embed. 
image_id_list = list()
for temp in coco_train:
    image_id_list.append(temp['image_id'])

# id_caption_dict is build to store the image_name and the sum of its captions' embeddings.
# id_num_dict is to store the image_name the numbers of its captions.
id_caption_dict = dict()
id_num_dict = dict()
for index in range(len(image_id_list)):
    image_id = image_id_list[index]
    if image_id not in id_caption_dict.keys():
        id_caption_dict[image_id] = caption_embed[index]
        id_num_dict[image_id] = 1
    else:
        id_caption_dict[image_id] += caption_embed[index]
        id_num_dict[image_id] += 1

np.save('id_caption_dict.npy', np.array(id_caption_dict))
np.save('id_num_dict.npy', np.array(id_num_dict))

# Here, we can compute the representation vectors of images.
train_image_embed = list()
for key in id_caption_dict.keys():
    train_image_embed.append(id_caption_dict[key]/id_num_dict[key])

train_image_embed = np.array(train_image_embed)

# As for validation and testing sets, we can easily compute the images' vectors with the embedding matrix of captions. Because each image has five captions. 
val_caption_embed = caption_embed[-10000: -5000]
test_caption_embed = caption_embed[-5000:]

new_val_caption_embed = val_caption_embed.reshape([-1, 5, 1536])
new_test_caption_embed = test_caption_embed.reshape([-1, 5, 1536])

val_image_embed = new_val_caption_embed.mean(1)
test_image_embed = new_test_caption_embed.mean(1)

# generating the image embedding matrix by concatenating the embeddings in training, validation, testing sets. 
image_embed = np.concatenate([train_image_embed, val_caption_embed, test_caption_embed], axis=0)
print(np.shape(image_embed))
np.save('image_embed.npy', image_embed)


# 伪代码中的生成语义ID函数
def generate_semantic_ids(X, k, J_x=None, absolute_indices=None):
    if absolute_indices is None:
        absolute_indices = np.arange(len(X))

    if J_x is None:
        J_x = []
    # 步骤C1：使用K均值聚类算法将文档嵌入向量X聚类成k个簇
    kmeans = KMeans(n_clusters=k,n_jobs=-1)
    labels = kmeans.fit_predict(X)
    # print(labels)
    J = []

    # 步骤C2：循环遍历每个簇
    for i in range(k):
        current_cluster_indices = np.where(labels == i)[0]
        current_cluster_size = len(current_cluster_indices)
        J_current = [str(i)] * current_cluster_size
        
        if current_cluster_size > c:
            # 步骤C3：递归生成子簇的语义ID
            J_rest, J_x = generate_semantic_ids(X[current_cluster_indices], k, J_x, absolute_indices[current_cluster_indices])
        else:
            # 步骤C4：生成当前簇的语义ID
            J_rest = [str(j) for j in range(current_cluster_size)]
            J_x.extend(absolute_indices[current_cluster_indices])
        
        # print(J_rest)
        
        # 步骤C5：将J_current和J_rest逐元素拼接成J_cluster
        J_cluster = [a + b for a, b in zip(J_current, J_rest)]
        
        # 将J_cluster中的语义ID添加到J列表中
        J.extend(J_cluster)

    return J, J_x

# 设定聚类簇数k和阈值c
k = 10
c = 10

# 调用生成语义ID函数
image_ids, image_idx = generate_semantic_ids(image_embed, k)

combined = sorted(zip(image_ids, image_idx), key=lambda x: x[1])

sorted_image_ids, sorted_image_idx = zip(*combined)
sorted_image_ids = list(sorted_image_ids)
 
img_caption_ids = [None] * len(caption_list)
for i, (img_id, img_idx) in enumerate(zip(sorted_image_ids, sorted_image_idx)):
    img_name = image_list[img_idx]
    caption_indices = image_caption[img_name]
    sorted_image_ids[i] += '0'
    for j, caption_idx in enumerate(caption_indices, start=1):
        caption_id = img_id + str(j)
        img_caption_ids[caption_idx] = (sorted_image_ids[i], caption_id)

with open('img_caption_ids.json', 'w') as f:
    json.dump(img_caption_ids, f)
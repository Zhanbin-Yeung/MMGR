import torch
import time
import numpy as np
import torch.utils.data as Data
import torch.nn as nn
import pickle
import random


seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic=True

 # total number of items
content = torch.load('bert_img_embeds.pt')
print(content.size())


class MyDataSet(Data.Dataset):
    def __init__(self, enc_inputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx]


class RQVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, output_dim, n_embedding_1, n_embedding_2, n_embedding_3, n_embedding_4, n_embedding_5):
        super().__init__()
        # self.encoder = nn.Sequential(
        #     nn.Linear(input_dim, hidden_dim_1),
        #     nn.ReLU(),
        #     # nn.Dropout(0.2),
        #     nn.Linear(hidden_dim_1, hidden_dim_2),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim_2, output_dim)
        # )
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim)
        )

        self.vq_embedding_1 = nn.Embedding(n_embedding_1, output_dim)
        torch.nn.init.xavier_normal_(self.vq_embedding_1.weight.data)
        # self.vq_embedding_1.weight.data.uniform_(-1.0 / n_embedding_1, 1.0 / n_embedding_1)
        self.vq_embedding_2 = nn.Embedding(n_embedding_2, output_dim)
        torch.nn.init.xavier_normal_(self.vq_embedding_2.weight.data)
        # self.vq_embedding_2.weight.data.uniform_(-1.0 / n_embedding_2, 1.0 / n_embedding_2)
        self.vq_embedding_3 = nn.Embedding(n_embedding_3, output_dim)
        torch.nn.init.xavier_normal_(self.vq_embedding_3.weight.data)
        # self.vq_embedding_3.weight.data.uniform_(-1.0 / n_embedding_3, 1.0 / n_embedding_3)
        self.vq_embedding_4 = nn.Embedding(n_embedding_4, output_dim)
        torch.nn.init.xavier_normal_(self.vq_embedding_4.weight.data)
        self.vq_embedding_5 = nn.Embedding(n_embedding_5, output_dim)
        torch.nn.init.xavier_normal_(self.vq_embedding_5.weight.data)
        
        # self.decoder = nn.Sequential(
        #     nn.Linear(output_dim, hidden_dim_2),
        #     # nn.Sigmoid(),
        #     nn.ReLU(),
        #     # nn.Dropout(0.2),
        #     nn.Linear(hidden_dim_2, hidden_dim_1),
        #     nn.ReLU(),
        #     # nn.Sigmoid(),
        #     nn.Linear(hidden_dim_1, input_dim)
        # )
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, input_dim)
        )


    def forward(self, x):
        

        # 使用torch.multinomial在每行上按照给定的概率分布r进行采样
        sampled_indices = torch.multinomial(torch.tensor([0.7, 0.2, 0.1]), 1, replacement=True)

        ze_1 = self.encoder(x)
        # ze_1 = x
        # ze: [N, C]
        # embedding: [K, C]

        # block 1
        N, C = ze_1.shape
        embedding_1 = self.vq_embedding_1.weight.data
        K_1, _ = embedding_1.shape
        embedding_broadcast_1 = embedding_1.reshape(1, K_1, C)
        ze_broadcast_1 = ze_1.reshape(N, 1, C)
        distance_1 = torch.sum((embedding_broadcast_1 - ze_broadcast_1)**2, 2)
        # N
        nearest_neighbor_1 = torch.argmin(distance_1, 1)
        # values, indics = torch.topk(distance_1, 2, dim=1, largest=False)
        # # 根据采样的索引获取对应的值
        # nearest_neighbor_1 = indics[torch.arange(indics.size(0)), sampled_indices.view(-1)]
        
        # zq: [N, C]
        zq_1 = self.vq_embedding_1(nearest_neighbor_1)

        # block 2
        ze_2 = ze_1 - zq_1
        embedding_2 = self.vq_embedding_2.weight.data
        K_2, _ = embedding_2.shape
        embedding_broadcast_2 = embedding_2.reshape(1, K_2, C)
        ze_broadcast_2 = ze_2.reshape(N, 1, C)
        distance_2 = torch.sum((embedding_broadcast_2 - ze_broadcast_2) ** 2, 2)
        # N
        nearest_neighbor_2 = torch.argmin(distance_2, 1)
        # values, indics = torch.topk(distance_2, 2, dim=1, largest=False)
        # # 根据采样的索引获取对应的值
        # nearest_neighbor_2 = indics[torch.arange(indics.size(0)), sampled_indices.view(-1)]
        # zq: [N, C]
        zq_2 = self.vq_embedding_2(nearest_neighbor_2)

        # block 3
        ze_3 = ze_2 - zq_2
        embedding_3 = self.vq_embedding_3.weight.data
        K_3, _ = embedding_3.shape
        embedding_broadcast_3 = embedding_3.reshape(1, K_3, C)
        ze_broadcast_3 = ze_3.reshape(N, 1, C)
        distance_3 = torch.sum((embedding_broadcast_3 - ze_broadcast_3) ** 2, 2)
        # N
        nearest_neighbor_3 = torch.argmin(distance_3, 1)
        # values, indics = torch.topk(distance_3, 2, dim=1, largest=False)
        # # 根据采样的索引获取对应的值
        # nearest_neighbor_3 = indics[torch.arange(indics.size(0)), sampled_indices.view(-1)]
        # zq: [N, C]
        zq_3 = self.vq_embedding_3(nearest_neighbor_3)
        
        # block 4
        ze_4 = ze_3 - zq_3
        embedding_4 = self.vq_embedding_4.weight.data
        K_4, _ = embedding_4.shape
        embedding_broadcast_4 = embedding_4.reshape(1, K_4, C)
        ze_broadcast_4 = ze_4.reshape(N, 1, C)
        distance_4 = torch.sum((embedding_broadcast_4 - ze_broadcast_4) ** 2, 2)
        # N
        nearest_neighbor_4 = torch.argmin(distance_4, 1)
        # values, indics = torch.topk(distance_4, 3, dim=1, largest=False)
        # # 根据采样的索引获取对应的值
        # nearest_neighbor_4 = indics[torch.arange(indics.size(0)), sampled_indices.view(-1)]
        # zq: [N, C]
        zq_4 = self.vq_embedding_4(nearest_neighbor_4)

        # # block 4
        # ze_5 = ze_4 - zq_4
        # embedding_5 = self.vq_embedding_5.weight.data
        # K_5, _ = embedding_5.shape
        # embedding_broadcast_5 = embedding_5.reshape(1, K_5, C)
        # ze_broadcast_5 = ze_5.reshape(N, 1, C)
        # distance_5 = torch.sum((embedding_broadcast_5 - ze_broadcast_5) ** 2, 2)
        # # N
        # nearest_neighbor_5 = torch.argmin(distance_5, 1)
        # # zq: [N, C]
        # zq_5 = self.vq_embedding_4(nearest_neighbor_5)
        
        # decoder_input = zq_1 + zq_2 + zq_3
        decoder_input = ze_1 + (-ze_1 + (zq_1 + zq_2 + zq_3 + zq_4 )).detach()
        # decoder_input = ze_1 + (-ze_1 + (zq_1 + zq_2 + zq_3 + zq_4 + zq_5)).detach()
        x_hat = self.decoder(decoder_input)
        
        return x_hat, ze_1, ze_2, ze_3, ze_4, zq_1, zq_2, zq_3, zq_4, nearest_neighbor_1, nearest_neighbor_2, nearest_neighbor_3, nearest_neighbor_4
        # return x_hat, ze_1, ze_2, ze_3, ze_4, ze_5, zq_1, zq_2, zq_3, zq_4, zq_5, nearest_neighbor_1, nearest_neighbor_2, nearest_neighbor_3, nearest_neighbor_4, nearest_neighbor_5


batch_size = 4096
lr = 1e-3
n_epochs = 250
l_w_embedding = 0.05
l_w_commitment = 0.05
loader = Data.DataLoader(MyDataSet(content), batch_size=batch_size, shuffle=True)
model = RQVAE(768, 256, 32, 4, 64, 64, 64, 64, 64).cuda()
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=1e-5)
mse_loss = nn.MSELoss()
tic = time.time()
for e in range(n_epochs):
    total_loss = 0
    for x in loader:
        current_batch_size = x.shape[0]
        x = x.cuda()
        
        x_hat, ze_1, ze_2, ze_3, ze_4, zq_1, zq_2, zq_3, zq_4, id_1, id_2, id_3, id_4 = model(x)
        # x_hat, ze_1, ze_2, ze_3, ze_4, ze_5, zq_1, zq_2, zq_3, zq_4, zq_5, id_1, id_2, id_3, id_4, id_5 = model(x)
        
        l_reconstruct = mse_loss(x, x_hat)
        l_embedding = mse_loss(ze_1.detach(), zq_1) + mse_loss(ze_2.detach(), zq_2) + mse_loss(ze_3.detach(), zq_3) + mse_loss(ze_4.detach(), zq_4)# + mse_loss(ze_5.detach(), zq_5)
        l_commitment = mse_loss(ze_1, zq_1.detach()) + mse_loss(ze_2, zq_2.detach()) + mse_loss(ze_3, zq_3.detach()) + mse_loss(ze_4, zq_4.detach())# + mse_loss(ze_5, zq_5.detach())
        loss = l_reconstruct + l_w_embedding * l_embedding + l_w_commitment * l_commitment
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * current_batch_size
        # print(id_1[0], id_2[0], id_3[0])
    total_loss /= len(loader.dataset)
    toc = time.time()
    print(f'epoch {e} loss: {total_loss:.5f} elapsed {(toc - tic):.2f}s')


model.eval()
with torch.no_grad():
    ITEM_NUM = content.shape[0]
    # x = torch.Tensor(content).cuda()
    # _, _, _, _, _, _, _, nearest_neighbor_1, nearest_neighbor_2, nearest_neighbor_3 = model(x)
    # print(nearest_neighbor_1.numpy())
    # print(nearest_neighbor_2.numpy())
    # print(nearest_neighbor_3.numpy())

    id_1 = np.empty(shape=(0))
    id_2 = np.empty(shape=(0))
    id_3 = np.empty(shape=(0))
    id_4 = np.empty(shape=(0))
    id_5 = np.empty(shape=(0))

    for batch in np.array_split(np.array(list(range(ITEM_NUM))), indices_or_sections=4):
        x = torch.Tensor(content[batch, :]).cuda()
        _, _, _, _, _, _, _, _, _,  nearest_neighbor_1, nearest_neighbor_2, nearest_neighbor_3, nearest_neighbor_4 = model(x)
        # _, _, _, _, _, _, _, _, _, _, _, nearest_neighbor_1, nearest_neighbor_2, nearest_neighbor_3, nearest_neighbor_4, nearest_neighbor_5 = model(x)
        id_1 = np.append(id_1, nearest_neighbor_1.cpu().numpy())
        id_2 = np.append(id_2, nearest_neighbor_2.cpu().numpy())
        id_3 = np.append(id_3, nearest_neighbor_3.cpu().numpy())
        id_4 = np.append(id_4, nearest_neighbor_4.cpu().numpy())
        # id_5 = np.append(id_5, nearest_neighbor_5.cpu().numpy())
    id_1 = id_1.reshape(-1, 1)
    id_2 = id_2.reshape(-1, 1)
    id_3 = id_3.reshape(-1, 1)
    id_4 = id_4.reshape(-1, 1)
    # id_5 = id_5.reshape(-1, 1)
    s_id = np.concatenate((id_1, id_2, id_3, id_4), axis=1) + 1
    # s_id = np.concatenate((id_1, id_2, id_3, id_4, id_5), axis=1) + 1
    s_id = s_id.astype(np.int16)

np.save('new_id.npy', s_id)
print(s_id)
print('Done')

def count_duplicate_elements(input_list):
    element_count = {}
    duplicate_count = 0

    for sublist in input_list:
        sublist_tuple = tuple(sublist)
        if sublist_tuple in element_count:
            element_count[sublist_tuple] += 1
            if element_count[sublist_tuple] == 2:
                duplicate_count += 1
        else:
            element_count[sublist_tuple] = 1

    return element_count, duplicate_count

new_id = np.load('new_id.npy')

# input_list = [[1,2,3],[2,3,4],[1,2,3],[2,3,5]]
element_count, duplicate_count = count_duplicate_elements(list(new_id))
print(len(element_count.keys()))
print(len(new_id))


# new_id = np.load('new_id.npy')

# # input_list = [[1,2,3],[2,3,4],[1,2,3],[2,3,5]]
# output = count_duplicate_elements(list(new_id))
# print(output)


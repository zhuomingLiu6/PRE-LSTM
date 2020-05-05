# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
import torch as t
from torch import nn
import tqdm
from torchnet import meter
import ipdb
from torch.autograd import Variable
from utils import Visualizer
import random
import csv
from model import *
from function import *

#tablepath = r'C:\Users\XPS\Desktop\唯一表格名字.txt'
#测试词向量
model_path = r"C:\Users\XPS\Desktop\训练结果\28\test_19_withoutvec_arg_28.pth"
parsed_data_path = r"C:\Users\XPS\Desktop\数据\增广的数据\sequence_without_vec.npz"
#model_path = r"C:\Users\XPS\Desktop\训练结果\28\testtestingfix_19_arg_28.pth"
#parsed_data_path = r"C:\Users\XPS\Desktop\数据\增广的数据\sequence.npz"

data, word2ix, ix2word = load_data_(parsed_data_path)
#data = t.from_numpy(data)
#dataloader = t.utils.data.DataLoader(data,batch_size=opt.batch_size,shuffle=True,num_workers=1)

#data, word2ix_train, ix2word_train, word2ix_fix, ix2word_fix = load_data(parsed_data_path)

model = TrainingModel(len(word2ix), 200, 400);
#optimizer = t.optim.Adam(model.parameters(), lr=opt.lr)
#criterion = nn.CrossEntropyLoss()
#测试对应的词向量

#model = TrainingModel_Vec(len(word2ix_train), len(word2ix_fix), 200, 400)

map_location = lambda s, l: s
state_dict = t.load(model_path, map_location=map_location)
model.load_state_dict(state_dict)
#print(model.embeddings.weight.detach().numpy()[1183])

#print("\n")
#print([word2ix['实收资本']])

hello_idx = torch.LongTensor([word2ix['镇街6']])
#hello_idx = torch.LongTensor([word2ix_train['镇街6']])
hello_idx = Variable(hello_idx)
hello_embed = model.embeddings(hello_idx)
#print(hello_embed)

dets = hello_embed.detach().numpy()
#np.savetxt("街镇.txt", dets,fmt='%f',delimiter=',')
np.savetxt("街镇.txt", dets,fmt='%f',delimiter=',')
#print(model.embeddings.weight.detach().numpy()[0])
#print(model.embeddings.weight.detach().numpy()[1])
'''
'''
#向量推荐
deparment_idx = torch.LongTensor([word2ix['女生']])
#deparment_idx = torch.LongTensor([word2ix_fix['女生']])
deparment_idx = Variable(deparment_idx)
deparment_embed = model.embeddings(deparment_idx)
#deparment_embed = model.embeddingsfix(deparment_idx)
dets = deparment_embed.detach().numpy()
#np.savetxt("女生.txt", dets,fmt='%f',delimiter=',')
np.savetxt("女生.txt", dets,fmt='%f',delimiter=',')


key_idx = torch.LongTensor([word2ix['流动人口表']])
#key_idx = torch.LongTensor([word2ix_train['流动人口表']])
key_idx = Variable(key_idx)
key_embed = model.embeddings(key_idx)
#np.savetxt("流动人口表.txt", dets,fmt='%f',delimiter=',')
dets = key_embed.detach().numpy()
np.savetxt("流动人口表.txt", dets,fmt='%f',delimiter=',')

#print(deparment_embed)
#print(key_embed)
#print(deparment_embed+key_embed)
'''
test = deparment_embed+key_embed
#print(t.mm(deparment_embed,t.transpose(key_embed,1,0)))

file_object=open(tablepath,'r',encoding='utf-8')
chunk_data = file_object.read()
file_object.close()
chunk_data = chunk_data.split('\n')

table_idx = []
table_embeds = []
for table in chunk_data:
	idx = Variable(torch.LongTensor([word2ix[table]]))
	table_idx.append(idx)
	table_embeds.append(model.embeddings(idx))

result = []
for table_embed in table_embeds:
	result.append(t.mm(test,t.transpose(table_embed,1,0)))

def takeSecond(elem):
    return elem[1]

tuple_result = []
for i in range(len(result)):
	#print(chunk_data[i])
	#print(result[i].detach().numpy()[0][0])
	tuple_result.append((chunk_data[i],result[i].detach().numpy()[0][0]))
tuple_result.sort(key=takeSecond)
for i in tuple_result:
	print(i[0])
	print(i[1])'''

'''
#生成处理好的数据
generate_parsed_data()'''
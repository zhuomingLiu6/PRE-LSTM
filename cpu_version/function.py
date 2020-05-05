# -*- coding: utf-8 -*-
import numpy as np
import sys, os
import torch as t
from torch import nn
import torch.nn.functional as F
from torchnet import meter
import tqdm
import ipdb
from torch.autograd import Variable
import random
import heapq
import csv


def final_test(testresult,testtarget):
    #testresult 2 demension[testsetsize,10],testtarget [testsetsize] 
    count = 0
    rankingnum = 0
    for i, predict in enumerate(testresult):
        if testtarget[i] in predict:
            count += 1       
        try:
            result = predict.index(testtarget[i])
        except:
            result = 10
        rankingnum += result
    return count / len(testtarget) , rankingnum / len(testtarget)



def generate_test(model, start_words, ix2word_train, word2ix_train, ix2word_fix, word2ix_fix,opt):
    # the input is the no. of the word [d,K]
    final_result = []
    for start_word in start_words:

        results = start_word
        start_word_len = len(start_word)
        # 手动设置第一个词为<START>
        input = t.Tensor([word2ix_train['<START>']]).view(1, 1).long()
        if opt.use_gpu: input = input.cuda()
        hidden = None

        #生成    
        for i in range(start_word_len+1):
            if i == 0:
                #要train的idx
                output, hidden = model(input, hidden, 1 , 0)
                w = results[i]
                #print(w)
                input = input.data.new([w]).view(1, 1)
            elif i == 1:
                #这里的input是部门所以要还是要train_idx
                output, hidden = model(input, hidden, 1, 0)
                w = results[i]
                input = input.data.new([w]).view(1, 1)
            else:
                #这里的前一个就是关键词了所以要用的应该是fix的idx
                output, hidden = model(input, hidden, 1 ,1)
                top_indexes = output.data[0].topk(10)[1]
                #print("testing")
                #print(output.data[0].topk(10))
                #print(ix2word[196])
                #print(results)
                for ii in range(len(top_indexes)):
                    if top_indexes[ii].item() in ix2word_train:
                        results.append(top_indexes[ii].item())
        final_result.append(results)
    return final_result


#generation with index
#the input start_words is the 2 demension list, each list in start_words is the sequence for generation
def generate(model, start_words, ix2word_train, word2ix_train, ix2word_fix, word2ix_fix,opt):
    # the input is the word set

    final_result = []
    for start_word in start_words:

        results = start_word
        start_word_len = len(start_word)
        # 手动设置第一个词为<START>
        input = t.Tensor([word2ix_train['<START>']]).view(1, 1).long()
        if opt.use_gpu: input = input.cuda()
        hidden = None

        #生成    
        for i in range(start_word_len+1):
            if i == 0:
                #要train的idx
                output, hidden = model(input, hidden, 1 , 0)
                w = results[i]
                input = input.data.new([word2ix_train[w]]).view(1, 1)
            elif i == 1:
                #这里的input是部门所以要还是要train_idx
                output, hidden = model(input, hidden, 1, 0)
                w = results[i]
                input = input.data.new([word2ix_fix[w]]).view(1, 1)
            else:
                #这里的前一个就是关键词了所以要用的应该是fix的idx
                output, hidden = model(input, hidden, 1 ,1)
                top_indexes = output.data[0].topk(10)[1]
                #print("testing")
                #print(output.data[0].topk(10))
                #print(ix2word[196])
                #print(results)
                for ii in range(len(top_indexes)):
                    if top_indexes[ii].item() in ix2word_train:
                        results.append(ix2word_train[top_indexes[ii].item()])
        final_result.append(results)
    return final_result

# function for generation with index
def generate_sim(model, start_words, ix2word, word2ix,opt):
    final_result = []
    #print(start_words)
    for start_word in start_words:
        #print("test")
        results = start_word
        start_word_len = len(start_word)
        # 手动设置第一个词为<START>
        input = t.Tensor([word2ix['<START>']]).view(1, 1).long()
        if opt.use_gpu: input = input.cuda()
        hidden = None
        #print(results)
        #生成
        for i in range(start_word_len+1):
            output, hidden = model(input, hidden)

            if i < start_word_len:
                w = results[i]
                input = input.data.new([word2ix[w]]).view(1, 1)
            else:
                top_indexes = output.data[0].topk(10)[1]
                #print("testing")
                #print(output.data[0].topk(10))
                #print(ix2word[196])
                for i in range(len(top_indexes)):
                	results.append(ix2word[top_indexes[i].item()])
        final_result.append(results)
    return final_result


# generation with the vector for the key word
def generate_with_vector(model, start_words, ix2word_train, word2ix_train, ix2word_fix, word2ix_fix,opt):
    #数据入口为 部门名字 和 关键词向量 可以是一批
    final_result = []
    #print(start_words)
    for start_word in start_words:
        #print("test")
        results = start_word
        start_word_len = len(start_word)
        # 手动设置第一个词为<START>
        input = t.Tensor([word2ix_train['<START>']]).view(1, 1).long()
        
        if opt.use_gpu: input = input.cuda()
        hidden = None

        #生成    
        for i in range(start_word_len+1):
            if i == 0:
                #要train的idx
                output, hidden = model(input, hidden, 1, 0)
                w = results[i]
                input = input.data.new([word2ix_train[w]]).view(1, 1)
            elif i == 1:
                #这里的input是部门所以要还是要train_idx
                output, hidden = model(input, hidden, 1, 0)
                w = results[i]
                #这里放进了vec
                #print(w)
                input = [[w]]
                #print(input)
            else:
                #这里的前一个就是关键词了所以要用的应该是fix的idx
                #这里传入了一个vector
                output, hidden = model(input, hidden, 2)
                top_indexes = output.data[0].topk(10)[1]
                #print("testing")
                #print(output.data[0].topk(10))
                #print(ix2word[196])
                #print(results)
                for ii in range(len(top_indexes)):
                    if top_indexes[ii].item() in ix2word_train:
                        results.append(ix2word_train[top_indexes[ii].item()])
        final_result.append(results)
    return final_result


# generation version for one sequence only
def gen_single_vec(**kwargs):
    for k, v in kwargs.items():
        setattr(opt, k, v.strip("'"))

    start_words = [['法院','企业类型']]

    data, word2ix_train, ix2word_train, word2ix_fix, ix2word_fix = load_data(opt.parsed_data_path)
    model = TrainingModel_Vec(len(word2ix_train), len(word2ix_fix), 200, 400)
    map_location = lambda s, l: s
    state_dict = t.load(opt.model_path, map_location=map_location)
    model.load_state_dict(state_dict)

    pretrained_weight = form_matrix(ix2word_fix,opt.pathforvec)
    pretrained_weight = np.array(pretrained_weight)
    model.embeddingsfix.weight.data.copy_(t.from_numpy(pretrained_weight))
    

    if opt.use_gpu:
        model.cuda()

    result = generate(model, start_words, ix2word_train, word2ix_train, ix2word_fix, word2ix_fix)
    #print(result)
    #print(','.join(result))

# called function for generation with index with one pair
def gen_single_novec(**kwargs):
    """
    提供命令行接口，用以生成相应的诗
    """

    data, word2ix, ix2word = load_data_(opt.parsed_data_path)
    model = TrainingModel(len(word2ix), 200, 400);
    map_location = lambda s, l: s
    state_dict = t.load(opt.model_path, map_location=map_location)
    model.load_state_dict(state_dict)

    if opt.use_gpu:
        model.cuda()

    result = generate_sim(model, opt.start_words, ix2word, word2ix)
    print(','.join(result))



#load the parsed data
def load_data(parsed_data_path):
	data = np.load(parsed_data_path)
	data, word2ix_train, ix2word_train, word2ix_fix, ix2word_fix = data['data'], data['word2ix_train'].item(), data['ix2word_train'].item(), data['word2ix_fix'].item(), data['ix2word_fix'].item()
	return data, word2ix_train, ix2word_train, word2ix_fix, ix2word_fix

#load the parsed data(for without vector version)
def load_data_(parsed_data_path):
    data = np.load(parsed_data_path)
    data, word2ix, ix2word = data['data'], data['word2ix'].item(), data['ix2word'].item()
    return data, word2ix, ix2word


# convert the matrix with index to matrix with the real name for print
# the input is the two demension list
def transfer_and_print(ix2word_train,ix2word_fix,data,fix):
	final = []
	for i,row in enumerate(data):
		print(row)
		temp = []
		for item in row:
			if i == fix:
				temp.append(ix2word_fix[item])
			else:
				temp.append(ix2word_train[item])
		final.append(temp)
	return final

# convert the matrix with index to matrix with the real name for print
# the input is the two demension list(for without vector version)
def transfer_and_print_(ix2word,data):
    final = []
    for row in data:
        temp = []
        for item in row:
            temp.append(ix2word[item])
        final.append(temp)
    return final

# read the selected vector, the pattern for the vector is:
# vectorname:demension1,demension2,···
def read_vec(pathforvec):
    targetfile = open(pathforvec,'r', encoding='UTF-8')
    target = targetfile.read()
    target = target.split('\n')
    targetfile.close()

    vec = []
    #分出vec
    for ele in target:
        temp = ele.split(':')
        name = temp[0]
        realrec = list(map(float,temp[1].split(',')))
        vec.append([name,realrec])
    return vec

#set up the matrix for converting the department name or table name to the vector
#as the order in the index
def form_matrix(ix2word_fix,pathforvec):
    vec = read_vec(pathforvec)

    matrixforfix = []
    for i in range(len(ix2word_fix)):
    	word = ix2word_fix[i]
    	#print(word)
    	for item in vec:
    		if word == item[0]:
    			matrixforfix.append(item[1])

    return matrixforfix

# for selecting the first two element in the sequence which represent the department and the keyword
def mystrip(data,datalen):
    final_result = []
    for item in data:
        if datalen == 3:
            final_result.append(item[:2])
        elif datalen == 5:
            final_result.append(item[1:3])
    return final_result

# for splitting the inner sequence into list
def data_split_inner(data,keyforsplit):
    result = []
    for item in data:
        result.append(item.split(keyforsplit))
    return result

# for writing the result into csv
def write_csv(data,writepath):
    with open(writepath, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        for row in data:
            #print(row)
            csv_writer.writerow(row)

# for converting the result into a encrypt version
def covert_to_encrypt(result,tablepath,departmentpath):
    unavailable = 0

    file_object=open(tablepath,'r',encoding='utf-8')
    table = file_object.read()
    file_object.close()
    table = table.split('\n')
    table = data_split_inner(table,' ')

    tableset = {}
    #创造字典
    for row in table:
        #print(row)
        tableset.update({row[0]: row[1]})

    file_object=open(departmentpath,'r',encoding='utf-8')
    department = file_object.read()
    file_object.close()
    department = department.split('\n')
    department = data_split_inner(department,' ')

    departmentset = {}
    for row in department:
        departmentset.update({row[0]: row[1]})

    encryptreuslt = []
    for row in result:
        temp = []
        for i,item in enumerate(row):
            if i == 0:
                temp.append(departmentset[item])
            elif i == 1:
                temp.append(item)
            else:
                if item in tableset.keys():
                    temp.append(tableset[item])
                else:
                    temp.append('NA')
                    unavailable += 1
        encryptreuslt.append(temp)
    return encryptreuslt, unavailable/(len(result)*10)


# the second element of the finaldata is a key word
# this function aim to covert the key word to the vector
# vector format [keyword, [200demension vector]] in vec_set
# final data format [department, keyword]
def converttovector(finaldata,vec_set):
    result = []
    for row in finaldata:
        temp = []
        temp.append(row[0])
        for vec in vec_set:
            if vec[0] == row[1]:
                temp.append(vec[1])
        result.append(temp)
    return result

# the second element of the finaldata is the vector for the keyword
# vector format [keyword, [200demension vector]] in vec_set
# final data format [department, [200demension vector represent a key word], ···10 element for names of recommended tables]
def rever_to_name(withvec,vec_set):
    result = []
    for row in withvec:
        #print(row)
        for vec in vec_set:
            if vec[1] == row[1]:
                row[1] = vec[0]
        result.append(row)
    return result
# -*- coding: utf-8 -*-
import numpy as np
import csv

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
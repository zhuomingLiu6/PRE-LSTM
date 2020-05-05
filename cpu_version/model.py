# -*- coding: utf-8 -*-
import numpy as np
import sys, os
import torch as t
from torch import nn
import torch.nn.functional as F
from torchnet import meter


#version for vector

#definition of training model
class TrainingModel_Vec(nn.Module):
    #vocab_size mean how many distinct word would appear in the poem
    def __init__(self, vocab_size_train, vocab_size_fix, embedding_dim, hidden_dim):
        super(TrainingModel_Vec, self).__init__()
        self.hidden_dim = hidden_dim
        print("vocab_size_train: " + str(vocab_size_train))
        self.embeddings = nn.Embedding(vocab_size_train, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=2)#lstm里面有参数可以通过梯度下降惊醒拟合，通过input得到输入和输出然后计算loss返回提反独下降
        
        #the embeding for the pretrained word vector
        self.embeddingsfix = nn.Embedding(vocab_size_fix, embedding_dim)
        for p in self.embeddingsfix.parameters():
        	p.requires_grad = False
        #self.lstmfix = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=2)

        self.linear1 = nn.Linear(self.hidden_dim, vocab_size_fix + vocab_size_train)

    def forward(self, input, hidden=None, mark=None, num=None):
        #for training
        if mark == None:
            #the input usuassly is a two demension tensor the size is [seq_len,batchsize]
            #every row of the input is the first element of all the seqences in the this batch
            seq_len, batch_size = input.size()
            if hidden is None:
                h_0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
                c_0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
            else:
            	h_0, c_0 = hidden

	        # size: (seq_len,batch_size,embeding_dim) 去掉tensor的第三行
            # our input have specific format, ["<START>"] + department + keyword + tablename + ["<EOP>"]
            # for the training data the format is : ["<START>"] + department + keyword + tablename 
	       
            # size: (seq_len,batch_size,embeding_dim)
            #change for the department and the tablename
            embedschange = self.embeddings(t.cat([input[0:2],input[3:]],dim = 0))
	       #change for the keyword
            embedsfix = self.embeddingsfix(input[2:3])
	        #concatenate the embedding
            embeds=t.cat([embedschange[0:2],embedsfix,embedschange[2:]],dim = 0)

	        
	        # output size: (seq_len,batch_size,hidden_dim)
            output, hidden = self.lstm(embeds, (h_0, c_0))

	        # size: (seq_len*batch_size,vocab_size)
            output = self.linear1(output.view(seq_len * batch_size, -1))
            return output, hidden
	    #for generation with index
        elif mark == 1:

            seq_len, batch_size = input.size()
            if hidden is None:
                h_0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
                c_0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
            else:
                h_0, c_0 = hidden

            # size: (seq_len,batch_size,embeding_dim)
            #for converting the department or table index to vector
            if num == 0: 	
                embeds = self.embeddings(input)
            #for converting the keyword index to vector
            else:
            	embeds = self.embeddingsfix(input)
            # output size: (seq_len,batch_size,hidden_dim)
            output, hidden = self.lstm(embeds, (h_0, c_0))

            # size: (seq_len*batch_size,vocab_size)
            output = self.linear1(output.view(seq_len * batch_size, -1))
            return output, hidden
        #for generation with vector
        elif mark == 2:

            #the input is the vector representation of the key word, a 200 demension vector
            embeds = t.tensor(input)

            if hidden is None:
                h_0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
                c_0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
            else:
                h_0, c_0 = hidden

            output, hidden = self.lstm(embeds, (h_0, c_0))

            # size: (seq_len*batch_size,vocab_size)
            output = self.linear1(output.view(1, -1))
            return output, hidden


#definition of training model
class TrainingModel_Vec_Dropout(nn.Module):
    #vocab_size mean how many distinct word would appear in the poem
    def __init__(self, vocab_size_train, vocab_size_fix, embedding_dim, hidden_dim, dropoutvalue = 0):
        super(TrainingModel_Vec_Dropout, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size_train, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=2,dropout=dropoutvalue)#lstm里面有参数可以通过梯度下降惊醒拟合，通过input得到输入和输出然后计算loss返回提反独下降
        
        #the embeding for the pretrained word vector
        self.embeddingsfix = nn.Embedding(vocab_size_fix, embedding_dim)
        for p in self.embeddingsfix.parameters():
            p.requires_grad = False
        #self.lstmfix = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=2)

        self.linear1 = nn.Linear(self.hidden_dim, vocab_size_fix + vocab_size_train)

    def forward(self, input, hidden=None, mark=None, num=None):
        #for training
        if mark == None:
            #the input usuassly is a two demension tensor the size is [seq_len,batchsize]
            #every row of the input is the first element of all the seqences in the this batch
            seq_len, batch_size = input.size()
            if hidden is None:
                h_0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
                c_0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
            else:
                h_0, c_0 = hidden

            # size: (seq_len,batch_size,embeding_dim) 去掉tensor的第三行
            # our input have specific format, ["<START>"] + department + keyword + tablename + ["<EOP>"]
            # for the training data the format is : ["<START>"] + department + keyword + tablename 
           
            # size: (seq_len,batch_size,embeding_dim)
            #change for the department and the tablename
            embedschange = self.embeddings(t.cat([input[0:2],input[3:]],dim = 0))
           #change for the keyword
            embedsfix = self.embeddingsfix(input[2:3])
            #concatenate the embedding
            embeds=t.cat([embedschange[0:2],embedsfix,embedschange[2:]],dim = 0)

            
            # output size: (seq_len,batch_size,hidden_dim)
            output, hidden = self.lstm(embeds, (h_0, c_0))

            # size: (seq_len*batch_size,vocab_size)
            output = self.linear1(output.view(seq_len * batch_size, -1))
            return output, hidden
        #for generation with index
        elif mark == 1:

            seq_len, batch_size = input.size()
            if hidden is None:
                h_0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
                c_0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
            else:
                h_0, c_0 = hidden

            # size: (seq_len,batch_size,embeding_dim)
            #for converting the department or table index to vector
            if num == 0:    
                embeds = self.embeddings(input)
            #for converting the keyword index to vector
            else:
                embeds = self.embeddingsfix(input)
            # output size: (seq_len,batch_size,hidden_dim)
            output, hidden = self.lstm(embeds, (h_0, c_0))

            # size: (seq_len*batch_size,vocab_size)
            output = self.linear1(output.view(seq_len * batch_size, -1))
            return output, hidden
        #for generation with vector
        elif mark == 2:

            #the input is the vector representation of the key word, a 200 demension vector
            embeds = t.tensor(input)

            if hidden is None:
                h_0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
                c_0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
            else:
                h_0, c_0 = hidden

            output, hidden = self.lstm(embeds, (h_0, c_0))

            # size: (seq_len*batch_size,vocab_size)
            output = self.linear1(output.view(1, -1))
            return output, hidden

class TrainingModel_GRUVec(nn.Module):
    #vocab_size mean how many distinct word would appear in the poem
    def __init__(self, vocab_size_train, vocab_size_fix, embedding_dim, hidden_dim):
        super(TrainingModel_GRUVec, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size_train, embedding_dim)
        self.gru = nn.GRU(embedding_dim, self.hidden_dim, num_layers=2)#lstm里面有参数可以通过梯度下降惊醒拟合，通过input得到输入和输出然后计算loss返回提反独下降
        
        #the embeding for the pretrained word vector
        self.embeddingsfix = nn.Embedding(vocab_size_fix, embedding_dim)
        for p in self.embeddingsfix.parameters():
            p.requires_grad = False
        #self.lstmfix = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=2)

        self.linear1 = nn.Linear(self.hidden_dim, vocab_size_fix + vocab_size_train)

    def forward(self, input, hidden=None, mark=None, num=None):
        #for training
        if mark == None:
            #the input usuassly is a two demension tensor the size is [seq_len,batchsize]
            #every row of the input is the first element of all the seqences in the this batch
            seq_len, batch_size = input.size()
            if hidden is None:
                h_0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
                #c_0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
            else:
                #h_0, c_0 = hidden
                h_0 = hidden

            # size: (seq_len,batch_size,embeding_dim) 去掉tensor的第三行
            # our input have specific format, ["<START>"] + department + keyword + tablename + ["<EOP>"]
            # for the training data the format is : ["<START>"] + department + keyword + tablename 
           
            # size: (seq_len,batch_size,embeding_dim)
            #change for the department and the tablename
            embedschange = self.embeddings(t.cat([input[0:2],input[3:]],dim = 0))
           #change for the keyword
            embedsfix = self.embeddingsfix(input[2:3])
            #concatenate the embedding
            embeds=t.cat([embedschange[0:2],embedsfix,embedschange[2:]],dim = 0)

            
            # output size: (seq_len,batch_size,hidden_dim)
            output, hidden = self.gru(embeds, h_0)

            # size: (seq_len*batch_size,vocab_size)
            output = self.linear1(output.view(seq_len * batch_size, -1))
            return output, hidden
        #for generation with index
        elif mark == 1:

            seq_len, batch_size = input.size()
            if hidden is None:
                h_0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
                #c_0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
            else:
               # h_0, c_0 = hidden
               h_0 = hidden

            # size: (seq_len,batch_size,embeding_dim)
            #for converting the department or table index to vector
            if num == 0:    
                embeds = self.embeddings(input)
            #for converting the keyword index to vector
            else:
                embeds = self.embeddingsfix(input)
            # output size: (seq_len,batch_size,hidden_dim)
            output, hidden = self.gru(embeds, h_0)

            # size: (seq_len*batch_size,vocab_size)
            output = self.linear1(output.view(seq_len * batch_size, -1))
            return output, hidden
        #for generation with vector
        elif mark == 2:

            #the input is the vector representation of the key word, a 200 demension vector
            embeds = t.tensor(input)

            if hidden is None:
                h_0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
                #c_0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
            else:
                #h_0, c_0 = hidden
                 h_0  = hidden

            output, hidden = self.gru(embeds, h_0)

            # size: (seq_len*batch_size,vocab_size)
            output = self.linear1(output.view(1, -1))
            return output, hidden

#version for without vector
class TrainingModel(nn.Module):
    #vocab_size mean how many distinct word would appear in the poem
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(TrainingModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=2)#lstm里面有参数可以通过梯度下降惊醒拟合，通过input得到输入和输出然后计算loss返回提反独下降
        self.linear1 = nn.Linear(self.hidden_dim, vocab_size)

    def forward(self, input, hidden=None):
        #print('test')
        #print(input.size())
        seq_len, batch_size = input.size()
        #print(seq_len)
        #print(batch_size)
        if hidden is None:
            #  h_0 = 0.01*torch.Tensor(2, batch_size, self.hidden_dim).normal_().cuda()
            #  c_0 = 0.01*torch.Tensor(2, batch_size, self.hidden_dim).normal_().cuda()
            #print('test')
            #print(type(input))
            h_0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
        else:
            h_0, c_0 = hidden
        # size: (seq_len,batch_size,embeding_dim)
        embeds = self.embeddings(input)
        # output size: (seq_len,batch_size,hidden_dim)
        output, hidden = self.lstm(embeds, (h_0, c_0))

        # size: (seq_len*batch_size,vocab_size)
        #print("beforeenter")
        #print(output.view(seq_len * batch_size, -1).size())
        output = self.linear1(output.view(seq_len * batch_size, -1))
        #print("after_enter")
        #print(output.size())
        return output, hidden

class TrainingModel_RNNVec(nn.Module):
    #vocab_size mean how many distinct word would appear in the poem
    def __init__(self, vocab_size_train, vocab_size_fix, embedding_dim, hidden_dim):
        super(TrainingModel_RNNVec, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size_train, embedding_dim)
        self.gru = nn.RNN(embedding_dim, self.hidden_dim, num_layers=2)#lstm里面有参数可以通过梯度下降惊醒拟合，通过input得到输入和输出然后计算loss返回提反独下降
        
        #the embeding for the pretrained word vector
        self.embeddingsfix = nn.Embedding(vocab_size_fix, embedding_dim)
        for p in self.embeddingsfix.parameters():
            p.requires_grad = False
        #self.lstmfix = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=2)

        self.linear1 = nn.Linear(self.hidden_dim, vocab_size_fix + vocab_size_train)

    def forward(self, input, hidden=None, mark=None, num=None):
        #for training
        if mark == None:
            #the input usuassly is a two demension tensor the size is [seq_len,batchsize]
            #every row of the input is the first element of all the seqences in the this batch
            seq_len, batch_size = input.size()
            if hidden is None:
                h_0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
                #c_0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
            else:
                #h_0, c_0 = hidden
                h_0 = hidden

            # size: (seq_len,batch_size,embeding_dim) 去掉tensor的第三行
            # our input have specific format, ["<START>"] + department + keyword + tablename + ["<EOP>"]
            # for the training data the format is : ["<START>"] + department + keyword + tablename 
           
            # size: (seq_len,batch_size,embeding_dim)
            #change for the department and the tablename
            embedschange = self.embeddings(t.cat([input[0:2],input[3:]],dim = 0))
           #change for the keyword
            embedsfix = self.embeddingsfix(input[2:3])
            #concatenate the embedding
            embeds=t.cat([embedschange[0:2],embedsfix,embedschange[2:]],dim = 0)

            
            # output size: (seq_len,batch_size,hidden_dim)
            output, hidden = self.gru(embeds, h_0)

            # size: (seq_len*batch_size,vocab_size)
            output = self.linear1(output.view(seq_len * batch_size, -1))
            return output, hidden
        #for generation with index
        elif mark == 1:

            seq_len, batch_size = input.size()
            if hidden is None:
                h_0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
                #c_0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
            else:
               # h_0, c_0 = hidden
               h_0 = hidden

            # size: (seq_len,batch_size,embeding_dim)
            #for converting the department or table index to vector
            if num == 0:    
                embeds = self.embeddings(input)
            #for converting the keyword index to vector
            else:
                embeds = self.embeddingsfix(input)
            # output size: (seq_len,batch_size,hidden_dim)
            output, hidden = self.gru(embeds, h_0)

            # size: (seq_len*batch_size,vocab_size)
            output = self.linear1(output.view(seq_len * batch_size, -1))
            return output, hidden
        #for generation with vector
        elif mark == 2:

            #the input is the vector representation of the key word, a 200 demension vector
            embeds = t.tensor(input)

            if hidden is None:
                h_0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
                #c_0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
            else:
                #h_0, c_0 = hidden
                 h_0  = hidden

            output, hidden = self.gru(embeds, h_0)

            # size: (seq_len*batch_size,vocab_size)
            output = self.linear1(output.view(1, -1))
            return output, hidden

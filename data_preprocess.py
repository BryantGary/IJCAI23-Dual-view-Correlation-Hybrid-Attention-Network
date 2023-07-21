# coding: utf-8
# -*- coding: utf-8 -*- 
import os
import shutil
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import collections
import io
import random 

#Mixed views: distributed by patients

filepathMLO = "./preprocesspath_MLO/"
all_pklMLO = os.listdir(filepathMLO)
save_dirMLO = "patientMLO"
filepathCC = "./preprocesspath_CC/"
all_pklCC = os.listdir(filepathCC)
save_dirCC = "patientCC"
labelfileMLO = './label_MLO.txt'
labelfileCC = './label_CC.txt'
preprocesspathMLO= "./patient_MLO/"
preprocesspathCC= "./patient_CC/"
patientlist= "./patientlist1corr.txt" 

def readlist():
  '''read the label as a dict from labelfile'''
  mylist = []
  with open(patientlist, 'r') as f:
    flines = f.readlines()
    for line in flines:
       data= line.split()      
       mylist.append(data[0])
  return mylist

for pkl in all_pklMLO:
	idx = pkl.split("_")[0]
	my_dir = os.path.join(save_dirMLO, idx)
	if not os.path.exists(my_dir):
	  os.makedirs(my_dir)
	pkl1=filepathMLO+ '/'+ pkl      
	shutil.copy(pkl1,my_dir)

for pkl in all_pklCC:
	idx = pkl.split("_")[0]
	my_dir = os.path.join(save_dirCC, idx)
	if not os.path.exists(my_dir):
	  os.makedirs(my_dir)
	pkl1=filepathCC+ '/'+ pkl      
	shutil.copy(pkl1,my_dir)

def readlabelMLO():
  '''read the label as a dict from labelfile'''
  mydict = {}
  with open(labelfileMLO, 'r') as f:
    flines = f.readlines()
    for line in flines:
      data = line.split()
      if int(data[1]) == 0:
        mydict[data[0]] = int(data[1])
      else:
        assert(int(data[1])==2 or int(data[1])==1)
        mydict[data[0]] = int(data[1])-1
  return mydict

def readlabelCC():
  '''read the label as a dict from labelfile'''
  mydict = {}
  with open(labelfileCC, 'r') as f:
    flines = f.readlines()
    for line in flines:
      data = line.split()
      if int(data[1]) == 0:
        mydict[data[0]] = int(data[1])
      else:
        assert(int(data[1])==2 or int(data[1])==1)
        mydict[data[0]] = int(data[1])-1
  return mydict

def show_files(path, train_test,all_files):
    file_list = os.listdir(path)
    for file in file_list:
      cur_path = os.path.join(path, file)
      if os.path.isdir(cur_path):
        show_files(cur_path,train_test, all_files)
      else:
        if file.split("_")[0] in train_test:
          all_files.append(file)

    return all_files


def loadim(fname, preprocesspath,save_dir):
  ''' from preprocess path load fname
  fname file name in preprocesspath
  aug is true, we augment im fliplr, rot 4'''
  ims = []
  with open(os.path.join(save_dir,fname.split("_")[0], fname), 'rb') as inputfile:
    #print(inputfile)
    im = pickle.load(inputfile, encoding='bytes')
    #up_bound = np.random.choice(174)                          #zero out square
    #right_bound = np.random.choice(174)
    img = im
   
    #img[up_bound:(up_bound+50), right_bound:(right_bound+50)] = 0.0
    ims.append(img)
    inputfile.close()
  return ims

def loaddata():
	mydictCC = readlabelCC()
	mydictkeyCC = list(mydictCC.keys())
	mydictvalueCC = list(mydictCC.values())
	mydictMLO = readlabelMLO()
	mydictkeyMLO = list(mydictMLO.keys())
	mydictvalueMLO = list(mydictMLO.values())
	print("mydictCC",len(mydictkeyCC))
	print("mydictMLO",len(mydictkeyMLO))
	all_patientMLO = os.listdir("./patientMLO/")
	all_patientCC = os.listdir("./patientCC/")
	new_patient=[]
	for patientname in all_patientMLO:
		if patientname in all_patientCC:
			new_patient.append(patientname)

	num_patient= len(new_patient)
	print("num_patient",num_patient)
	patient_dict = {}
	flag = 0

	while flag == 0:
		#Patient list
		#random.shuffle(new_patient)
		new_patient = readlist()
		len_train = 0.8*(len(new_patient))		

		train = new_patient[:int(len_train)]
		test = new_patient[int(len_train):]
        
		print("train patient:",len(train))
		print("test patient:",len(test))

		'''Loading Train/ test patient file name'''
		contentstrainMLO = show_files("./patientMLO/", train, [])
		contentstrainCC = show_files("./patientCC/", train, [])
		contentstestCC = show_files("./patientCC/", test, [])
		contentstestMLO = show_files("./patientMLO/", test, [])
   
		for i in range(len(contentstrainMLO)):
			if(contentstrainMLO[i]!= contentstrainCC[i]):
				 print("error!")
    
		for i in range(len(contentstestMLO)):
			if(contentstestMLO[i]!= contentstestCC[i]):
				 print("error!")
         
		mydict_trainMLO= []
		mydict_trainCC= []
		mydict_testMLO = []
		mydict_testCC = []
		dict_trainMLO=collections.OrderedDict()
		dict_trainCC=collections.OrderedDict()
		dict_testMLO=collections.OrderedDict()
		dict_testCC=collections.OrderedDict()
		print("contentstrainMLO:",len(contentstrainMLO))
		print("contentstrainCC:",len(contentstrainCC))
		print("contentstestMLO:",len(contentstestMLO))
		print("contentstestCC:",len(contentstestCC))

		'''Loading train/test patient image-label pair'''         
		for i in range(len(contentstrainMLO)):
			for j in range(len(mydictkeyMLO)):
			  #if not mydictkey[j]+'.pickle' in contents1:
			  #	print(mydictkey[j])         
			  if (mydictkeyMLO[j] +'.pickle' == contentstrainMLO[i]):
			    mydict_trainMLO.append(mydictvalueMLO[j])
			    dict_trainMLO[mydictkeyMLO[j]]=mydictvalueMLO[j]
  
		print("mydict_trainMLO:",len(mydict_trainMLO))
		#for i in mydict_trainMLO:
		  #print(i)
        
		for i in range(len(contentstrainCC)):
			for j in range(len(mydictkeyCC)):
			  #if not mydictkey[j]+'.pickle' in contents1:
			  #	print(mydictkey[j])         
			  if (mydictkeyCC[j] +'.pickle' == contentstrainCC[i]):
			    mydict_trainCC.append(mydictvalueCC[j])
			    dict_trainCC[mydictkeyCC[j]]=mydictvalueCC[j]
  
		print("mydict_trainCC:",len(mydict_trainCC))
		#for i in mydict_trainCC:
		  #print(i)
         
		for i in range(len(contentstestCC)): 
			for j in range(len(mydictkeyCC)):
			  #if not mydictkey[j]+'.pickle' in contents2:
			  #	print(mydictkey[j])         
			  if (mydictkeyCC[j] +'.pickle' == contentstestCC[i]):
			    mydict_testCC.append(mydictvalueCC[j])
			    dict_testCC[mydictkeyCC[j]]=mydictvalueCC[j]
                                                
		print("mydict_testCC:",len(mydict_testCC))

		for i in range(len(contentstestMLO)): 
			for j in range(len(mydictkeyMLO)):
			  #if not mydictkey[j]+'.pickle' in contents2:
			  #	print(mydictkey[j])         
			  if (mydictkeyMLO[j] +'.pickle' == contentstestMLO[i]):
			    mydict_testMLO.append(mydictvalueMLO[j])
			    dict_testMLO[mydictkeyMLO[j]]=mydictvalueMLO[j]
                                                
		print("mydict_testMLO:",len(mydict_testMLO))

		#for i in mydict_test:
		  #print(i)   

		'''Error check'''    
		#print("dict_train:",len(dict_train.keys()))  
		for i in contentstrainMLO:
			if not i.split('.')[0] in mydictkeyMLO:
			  print(i)
		for i in contentstrainCC:
			if not i.split('.')[0] in mydictkeyCC:
			  print(i)
		for i in contentstestMLO:
			if not i.split('.')[0] in mydictkeyMLO:
			  print(i)    
		for i in contentstestCC:
			if not i.split('.')[0] in mydictkeyCC:
			  print(i)                  

		'''Pickle to numpy''' 
		traindataMLO, trainlabelMLO = np.zeros((len(contentstrainMLO),256,256)), np.zeros((len(contentstrainMLO),))
		testdataMLO, testlabelMLO =  np.zeros((len(contentstestMLO),256,256)), np.zeros((len(contentstestMLO),))
		traindataCC, trainlabelCC = np.zeros((len(contentstrainCC),256,256)), np.zeros((len(contentstrainCC),))
		testdataCC, testlabelCC =  np.zeros((len(contentstestCC),256,256)), np.zeros((len(contentstestCC),))

		traincountMLO = 0
		print("train_numMLO:",len(contentstrainMLO))
		for i in range(len(contentstrainMLO)):
				ims = loadim(contentstrainMLO[i],preprocesspath=preprocesspathMLO,save_dir=save_dirMLO)
				for im in ims:
				  traindataMLO[traincountMLO, :, :] = im
				  trainlabelMLO[traincountMLO] = mydict_trainMLO[i]      
				  traincountMLO += 1
		assert(traincountMLO==traindataMLO.shape[0])

		traincountCC = 0
		print("train_numCC:",len(contentstrainCC))
		for i in range(len(contentstrainCC)):
				ims = loadim(contentstrainCC[i],preprocesspath=preprocesspathCC,save_dir=save_dirCC)
				for im in ims:
				  traindataCC[traincountCC, :, :] = im
				  trainlabelCC[traincountCC] = mydict_trainCC[i]      
				  traincountCC += 1
		assert(traincountCC==traindataCC.shape[0])

		testcountMLO = 0
		print("test_numMLO:",len(contentstestMLO))
		for j in range(len(contentstestMLO)):
				ims = loadim(contentstestMLO[j],preprocesspath=preprocesspathMLO,save_dir=save_dirMLO)
				testdataMLO[testcountMLO,:,:] = ims[0]
				testlabelMLO[testcountMLO] = mydict_testMLO[j]
				testcountMLO += 1
				
		assert(testcountMLO==testdataMLO.shape[0])

		testcountCC = 0
		print("test_numCC:",len(contentstestCC))
		for j in range(len(contentstestCC)):
				ims = loadim(contentstestCC[j],preprocesspath=preprocesspathCC,save_dir=save_dirCC)
				testdataCC[testcountCC,:,:] = ims[0]
				testlabelCC[testcountCC] = mydict_testCC[j]
				testcountCC += 1
				
		assert(testcountCC==testdataCC.shape[0])

		trYMLO = trainlabelMLO.reshape((trainlabelMLO.shape[0],1))
		teYMLO = testlabelMLO.reshape((testlabelMLO.shape[0],1))
		trYCC = trainlabelCC.reshape((trainlabelCC.shape[0],1))
		teYCC = testlabelCC.reshape((testlabelCC.shape[0],1))

	return traindataMLO,trainlabelMLO,testdataMLO,testlabelMLO,traindataCC,trainlabelCC,testdataCC,testlabelCC
 
#traindataMLO,trainlabelMLO,testdataMLO,testlabelMLO,traindataCC,trainlabelCC,testdataCC,testlabelCC = loaddata()






import pickle

with open("data1.pickle",'rb') as fil:
    dat = pickle.load(fil , encoding = 'latin1')

#print(dat)
#print(dir(dat))
#print(dat.keys())
print(dat['params'])

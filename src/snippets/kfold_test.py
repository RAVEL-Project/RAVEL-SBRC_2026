from numpy import array
from sklearn.model_selection import KFold


def kFold_ing(k,data, trainProportion= 50,random_state=1, shuffle=True):

    # prepare cross validation
    kfold = KFold(k, shuffle, random_state)
    # enumerate splits
    for train, test in kfold.split(data):
        print('train: %s, test: %s' % (data[train], data[test]))
    return data


import random
def kFold( k, data, seed = 22):
        
    size = len(data)
    subset_size = round(size / k)
    random.Random(seed).shuffle(data)
    subsets = [data[x:x+subset_size] for x in range(0, len(data), subset_size)]
    print(subsets)
    
    kfolds = []
    for i in range(k):
        test = subsets[i]
        train = []
        for subset in subsets:
            if subset != test:
                train.append(subset)
        kfolds.append((train,test))    
    return kfolds



def kFold_er( k, data, testProp=50, seed = 22):
        
    size = len(data)
    subset_size = round(size / k)
    random.Random(seed).shuffle(data)
    subsets = [data[x:x+subset_size] for x in range(0, len(data), subset_size)]
    datasets=[]
    
    for fold in range (k):

        count=0
        n_testFolds= int(round(k*testProp/100 ,0))
        train=[]
        test=[]
        
        for s in range(k):
            
            pos=fold +s
            pos = pos%k      
            if (count >= n_testFolds):
                train+=subsets[pos]
            else:
                test+=subsets[pos]
            count+=1

        datasets.append([train,test])

            
    return datasets
print( kFold_er(6,[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],30) )
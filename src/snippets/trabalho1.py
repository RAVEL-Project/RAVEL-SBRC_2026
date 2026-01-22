import random
import wisardpkg as wp
from datetime import datetime

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


def dataLoad_er(dataTrain='mnist_train.csv', dataTest='mnist_test.csv',th=127):   
    
    trainSet=[]

    with open(dataTrain, 'r', encoding="utf8") as mnist_train:
        mnist_train.readline()
        
        for line in mnist_train:

            col=line.split(",")
            #labels.append(str(col[0]))
            v=[]
            size = len(col)

            for i in range(size):
                if i != 0:               
                        
                    if int(col [i]) < th:
                        v.append(0)
                    else:
                        v.append(1)

                else: 
                    v.append(col[i])                    
            trainSet.append(v)
 

    testSet = []

    with open(dataTest, 'r', encoding="utf8") as mnist_test:
        mnist_test.readline()
        for line in mnist_test:

            coltest=line.split(",")
            #print(coltest)
            
            #labelsTest.append(str(coltest[0]))
            vtest=[]
            sizeTest=len(coltest)

            for i in range(sizeTest):

                if i != 0:               
                        
                    if int(coltest [i]) < th:
                        vtest.append(0)
                    else:
                        vtest.append(1)

                else: 
                    vtest.append(coltest[i])

            testSet.append(vtest)
                
    return (trainSet,testSet)


def theThing(trainSet,testSet,labels,labelsTest,nRounds=10,addressSize=6,kfold="no"):

    cfmatrix=[]
    acc=[]
    n=0
    count=0

    for item in range(10):
        
            cfmatrix.append([0]*10)
        

    #----------------The Thing----------------#
    for round in range (nRounds):
        print("processing round ",round)
        timeStartTrain = datetime.now()
        timeStartTrain.microsecond

        ignoreZero  = False # optional; causes the rams to ignore the address 0
        # False by default for performance reasons,
        # when True, WiSARD prints the progress of train() and classify()
        verbose = False
        wsd = wp.Wisard(addressSize, ignoreZero=ignoreZero, verbose=verbose)
        
        # train using the input data
        wsd.train(trainSet,labels)

        timeEndTrain = datetime.now()
        timeEndTrain.microsecond


        timeStartClassify = datetime.now()
        timeStartClassify.microsecond

        out = wsd.classify(testSet)

        timeEndClassify = datetime.now()
        timeEndClassify.microsecond

    #----------------The Thing----------------#

        for i,d in enumerate(labelsTest):
            cfmatrix[int(d)][int(out[i])]+=1
            n+=1
            if(out[i]==d):
                count+=1/nRounds
                p=count/n
                q=1-p
                var=p*q

        acc.append(p)
        f=open("netMapsTh_addSz_"+str(addressSize)+str(kfold)+".txt",'a')
        f.write("Wisard: round "+str(round) +" --  localAccuracy = "+str(p)+ "   localSTD = "+str(var** 0.5)+"\n")
        f.write("Execution time train --> "+ str(timeEndTrain-timeStartTrain) +"  Classify --> "+ str(timeEndClassify-timeStartClassify)+"\n")
        f.write(str(wsd.json(True,"/data"))+"\n\n")
        f.close()
        #print("acuracy = ", p, "    desvio pardão = ", var** 0.5)
        #print (len(acc))

    print("Summarizing Statistics")
    accTotal=0
    for line in range(nRounds):
        accTotal+=acc[line]/nRounds
        #print (cfmatrix[line])
        qt=1-accTotal


    v=nRounds*accTotal*qt
    #print("acuracy = ", accTotal, "    desvio pardão = ",v**0.5)
    fd=open("SummaryTh_addSz_"+str(addressSize)+str(kfold)+".txt",'a')
    fd.write("Wisard: threshhold 10 rounds -- accuracy  "+ str(accTotal) +"    desvio pardão = " + str(v**0.5)+"\n"+"Confusion Matrix : \n")
    fd.write("       0     1     2     3     4     5     6     7     8     9 \n")
    for c in range(10):
        fd.write("  "+str(c)+"  ")
        for l in range(10):
            fd.write('{0:05d}'.format(cfmatrix[c][l])+" ")
        fd.write("\n")
    fd.close()
        
mnistData = dataLoad_er()
size=len(mnistData[0][0])


#-------------------no kfold-----------------------#
dataTrain=[]
dataTest=[]
labelsTest=[]
labelsTrain=[]

for pos in mnistData[0]:
    
    dataTrain.append(pos[1:size])
    labelsTrain.append(pos[0])

for p in mnistData[1]:
    #print(len(p),size)

    dataTest.append(p[1:size])
    labelsTest.append(p[0])

addSizes=[2,3,4,5,6,7,8,9]

for add in addSizes:
    theThing(dataTrain,dataTest,labelsTrain,labelsTest,10, add)


#---------------- kfold ------------------#
data = mnistData[0] + mnistData[1]
proportions=[50,40,30,20,10]

for prop in proportions:
    k=10
    kfoldData=kFold_er(10,data)
    foldCont=0
    for fold in kfoldData:
        
        trainFold = fold[0]
        testFold=fold[1]


        dataTrain=[]
        dataTest=[]
        labelsTest=[]
        labelsTrain=[]

        for pos in testFold:
            
            dataTrain.append(pos[1:size])
            labelsTrain.append(pos[0])

        for p in trainFold:
            #print(len(p),size)

            dataTest.append(p[1:size])
            labelsTest.append(p[0])
        
        for add in addSizes:
            theThing(dataTrain,dataTest,labelsTrain,labelsTest,1, add,"kfold"+str(foldCont)+str(prop))
        foldCont+=1
     
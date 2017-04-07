from numpy import *
import kNNHandWriting
'''
class KnnHandWriting(kNNBase.KnnBase):
    def __init__(self, kval):
        kNNBase.KnnBase.__init__(self)
        self.k = kval
        self.imgReader = ""

    def create_data_set(self, train_images_file, train_label_file, test_images_file, test_label_file):
        self.imgReader = imgReader.TrainDataSet(train_images_file, train_label_file, test_images_file, test_label_file)
        self.imgReader.read_train_img()
        self.imgReader.read_train_label()
        self.imgReader.read_test_img()
        self.imgReader.read_test_label()

    def classfy(self, inX, trainset, labels):
        dataSetSize = len(trainset)
        #print ("dataSetSize:" + str(dataSetSize))
        #print (trainset)
        diffMat = tile(inX, (dataSetSize, 1)) - trainset
        #print(diffMat)
        sqDiffMat = diffMat ** 2
        #print(sqDiffMat)
        sqDistance = sqDiffMat.sum(axis=1)
        #print(sqDistance)
        distances = sqDistance ** 0.5
        #print(distances)
        sortedDistanceIndicies = distances.argsort()
        #print(sortedDistanceIndicies)
        classCount = {}
        for i in range(self.k):
            voteILabel = labels[sortedDistanceIndicies[i]]
            #print("---" + voteILabel)
            classCount[voteILabel] = classCount.get(voteILabel, 0) + 1
            #print("---" + str(classCount[voteILabel]))
        sortedClassCount = sorted(classCount.iteritems(),
                                  key=operator.itemgetter(1),
                                  reverse=True)
        #print (sortedClassCount)
        return sortedClassCount[0]

'''

knn = kNNHandWriting.KnnHandWriting(15)
knn.create_data_set("/Users/fangzhengzhu/tensor/GoMNIST/data/train-images-idx3-ubyte",
                    "/Users/fangzhengzhu/tensor/GoMNIST/data/train-labels-idx1-ubyte",
                    "/Users/fangzhengzhu/tensor/GoMNIST/data/t10k-images-idx3-ubyte",
                    "/Users/fangzhengzhu/tensor/GoMNIST/data/t10k-labels-idx1-ubyte")

count = 0
total = 10000
start = 0
for i in range(start,start+total):
    result = knn.classfy(knn.imgReader.get_test_set()[i],
            knn.imgReader.get_train_set(),
            knn.imgReader.get_train_label())
    #print (result)
    print (str(i) + ":" + str(knn.imgReader.get_test_label()[i] == result[0]))
    if knn.imgReader.get_test_label()[i] == result[0] :
        count += 1

print ("Total: "+str(total) +", correct:" + str(count))


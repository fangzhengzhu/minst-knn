from numpy import *
import operator
import  kNNBase
import imgReader

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

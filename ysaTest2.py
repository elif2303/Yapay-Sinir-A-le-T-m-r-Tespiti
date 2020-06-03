import numpy as np
import cv2 
from PIL import Image
from matplotlib import pyplot as plt
from numpy import save
from numpy import load
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report


class YSATest():

    def __init__(self):

        self.patch_boyut = 32 

        self.loadedFeatureTumor5 = np.load('hasta4_featureTumor_{}.npy'.format(self.patch_boyut))
        self.loadedFeatureArka5 = np.load('hasta4_featureArkaplan_{}.npy'.format(self.patch_boyut))

    
    def testData(self):
    
        self.testData = np.concatenate((self.loadedFeatureArka5,self.loadedFeatureTumor5), axis=0)    
        print(self.testData.shape)                                         #(2044,324)    toplam voksel sayıları, hog özellik sayısı 

    
    def tumorEtiketVektor(self):             #01
    
        self.tumorEtiketVektor = np.concatenate((np.zeros((self.loadedFeatureTumor5.shape[0],1), dtype=np.float), np.ones((self.loadedFeatureTumor5.shape[0],1), dtype=np.float)), axis=1)    
        print(self.tumorEtiketVektor.shape)                                 #(1039,2)      tumor vokselsayıları,2
        print(self.tumorEtiketVektor)
    
    def arkaplanEtiketVektor(self):         #10

        self.arkaplanEtiketVektor = np.concatenate((np.ones((self.loadedFeatureArka5.shape[0],1), dtype=np.float),np.zeros((self.loadedFeatureArka5.shape[0],1), dtype=np.float)), axis=1)
        print(self.arkaplanEtiketVektor.shape)                              #(1005,2)      arkaplan voksel sayıları,2  
        print(self.arkaplanEtiketVektor)
    
    def etiketVektor(self):
        self.testEtiket = np.concatenate((self.arkaplanEtiketVektor , self.tumorEtiketVektor) , axis=0)
        print(self.testEtiket.shape)                                      #(2044,2)      toplam voksel sayılar,1

    
    def test(self):                                                             

        self.YSA = cv2.ml.ANN_MLP_load('YSA4')      
                
        print(self.YSA.getTrainMethod())                #0
        print(self.YSA.getLayerSizes())                 #[[324][100][2]]
        print(self.YSA.getTermCriteria())               #(3, 300, 0.001)
        
        self.retval, self.result = self.YSA.predict(self.testData)  
        print(self.result)                #sonuç matrisi        
        print(self.result.shape)          #(2044,1)

        self.result = np.round(self.result)             #yuvarlama yapıldı
                
        self.karmasaMatrisi = np.zeros([2,2], dtype=np.int)

        for x in range(0,len(self.testData)):
            self.pred = self.result[x]           # [1 0] gibi bir YSA tahmin dizisi
            self.truth = self.testEtiket[x]      # [1 0] gibi bir gerçek etiket dizisi
          
            if self.truth[0]==1 and self.truth[1]==0 and self.pred[0]==1 and self.pred[1]==0:    #[1 0] arka plan etiketiydi ve arka plan  tahmin edilmişse TN'ye sayılır
                self.karmasaMatrisi[1,1]=self.karmasaMatrisi[1,1]+1 
            if self.truth[0]==0 and self.truth[1]==1 and self.pred[0]==0 and self.pred[1]==1:   #[0 1] tümör etiketiydi ve tümör tahmin edildiyse TP'ye sayılır
                self.karmasaMatrisi[0,0]=self.karmasaMatrisi[0,0]+1 
            if self.truth[0]==1 and self.truth[1]==0 and self.pred[0]==0 and self.pred[1]==1:    #[1 0] arka plan etiketiydi ve tümör tahmin edildiyse FP'ye sayılır
                self.karmasaMatrisi[0,1]=self.karmasaMatrisi[0,1]+1 
            if self.truth[0]==0 and self.truth[1]==1 and self.pred[0]==1 and self.pred[1]==0:    #[0 1] tümör etiketiydi ve arka plan tahmin edildiyse FN'ye sayılır
                self.karmasaMatrisi[1,0]=self.karmasaMatrisi[1,0]+1 

        print(self.karmasaMatrisi) 
        self.kesinlik = 100*(self.karmasaMatrisi[0,0]+self.karmasaMatrisi[1,1])/np.sum(self.karmasaMatrisi,axis=None)          #sistemin yüzde başarısı
        print(self.kesinlik)
        

        """
        self.check = np.concatenate((self.result , self.testEtiket) , axis=1)
        print(self.check) 
        """

    def accuracy(self):
       
        self.result = np.array(self.result , dtype=np.int)
        self.testEtiket = np.array(self.testEtiket , dtype=np.int)        
        
        for x in range(0,len(self.testData)):
            self.pred = self.result[x]          
            self.truth = self.testEtiket[x]
        
            self.confusion_matrix =confusion_matrix(self.pred , self.truth)     
            self.accuracy_score = accuracy_score(self.pred, self.truth)
           

        print(self.confusion_matrix)
        print(self.accuracy_score)
     

        
        self.check = np.concatenate((self.result , self.testEtiket) , axis=1)
        print(self.check)

if __name__ == "__main__":
    Test = YSATest()
    Test.testData()
    Test.tumorEtiketVektor()
    Test.arkaplanEtiketVektor()
    Test.etiketVektor()
    Test.test()
    #Test.accuracy()
    
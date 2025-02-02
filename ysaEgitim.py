import numpy as np
import cv2 as cv
from PIL import Image
from matplotlib import pyplot as plt
from numpy import save
from numpy import load


class YSAEgitim():

    def __init__(self):
        self.patch_boyut = 32
        self.loadedFeatureTumor1 = np.load('hasta1_featureTumor_{}.npy'.format(self.patch_boyut))
        self.loadedFeatureArka1 = np.load('hasta1_featureArkaplan_{}.npy'.format(self.patch_boyut))

        self.loadedFeatureTumor2 = np.load('hasta2_featureTumor_{}.npy'.format(self.patch_boyut))
        self.loadedFeatureArka2 = np.load('hasta2_featureArkaplan_{}.npy'.format(self.patch_boyut))

        self.loadedFeatureTumor3 = np.load('hasta3_featureTumor_{}.npy'.format(self.patch_boyut))
        self.loadedFeatureArka3 = np.load('hasta3_featureArkaplan_{}.npy'.format(self.patch_boyut))

        self.loadedFeatureTumor4 = np.load('hasta4_featureTumor_{}.npy'.format(self.patch_boyut))
        self.loadedFeatureArka4 = np.load('hasta4_featureArkaplan_{}.npy'.format(self.patch_boyut))

    def tumor(self):

        self.tumorVokselSayısı1 = np.concatenate((self.loadedFeatureTumor1,self.loadedFeatureTumor2), axis=0)
        self.tumorVokselSayısı2 = np.concatenate((self.loadedFeatureTumor3,self.loadedFeatureTumor4), axis=0)

        self.tumor = np.concatenate((self.tumorVokselSayısı1, self.tumorVokselSayısı2),axis=0)

        print(self.tumor.shape)             #(4052,324) toplam tümör voksel sayısı, hog özellik sayısı 

    def arkaplan(self):

        self.arkaplanVokselSayısı1 = np.concatenate((self.loadedFeatureArka1,self.loadedFeatureArka2), axis=0)
        self.arkaplanVokselSayısı2 = np.concatenate((self.loadedFeatureArka3,self.loadedFeatureArka4), axis=0)

        self.arkaplan = np.concatenate((self.arkaplanVokselSayısı1, self.arkaplanVokselSayısı2),axis=0)

        print(self.arkaplan.shape)              #(4015,324) toplam arkaplan voksel sayısı, hog özellik sayısı

    def egitimData(self):

        self.egitimData = np.concatenate((self.arkaplan,self.tumor), axis=0)

        print(self.egitimData.shape)            #(8067,324) toplam voksel sayıları, hog özellik sayısı
    

    def tumorEtiketVektor(self):             #01

        self.zerosTumor = np.zeros((self.tumor.shape[0],1), dtype=np.float)
        self.onesTumor = np.ones((self.tumor.shape[0],1), dtype=np.float)
        
        self.tumorEtiketVektor = np.concatenate((self.zerosTumor, self.onesTumor) , axis=1)
        print(self.tumorEtiketVektor.shape)                                 #(4052,2)      tumor vokselsayıları,2
        
    def arkaplanEtiketVektor(self):          #10
        
        self.zerosArka = np.zeros((self.arkaplan.shape[0],1), dtype=np.float)
        self.onesArka = np.ones((self.arkaplan.shape[0],1), dtype=np.float)

        self.arkaplanEtiketVektor = np.concatenate((self.onesArka, self.zerosArka), axis=1)
        print(self.arkaplanEtiketVektor.shape)                              #(4015,2)      arkaplan voksel sayıları,2   
    
    def etiketVektor(self):

        self.etiketVektor = np.concatenate((self.arkaplanEtiketVektor , self.tumorEtiketVektor) , axis=0)

        print(self.etiketVektor.shape)          #(8067,1) toplam voksel sayılar,1

    def createYSA(self):

        self.YSA = cv.ml.ANN_MLP_create()

    def train(self):
     
        self.layer_sizes = np.int64([324, 100 , 2])                                 #özellik sayısı(girdi) , gizli katman noran sayısı, çıktı
        self.YSA.setLayerSizes(self.layer_sizes)                                     #Giriş ve çıkış katmanları dahil olmak üzere her katmanda nöron sayısını belirten tamsayı vektörü
        
        self.YSA.setActivationFunction(cv.ml.ANN_MLP_SIGMOID_SYM ,0,0)                   #Etkinleştirme fonksiyonu
        
        self.YSA.setTermCriteria((cv.TermCriteria_MAX_ITER+cv.TermCriteria_EPS, 300, 0.0001))                #EPS hata değeri 0.01 altına düştüğünde dur 
           
        self.etiketVektor = np.array(self.etiketVektor, dtype=np.float32)
        self.egitimData = np.array(self.egitimData, dtype=np.float32)
        
        self.YSA.setTrainMethod(cv.ml.ANN_MLP_BACKPROP, 0.0001)                     #Eğitim yöntemini ve ortak parametreleri ayarlar
        self.YSA.train(self.egitimData , cv.ml.ROW_SAMPLE , self.etiketVektor)      #istatiksel modeli eğitir , ROW_SAMPLE örneklerin satırda yer aldığı anlamında , bir satırdaki 1x324 float değer 1 örnektir anlamında.
        self.test = self.YSA.save('YSA5')


        print(self.YSA.getTrainMethod())	
        print(self.YSA.getLayerSizes())	
        print(self.YSA.getTermCriteria())	


if __name__ == "__main__":
    
    Egitim = YSAEgitim()    
    
    Egitim.tumor()
    Egitim.arkaplan()
    Egitim.egitimData()
    Egitim.tumorEtiketVektor()
    Egitim.arkaplanEtiketVektor()
    Egitim.etiketVektor()
    Egitim.createYSA()
    Egitim.train()
    




    
     
import numpy as np
import cv2 
from PIL import Image
from matplotlib import pyplot as plt
from numpy import save
from numpy import load

class HOG:  
    
    def __init__(self):
        
        self.hasta = int(input("Toplamda 5 hastamız vardır.İstediğiniz hasta sayısını giriniz(0-4):"))
        
        if self.hasta > 4:
            print("Yanlış rakam girildi.(0-4)")
        
        if self.hasta == 0 :
            self.tumorVoksel = np.load('hasta1_tumor_64.npy ')
            self.arkaplanVoksel = np.load('hasta1_arkaplan_64.npy ')
        elif self.hasta == 1 :
            self.tumorVoksel = np.load('hasta2_tumor_64.npy ')
            self.arkaplanVoksel = np.load('hasta2_arkaplan_64.npy ')
        elif self.hasta == 2 :
            self.tumorVoksel = np.load('hasta3_tumor_64.npy ')
            self.arkaplanVoksel = np.load('hasta3_arkaplan_64.npy ')
        elif self.hasta == 3 :
            self.tumorVoksel = np.load('hasta4_tumor_64.npy ')
            self.arkaplanVoksel = np.load('hasta4_arkaplan_64.npy ')
        elif self.hasta == 4 :
            self.tumorVoksel = np.load('hasta5_tumor_64.npy ')
            self.arkaplanVoksel = np.load('hasta5_arkaplan_64.npy ')
        
    def tumorHog(self):
        
        self.patch_boyut = 32        

        self.featureTumorList=np.zeros((self.tumorVoksel.shape[0],324), dtype=np.float)

        for self.patch in range(len(self.tumorVoksel)):
            
            self.patch_image = self.tumorVoksel[self.patch,:,:]
            self.patch_image = self.patch_image*255                                                          #0 ila 255 arasına çekmek için
            self.patch_image = np.array(self.patch_image[16:48,16:48] , dtype=np.uint8)

            self.hog = cv2.HOGDescriptor((self.patch_boyut,self.patch_boyut),(16,16),(8,8),(8,8),9)          #varsayılan parametrelerle hog tanımlayıcı ve dedektör oluşturur
            self.featureTumor = self.hog.compute(self.patch_image) 
            self.featureTumorList[self.patch,:] = self.featureTumor.transpose()

        print(self.featureTumorList)       
        print(len(self.featureTumorList))

        plt.imshow(self.patch_image , 'gray')                                 
        plt.show()

        plt.plot(self.featureTumor)
        plt.show()

    def arkaplanHog(self):
        
        self.patch_boyut = 32
    
        self.featureArkaList = np.zeros((self.arkaplanVoksel.shape[0],324) , dtype=np.float) 
       
        for self.patch in range(len(self.arkaplanVoksel)):
            
            self.patch_image = self.arkaplanVoksel[self.patch,:,:]
            self.patch_image = self.patch_image*255                           
            self.patch_image = np.array(self.patch_image[16:48,16:48] , dtype=np.uint8)

            self.hog = cv2.HOGDescriptor((self.patch_boyut,self.patch_boyut),(16,16),(8,8),(8,8),9)       
            self.featureArka = self.hog.compute(self.patch_image)
            self.featureArkaList[self.patch,:] = self.featureArka.transpose()
           
        print(self.featureArkaList)
        print(len(self.featureArkaList)) 
        
        plt.imshow(self.patch_image , 'gray')                                 
        plt.show()  

        plt.plot(self.featureArka)
        plt.show()

    def file (self):
        
        self.patch_boyut = 32
        
        if self.hasta == 0 :
            np.save('hasta1_featureTumor_{}.npy'.format(self.patch_boyut) , self.featureTumorList , 'ab+')
            self.loadedFeatureTumor1 = np.load('hasta1_featureTumor_{}.npy'.format(self.patch_boyut))

            np.save('hasta1_featureArkaplan_{}.npy'.format(self.patch_boyut) , self.featureArkaList , 'ab+')
            self.loadedFeatureArka1 = np.load('hasta1_featureArkaplan_{}.npy'.format(self.patch_boyut))

        elif self.hasta == 1 :

            np.save('hasta2_featureTumor_{}.npy'.format(self.patch_boyut) , self.featureTumorList , 'ab+')
            self.loadedFeatureTumor2 = np.load('hasta2_featureTumor_{}.npy'.format(self.patch_boyut))

            np.save('hasta2_featureArkaplan_{}.npy'.format(self.patch_boyut) , self.featureArkaList , 'ab+')
            self.loadedFeatureArka2 = np.load('hasta2_featureArkaplan_{}.npy'.format(self.patch_boyut))

        elif self.hasta == 2 :
            np.save('hasta3_featureTumor_{}.npy'.format(self.patch_boyut) , self.featureTumorList , 'ab+')
            self.loadedFeatureTumor3 = np.load('hasta3_featureTumor_{}.npy'.format(self.patch_boyut))

            np.save('hasta3_featureArkaplan_{}.npy'.format(self.patch_boyut) , self.featureArkaList , 'ab+')
            self.loadedFeatureArka3 = np.load('hasta3_featureArkaplan_{}.npy'.format(self.patch_boyut))

        elif self.hasta == 3:
            np.save('hasta4_featureTumor_{}.npy'.format(self.patch_boyut) , self.featureTumorList , 'ab+')
            self.loadedFeatureTumor4 = np.load('hasta4_featureTumor_{}.npy'.format(self.patch_boyut))

            np.save('hasta4_featureArkaplan_{}.npy'.format(self.patch_boyut) , self.featureArkaList , 'ab+')
            self.loadedFeatureArka4 = np.load('hasta4_featureArkaplan_{}.npy'.format(self.patch_boyut))

        elif self.hasta == 4:
            np.save('hasta5_featureTumor_{}.npy'.format(self.patch_boyut) , self.featureTumorList , 'ab+')
            self.loadedFeatureTumor5 = np.load('hasta5_featureTumor_{}.npy'.format(self.patch_boyut))

            np.save('hasta5_featureArkaplan_{}.npy'.format(self.patch_boyut) , self.featureArkaList , 'ab+')
            self.loadedFeatureArka5 = np.load('hasta5_featureArkaplan_{}.npy'.format(self.patch_boyut))

if __name__ == "__main__":
    
    Hasta1 = HOG()
    
    Hasta1.tumorHog()
    Hasta1.arkaplanHog()
    Hasta1.file()
            
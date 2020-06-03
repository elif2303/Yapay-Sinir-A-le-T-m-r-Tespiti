import numpy as np
import pydicom
import SimpleITK as sitk
import cv2 
from matplotlib import pyplot as plt
from PIL import Image
import os
from scipy import misc
from numpy import save
from numpy import load
import random

    

img1 = sitk.ReadImage('D:/mr/TCGA-GBM/TCGA-02-0011-1.HASTA - GROUND TRUTH YAPILDI/02-01-1998-MRI BRAIN WWO CONTRAMR-31709/3-GROUND TRUTH YAPILDI -AX T2 FSE-77488/Hasta1.mha') 
img2 = sitk.ReadImage('D:/mr/TCGA-GBM/TCGA-02-0034-2.HASTA - GROUND TRUTH YAPILDI/07-27-1997-MRI BRAIN WWO CONTRAMR-39956/10-GROUND TRUTH YAPILDI-AX T2 FSE-01030/Hasta2.mha') 
img3 = sitk.ReadImage('D:/mr/TCGA-GBM/TCGA-02-0047-3.HASTA - GROUND TRUTH YAPILDI/12-15-1998-MRI BRAIN WWO CONTR-70492/3-GROUND TRUTH YAPILDI-AX T2 FSE-79920/Hasta3.mha') 
img4 = sitk.ReadImage('D:/mr/TCGA-GBM/TCGA-02-0085-4.HASTA-GROUNDTRUTH YAPILDI/01-30-1999-MRI BRAIN WWO CONTR-29853-GROUNDTRUTH YAPILDI/3-AX T2 FSE-87118-GROUNDTRUTH YAPILDI/Hasta4.mha') 
img5 = sitk.ReadImage('D:/mr/TCGA-GBM/TCGA-02-0048-5.HASTA -- GROUND TRUTH YAPILDI/01-29-1999-MRI BRAIN WWO CONTR-02900/3-AX T2 FSE-16941- GROUND TRUTH YAPILDI/Hasta5.mha') 

img = [ img1, img2, img3, img4, img5]

pathDicom1 = ('D:/mr/TCGA-GBM/TCGA-02-0011-1.HASTA - GROUND TRUTH YAPILDI/02-01-1998-MRI BRAIN WWO CONTRAMR-31709/3-GROUND TRUTH YAPILDI -AX T2 FSE-77488')
pathDicom2 = ('D:/mr/TCGA-GBM/TCGA-02-0034-2.HASTA - GROUND TRUTH YAPILDI/07-27-1997-MRI BRAIN WWO CONTRAMR-39956/10-GROUND TRUTH YAPILDI-AX T2 FSE-01030' )
pathDicom3 = ('D:/mr/TCGA-GBM/TCGA-02-0047-3.HASTA - GROUND TRUTH YAPILDI/12-15-1998-MRI BRAIN WWO CONTR-70492/3-GROUND TRUTH YAPILDI-AX T2 FSE-79920' )
pathDicom4 = ('D:/mr/TCGA-GBM/TCGA-02-0085-4.HASTA-GROUNDTRUTH YAPILDI/01-30-1999-MRI BRAIN WWO CONTR-29853-GROUNDTRUTH YAPILDI/3-AX T2 FSE-87118-GROUNDTRUTH YAPILDI' )
pathDicom5 = ('D:/mr/TCGA-GBM/TCGA-02-0048-5.HASTA -- GROUND TRUTH YAPILDI/01-29-1999-MRI BRAIN WWO CONTR-02900/3-AX T2 FSE-16941- GROUND TRUTH YAPILDI' )

pathDicom = [ pathDicom1, pathDicom2, pathDicom3, pathDicom4, pathDicom5]

hasta = int(input("Toplamda 5 hastamız vardır.İstediğiniz hasta sayısını giriniz(0-4):"))

if hasta > 4:
    print("Yanlış rakam girildi.(0-4)")
            

reader = sitk.ImageSeriesReader()                           #reader oluşturuldu
filenamesDICOM = reader.GetGDCMSeriesFileNames(pathDicom[hasta])   
reader.SetFileNames(filenamesDICOM)                         

ArrayDicom = reader.Execute()                               #orijinal dicom uzantılı görüntüler okunur
ArrayDicom = sitk.GetArrayFromImage(ArrayDicom)            

sad1 = np.size(ArrayDicom, axis = 0)                          #(0,1,2)   axis=0=24        derinlik 
sad2 = np.size(ArrayDicom, axis = 1)                          #(0,1,2)   axis=1=256       satır 
sad3 = np.size(ArrayDicom, axis = 2)                          #(0,1,2)   axis=2=256       sütun

ArrayDicomf=ArrayDicom.astype(float)                        #orijinal görüntü float tipine çevrildi
print("max değer:", np.amax(ArrayDicomf))               
print("min değer:", ArrayDicomf.min())                      #0.0           "-" değer var mı kontrolü

for i in range(0, len(ArrayDicomf)):                                                                          
    plt.subplot(6,6,i+1),plt.imshow(ArrayDicomf[i]/4.0,'gray')    
    plt.xticks([]),plt.yticks([])
plt.show() 

###########ArrayDicomf histogram grafiği çizme###################################################################################

flattened = ArrayDicomf.reshape([sad1*sad2*sad3])  # Flatten matrices

rhist,bins,_ = plt.hist(flattened, bins = 256)                 #bins = yoğunluk değerleri , rhist= pixel sayıları
plt.show() 

##########kümülatif toplam ve normalizasyonu##################################################################################################

kumulatifToplam = np.cumsum(np.array(rhist))         #kümülatif toplam    , her görüntü için ortak bir eşik değeri bulabilmek için

m = np.min(kumulatifToplam)                          #kümülatif toplam normalize edildi
n = np.max(kumulatifToplam)
kumulatifToplamNorm = (kumulatifToplam-m) / (n-m)

print(len(kumulatifToplamNorm))

plt.plot(kumulatifToplamNorm)
plt.show()

for i in range(0,len(kumulatifToplamNorm)):
    if kumulatifToplamNorm[i]>=0.90:
         esik=bins[i+1]
            
print("----", esik)
            
###########normalizasyon########################################################################################################

print("eşik değeri öncesi değer sayıları :" ,  (ArrayDicomf > esik).sum())        #1250 den büyük kaç değer var
ArrayDicomf[np.where(ArrayDicomf>esik)] = esik                    # görüntü daha büyük değerlere bölünüp kontrastı azalmaması için
print("eşik değerinden büyük değer sayısı : " , (ArrayDicomf > esik).sum())

l = np.min(ArrayDicomf)
h = np.max(ArrayDicomf)
norm = (ArrayDicomf -l) / (h-l)

print( "Min-Max Normalizasyon \n", norm)          #0-255 arasına çekilir

###########.mha okuma##########################################################################################################

img = [ img1, img2, img3, img4, img5]
        
mha = sitk.GetArrayFromImage(img[hasta])

mhaf=mha.astype(float)

for i in range(0, len(mhaf)):                                                                          
    plt.subplot(6,6,i+1),plt.imshow(mhaf[i]/4.0,'gray')    
    plt.xticks([]),plt.yticks([])
plt.show()


############dicom ve .mha çakıştırma(normalizasyon sonrası)###################################################################
patch_boyut = 64 

if hasta == 0:
    tumorVokselSayısı = 27541
    arkaplanVokselSayısı = 820331
elif hasta == 1 :
    tumorVokselSayısı = 32466
    arkaplanVokselSayısı = 852270
elif hasta == 2 :
    tumorVokselSayısı = 33284
    arkaplanVokselSayısı = 814588
elif hasta == 3 :
    tumorVokselSayısı = 22161
    arkaplanVokselSayısı = 788847
elif hasta == 4 :
    tumorVokselSayısı = 22867
    arkaplanVokselSayısı = 825005


tumor_patch_listesi = np.zeros((3000,patch_boyut,patch_boyut) , dtype=np.float)
arkaplan_patch_listesi = np.zeros((3000,patch_boyut,patch_boyut) , dtype=np.float)

tumor_voksel_sayac = 0
arkaplan_voksel_sayac = 0
for i in range(len(norm)): #(0,24)

    imgTemp = np.zeros((sad1, sad2, sad3 , 3 ), dtype=np.float) #(24,256,256)
    imgTemp[ i , : , : , 0 ] = norm[i] #görüntü oluşturuldu
    imgTemp[ i , : , : , 1 ] = norm[i]
    imgTemp[ i , : , : , 2 ] = norm[i]

    for j in range(0 , sad2):                                               #(0,256)
        for k in range(0 , sad3):                                           #(0,256)            
            if mhaf[i][j][k] == 1: 

                imgTemp[i , j , k , 0 ] = 1                                 #kırmızı kanal en yüksek
                imgTemp[i , j , k , 1 ] = 0                                 #yeşil kanal en düşük 
                imgTemp[i , j , k , 2 ] = 0                                 #mavi kanal en düşük     
                                
                            
                if ((patch_boyut/2)<=j<=sad2-(patch_boyut/2)) & ((patch_boyut/2)<=k<=sad3-(patch_boyut/2)):   
                    sayi1 = int(tumorVokselSayısı/1000)                   
                    tumor=random.randrange(0,sayi1)
                    if (tumor==10):
                        tumor_patch_listesi[tumor_voksel_sayac,:,:] = norm[i,j-int(patch_boyut/2):j+int(patch_boyut/2),k-int(patch_boyut/2):k+int(patch_boyut/2)]
                        tumor_voksel_sayac= tumor_voksel_sayac + 1   
                                    
            elif mhaf[i][j][k] == 0:                
                if ((patch_boyut/2)<=j<=sad2-(patch_boyut/2)) & ((patch_boyut/2)<=k<=sad3-(patch_boyut/2)):
                    sayi2= int(arkaplanVokselSayısı/1000)                 
                    arka=random.randrange(0,sayi2)
                    if (arka==100):
                        arkaplan_patch_listesi[arkaplan_voksel_sayac,:,:] = norm[i,j-int(patch_boyut/2):j+int(patch_boyut/2),k-int(patch_boyut/2):k+int(patch_boyut/2)]
                        arkaplan_voksel_sayac=arkaplan_voksel_sayac + 1    
                            
    plt.subplot(6,6,i+1),plt.imshow(imgTemp[i])   
    plt.xticks([]),plt.yticks([]) 
plt.show()  

tumor_patch_listesi =tumor_patch_listesi[:tumor_voksel_sayac , :, :]
arkaplan_patch_listesi =arkaplan_patch_listesi[:arkaplan_voksel_sayac , :, :]

"""
temp = tumor_patch_listesi[200,:,:]                    
temp = np.reshape(temp,(64,64))                          
plt.imshow(temp , 'gray')                                
plt.show()        
"""

#########5 MR için patch çıktıları#############################################################################################

print(arkaplan_voksel_sayac)
print(tumor_voksel_sayac)

#print(tumor_patch_listesi.shape)                       
#print(arkaplan_patch_listesi.shape)                 

print(len(tumor_patch_listesi))                       #array uzunlukları
print(len(arkaplan_patch_listesi))


#########dosyaya yazma########################################################################################################

if hasta == 0 :
    np.save('hasta1_tumor_64.npy' , tumor_patch_listesi , 'ab+')
    loadedTumor1 = np.load('hasta1_tumor_64.npy ')

    np.save('hasta1_arkaplan_64.npy' , arkaplan_patch_listesi , 'ab+')
    loadedArka1 = np.load('hasta1_arkaplan_64.npy ')


elif hasta == 1 :
    np.save('hasta2_tumor_64.npy' , tumor_patch_listesi , 'ab+')
    loadedTumor2 = np.load('hasta2_tumor_64.npy ')

    np.save('hasta2_arkaplan_64.npy' , arkaplan_patch_listesi , 'ab+')
    loadedArka2 = np.load('hasta2_arkaplan_64.npy ')


elif hasta == 2 :
    np.save('hasta3_tumor_64.npy' , tumor_patch_listesi , 'ab+')
    loadedTumor3 = np.load('hasta3_tumor_64.npy ')

    np.save('hasta3_arkaplan_64.npy' , arkaplan_patch_listesi , 'ab+')
    loadedArka3 = np.load('hasta3_arkaplan_64.npy ')


elif hasta == 3:
    np.save('hasta4_tumor_64.npy' , tumor_patch_listesi , 'ab+')
    loadedTumor4 = np.load('hasta4_tumor_64.npy ')

    np.save('hasta4_arkaplan_64.npy' , arkaplan_patch_listesi , 'ab+')
    loadedArka4 = np.load('hasta4_arkaplan_64.npy ')


elif hasta == 4:
    np.save('hasta5_tumor_64.npy' , tumor_patch_listesi , 'ab+')
    loadedTumor5 = np.load('hasta5_tumor_64.npy ')

    np.save('hasta5_arkaplan_64.npy' , arkaplan_patch_listesi , 'ab+')
    loadedArka5 = np.load('hasta5_arkaplan_64.npy ')
               

##########################################################################################################################
"""
np.save('temp.npy' , temp , 'ab+')               
loadedTemp=np.load('temp.npy')
"""
"""
temp2 = np.array(loadedTemp)                     #yüklenen görüntüler doğru çizilmiş mi                     
plt.imshow(temp2 , 'gray')                                 	
plt.show()
"""
###############################################################################################################################

from PictureHandler import PictureHandler
from PictureCreator import PictureCreator
import os,Properties,time
import sys
from keras.applications.resnet import ResNet50
from keras.layers import Dense
from keras.preprocessing import image

backgroundPath='Pictures/Work/Background'
chlorellaPath= 'Pictures/Work/Chlorella'
ph= PictureHandler()
pc=PictureCreator()
config=Properties.Properties()
backgroundsPath=os.listdir(backgroundPath)    
elementsPath = os.listdir(chlorellaPath)
elements= list()
backgrounds= list()
for i in range(len(backgroundsPath)):
    backgrounds.append(backgroundPath+'/'+backgroundsPath[i])

for i in range(len(elementsPath)):
    elements.append([chlorellaPath +'/'+elementsPath[i],'Chlorella'])


timeStart=(time.time())*1000
for i in range(config.TRAINING_LENGTH):
    trainPic,data=(pc.createPicture(backgrounds,elements))
    ph.save(trainPic,'Pictures/Created/'+str(i))
    sys.stdout.write("\rProgress: " +str(i)+" of "+str(config.TRAINING_LENGTH))
    sys.stdout.flush()

print ('\nMedium Training time: ' +str(((time.time()*1000)-timeStart)/config.TRAINING_LENGTH)+' ms')


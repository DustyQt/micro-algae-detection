from random import randint,choice
from PIL import Image,ImageDraw
import cv2 as cv
import os
import uuid
import numpy as np
from sklearn.model_selection import train_test_split
class PictureCreator:
    def __init__(self,properties):
        self.config= properties
        genDataDir=os.path.join(os.path.dirname(__file__), '../generatorData')
        self.classes=dict()
        for d in os.listdir(genDataDir):
        	fullpath=os.path.join(genDataDir,d)
        	self.classes[d]=[fullpath+'/'+name for name in os.listdir(fullpath) ]
        self.backgrounds=self.classes['background']
        self.classes.pop('background')   
        self.classNames=list(self.classes.keys())
        self.classes=list(self.classes.values())
        
    def createPicture(self):
        numberOfElements = randint(self.config['elements_quantity_range'][0],self.config['elements_quantity_range'][1])
        elements = list()
        bg=Image.open(self.backgrounds[randint(0,len(self.backgrounds)-1)])
        scale= self.config['image_size']/bg.size[0]
        for i in range(numberOfElements):
            eClass=randint(0,len(self.classes)-1)
            element = choice(self.classes[eClass])
            r = randint(0,360)
            image= Image.open(element).convert('RGBA').rotate(r)
            x = randint(0,bg.size[0]-image.size[0])
            y = randint(0,bg.size[1]-image.size[1])
            offset=(x,y)
            image= Image.open(element).convert('RGBA').rotate(r)
            boundary=[round(x *scale),round(y*scale),round(image.size[0]*scale),round(image.size[1]*scale)]
            elements.append([boundary, eClass])
            blackwhite=cv.cvtColor(cv.imread(element),cv.COLOR_BGR2GRAY)
            ret, thresh = cv.threshold(blackwhite,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
            mask=Image.frombytes('L',image.size, thresh ,'raw').rotate(r)
            bg.paste(image,offset,mask)
        bg.thumbnail((self.config['image_size'],self.config['image_size']), Image.LANCZOS)
        return bg, elements
        
class PictureHandler:
    def __init__(self):
        pass
    def open(self, filePath):
        image = Image.open(filePath)
        return image
    def show(self, image):
        image.show()

    def save(self, image: Image,name: str):
        image.save('generatedData/images/'+name+'.png')
    
    def drawBoundaries(self, img, info,classNames):
        image=img
        draw=ImageDraw.Draw(image)
        color = []
        for i in range(len(classNames)):
            color.append('#%06X' % randint(0, 0xFFFFFF))
        for i in range (len(info)):
            draw.rectangle([info[i][0][0],info[i][0][1],info[i][0][0]+info[i][0][2],info[i][0][1]+info[i][0][3]],outline=color[info[i][1]])
        del draw
        return image
    
class AnnotationHandler:
    def __init__(self,cats): 
        self.writeNames(cats)
    
    def generateBoundary(self,boundary,cat,size):
        x=boundary[0]/size[0]
        y=boundary[1]/size[1]
        w=boundary[2]/size[0]
        h=boundary[3]/size[1]
        return str(cat)+' '+str(x)+' '+str(y)+' '+str(w)+' '+str(h)
    
    def generateData(self,image,elements):
        annotations=[self.generateBoundary(e[0],e[1],image.size) for e in elements]
        filename=str(uuid.uuid4())
        self.writeLabels(filename,annotations)
        return filename
            
    def writeLabels(self,filename,annotations):
        file= open('generatedData/labels/'+filename+'.txt','w+')
        for element in annotations:
            file.write(element + "\n")
        file.close()

    def writeNames(self,content):
        file= open('generatedData/names.txt','w+')
        for element in content:
            file.write(element + "\n")
        file.close()
    
    def trainTestSplit(self,test_size,random_state):
        images = [os.path.join('generatedData/images', x) for x in os.listdir('generatedData/images')]
        train_images, test_images= train_test_split(images, test_size = test_size, random_state = random_state)
        file= open('generatedData/train.txt','w+')
        for element in train_images:
            file.write(element + "\n")
        file.close()
        file= open('generatedData/test.txt','w+')
        for element in test_images:
            file.write(element + "\n")
        file.close()

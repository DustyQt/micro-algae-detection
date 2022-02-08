from random import randint
from PIL import Image
import cv2 as cv
import Properties
class PictureCreator:
    def __init__(self):
        self.config= Properties.Properties()
        
    def createPicture(self, backgrounds: list, elements: list):
        numberOfElements = randint(self.config.ELEMENTS_QUANTITY_RANGE[0],self.config.ELEMENTS_QUANTITY_RANGE[1])
        elementsInfo = list()
        bg=Image.open(backgrounds[randint(0,len(backgrounds)-1)])
        scale= self.config.IMAGE_SIZE/bg.size[0]
        for i in range(numberOfElements):
            e = randint(0,len(elements)-1)
            element=elements[e]
            r = randint(0,360)
            image= Image.open(element[0]).convert('RGBA').rotate(r)
            x = randint(0,bg.size[0]-image.size[0])
            y = randint(0,bg.size[1]-image.size[1])
            offset=(x,y)
            image= Image.open(element[0]).convert('RGBA').rotate(r)
            boundary=[round(x *scale),round(y*scale),round((x+image.size[0])*scale),round((y+image.size[1])*scale)]
            elementsInfo.append([boundary, element[1]])
            blackwhite=cv.cvtColor(cv.imread(element[0]),cv.COLOR_BGR2GRAY)
            ret, thresh = cv.threshold(blackwhite,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
            mask=Image.frombytes('L',image.size, thresh ,'raw').rotate(r)
            bg.paste(image,offset,mask)
        bg.thumbnail((self.config.IMAGE_SIZE,self.config.IMAGE_SIZE), Image.LANCZOS)
        return bg, elementsInfo
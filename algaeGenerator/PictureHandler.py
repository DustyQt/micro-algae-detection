from PIL import Image, ImageDraw

class PictureHandler:
    def __init__(self):
        pass
    def open(self, filePath):
        image = Image.open(filePath)
        return image
    def show(self, image):
        image.show()

    def save(self, image: Image,name: str):
        image.save(str(name)+'.png')
    
    def drawBoundaries(self, img:list):
        image=img[0]
        draw=ImageDraw.Draw(image)
        for i in range (len(img[1])):
            draw.rectangle((img[1][i][0]),outline=(255,255,0))
        del draw
        return image
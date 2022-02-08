class Properties:
    def __init__(self):
        
        #Larger Side size for the image (the higher the size the heavier the image and processing time)
        self.IMAGE_SIZE = 250
        # anchor box scales(since the algaes and the picture size are small we use smaller than the normal 128, 256, 512)
        self.ANCHOR_BOX_SIZES = [32, 64, 128]
        # anchor box ratios(again we don't use the normal [1,2][2,1] because we are working with squareish boundingboxes)
        self.ANCHOR_BOX_RATIOS = [[1, 1], [1.3, 1], [1, 1.3]]
        # range of quantity of elements in generated images
        self.ELEMENTS_QUANTITY_RANGE = [0,30]
        # number of training examples per trainign session
        self.TRAINING_LENGTH= 30
        
        
            
       
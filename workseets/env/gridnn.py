from env.grid import *
import cv2
#============================================================================================
#======================Defining a Gridi: A Grid with Images States===========================
''' 
    Below we establish a class that will return after each step an observation which is the 
    image of the grid instead of its state id. This is essential to be able to deal with a 
    more general set of RL methods that are capable of learning by observing its environment 
    instead of being given the state's id. 

    Note that we save the images into a folder called img, so you will need to create such a 
    folder in the folder of this notebook. If you want to change this behaviour or save images 
    in the same folder of this notebook adjust the code accordingly.
'''

class iGrid(Grid):
    
    def __init__(self, animate=False, saveimg=False, resize=True, size=(50,84), **kw):
        super().__init__(**kw)
        self.i = 0                  # snapshot counter
        self.img = None             # snapshot image
        self.io = None              # snapshot io buffer
        self.animate = animate
        self.saveimg = saveimg
        self.resize = resize
        self.size = size

        
    # calling render__() directly is not a good idea for iGrid because s_() is calling it as well
    # calling render() allows us to turn animation/saveimg on/off
    def render(self, animate=None, saveimg=None, **kw):  
        if animate is not None: self.animate = animate 
        if saveimg is not None: self.saveimg = saveimg
        self.render_image()

    def render_image(self, saveimg):
       # prepare and scale the area that will be captured 
        if self.io is None: self.io = io.BytesIO()
        
        # scale = 0.0138888 # use this if you are using Jupyter notebooks
        scale = 0.01      # use this if you are using Jupyter Lab
        box = self.ax0.get_window_extent().transformed(mtransforms.Affine2D().scale(scale))
        
        # place frame in memory buffer then save to disk if you want
        plt.savefig(self.io, format='raw', bbox_inches=box)
        if saveimg or self.img is None:
            os.makedirs('img') if not os.path.exists('img') else None
            plt.savefig('img/img%d.png'%self.i, bbox_inches=box); self.i+=1 
            if self.img is None: 
                self.newshape = plt.imread('img/img0.png').shape
        #try:
        # reshape the image and store current image 
        self.img = np.reshape(np.frombuffer(self.io.getvalue(),dtype=np.uint8),newshape=self.newshape)[:,:,:3]
        #except:
            #self.img = np.frombuffer(self.io.getvalue(), dtype=np.uint8)[:,:,:3]
            #print('could not convert the image')
        if self.resize:
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            self.img = cv2.resize(self.img , dsize=(self.size[1],self.size[0]), interpolation=cv2.INTER_CUBIC)/255
            self.img = np.expand_dims(self.img, -1)# slightly better than  self.img = self.img[:,:,np.newaxis]
        # save only latest image to buffer and not accumulate
        self.io.seek(0) 

        
    def s_(self):
        self.render__(image=True, animate=self.animate, saveimg=self.saveimg)#, animate=self.animate)
        return self.img
    

def mazei(Grid=iGrid, r=6, c=9, **kw): # we cover this later
    return iGrid(gridsize=[r,c], s0=r//2*c, goals=[r*c-1], style='maze', **kw)#figsize is made ineffective

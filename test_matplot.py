import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

N = 1    # Num_of_img or Batch size
Dim = 1  # Dim_image
In = 10  # Input_size
Out = 10 # Output size
K = 3    # Filter_size
NF = 30  # Num_of_filter

def Convout_Visualization(grid_size,input):
    count=0
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            nparray=image[:,:,count]
            
            axes=plt.subplot2grid(grid_size, (i,j), rowspan=1, colspan=1)
            axes.get_xaxis().set_visible(False)
            axes.get_yaxis().set_visible(False)
            plt.imshow(nparray.eval(), cmap='gray')     # Do use .eval() to convert tf.tensor to python array
            #print(nparray)
            count+=1       
    plt.axis('off')
    plt.show()


# set input
img=mpimg.imread('sample.jpeg')
img=tf.convert_to_tensor(img,dtype=tf.float32)
img = tf.reshape(img, [N, In, In, Dim])

# set conv net
filter1=tf.ones([K, K, N, NF])          
filter2=tf.random_normal([K, K, N, NF])
conv = tf.nn.conv2d(img, filter2, strides=[1, 1, 1, 1],padding='SAME')

# set output
image=tf.reshape(conv,(Out, Out, NF))        

# set grid size you want to plot
grid_size=(3,10)
        
sess=tf.Session()
with sess.as_default():
    Convout_Visualization(grid_size, image)



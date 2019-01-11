import matplotlib . pyplot as plt
import numpy as np
import keras
from keras.datasets import cifar10
from keras . models import Sequential
from keras . layers import Dense , Dropout , Flatten
from keras . layers import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras import backend as K
def show_image_examples ( class_names , features , labels ) :
    print ( len ( class_names ) )
    num_classes = len ( class_names )
    fig = plt . figure ( figsize = (8 , 3 ) )
    for i in range ( num_classes ) :
        ax = fig . add_subplot ( 2 , 5 , 1 + i , xticks = [ ] , yticks = [ ] )
        idx = np. where ( labels [ : ] == i ) [ 0 ]
        features_idx = features [ idx , : : ]
        img_num = np. random. randint ( features_idx . shape [ 0 ] )
        im = features_idx [img_num, : : ]
        ax . set_title ( class_names [ i ] )
        plt .imshow (im )
    plt . show ( )
batch_size = 16
num_classes = 10
epochs = 5
# input image dimensions
img_rows , img_cols,img_planes = 32 , 32 , 3
# the data , s p l i t between t ra in and t e s t s e t s
( x_train , y_train ) , ( x_test , y_test ) = cifar10. load_data ( )
class_names = ['airplane','automobile' , 'bird' , 'cat','deer' ,'dog','frog','horse','ship','truck']
# Showing a few examples
show_image_examples ( class_names , x_train , y_train )
if K. image_data_format ( ) == 'channels_first':
    x_train = x_train . reshape ( x_train . shape [ 0 ] , 3 , img_rows , img_cols )
    x_test = x_test . reshape ( x_test . shape [ 0 ] , 3 , img_rows , img_cols )
    input_shape = ( 1 , img_rows , img_cols )
else :
    x_train = x_train . reshape ( x_train . shape [ 0 ] , img_rows , img_cols , 3 )
    x_test = x_test . reshape ( x_test . shape [ 0 ] , img_rows , img_cols , 3)
    input_shape = (img_rows , img_cols , 3 )
# Pick every 100 th sample to speed−up ( Se t t h i s to 1 in the f i n a l run . )
step = 100
x_train = x_train [ : :  , : , : ]
y_train = y_train [ : :  ]
x_test = x_test [ : : , : , : ]
y_test = y_test [ : :  ]
x_train = x_train . astype ('float32' )
x_test = x_test . astype ( 'float32' )
x_train /= 255
x_test /= 255
print ( 'x_train shape : ' , x_train.shape )
print ( x_train.shape[ 0 ] , 'train samples')
print (x_test. shape [ 0 ] , 'test samples')
# conver t c l a s s v e c t o r s to binary c l a s s ma tr ices
y_train = keras.utils . to_categorical ( y_train , num_classes )
y_test = keras . utils . to_categorical ( y_test , num_classes )
model = Sequential()
model.add(Conv2D(32, kernel_size=3,padding='same', activation='relu', input_shape=(32,32,3)))
model.add(Dropout(0.2))
model.add(Conv2D(32, kernel_size=3,padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=3,padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(64, kernel_size=3,padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=3,padding='same',activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(128, kernel_size=3,padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu',kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(num_classes , activation='softmax'))
model . compile ( loss =keras .losses . categorical_crossentropy ,optimizer=keras . optimizers .Adam( ) , metrics = [ 'accuracy'] )
model_info = model . fit ( x_train , y_train ,batch_size=batch_size ,epochs=epochs ,verbose =1 ,validation_data =( x_test , y_test ) )
model . summary ( )
score = model . evaluate ( x_test , y_test , verbose =0)
print ( 'Test loss :' , score [ 0 ] )
print ( ' Test accuracy :' , score [ 1 ]*100,'%')
def plot_model_history ( model_history ) :
    fig , axs = plt . subplots ( 1 , 2 , figsize = ( 15 , 5 ) )
# summarize h i s t o r y f o r accuracy
    axs [ 0 ] . plot ( range ( 1 , len ( model_history . history [ 'acc' ] ) + 1 ) ,
    model_history . history [ 'acc' ] )
    axs [ 0 ] . plot ( range ( 1 , len ( model_history . history [ 'val_acc' ] ) + 1 ) ,
    model_history . history [ 'val_acc' ] )
    axs [ 0 ] . set_title ( 'Model Accuracy' )
    axs [ 0 ] . set_ylabel ('Accuracy' )
    axs [ 0 ] . set_xlabel ('Epoch')
    axs [ 0 ] . set_xticks (np. arange ( 1 , len ( model_history . history ['acc' ] ) + 1 ) ,
     len ( model_history . history ['acc' ] ) / 10 )
    axs [ 0 ] . legend ( ['train' ,'val' ] , loc= 'best' )
# summarize h i s t o r y f o r l o s s
    axs [ 1 ] . plot ( range ( 1 , len ( model_history . history [ 'loss' ] ) + 1 ) ,
          model_history . history ['loss' ] )
    axs [ 1 ] . plot ( range ( 1 , len ( model_history . history ['val_loss' ] ) + 1 ) ,
        model_history . history ['val_loss' ] )
    axs [ 1 ] . set_title ('Model Loss' )
    axs [ 1 ] . set_ylabel ('Loss')
    axs [ 1 ] . set_xlabel ('Epoch')
    axs [ 1 ] . set_xticks (np. arange ( 1 , len ( model_history . history [ 'loss' ] ) + 1 ) ,
         len ( model_history . history ['loss' ] ) / 10 )
    axs [ 1 ] . legend ( ['train' , 'val'] , loc= 'best')
    plt . show ( )
plot_model_history ( model_info )

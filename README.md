# vanilla_cnn
a simple CNN for digit classification

# CNN structure:

 Layer (type)                Output Shape              Param #   
=================================================================
 input1 (InputLayer)         [(None, 28, 28, 1)]       0         
                                                                 
 conv1 (Conv2D)              (None, 28, 28, 6)         156       
                                                                 
 pool1 (MaxPooling2D)        (None, 14, 14, 6)         0         
                                                                 
 conv2 (Conv2D)              (None, 14, 14, 16)        2416      
                                                                 
 pool2 (MaxPooling2D)        (None, 7, 7, 16)          0         
                                                                 
 flatten (Flatten)           (None, 784)               0         
                                                                 
 dense1 (Dense)              (None, 400)               314000    
                                                                 
 dense2 (Dense)              (None, 120)               48120     
                                                                 
 dense3 (Dense)              (None, 84)                10164     
                                                                 
 dense4 (Dense)              (None, 10)                850       
                                                                 
 softmax (Softmax)           (None, 10)                0         
                                                                 
=================================================================
Total params: 375,706
Trainable params: 375,706
Non-trainable params: 0
_________________________________________________________________


# Accuracy
loss: 0.8028 - categorical_accuracy: 0.9964 - val_loss: 0.8137 - val_categorical_accuracy: 0.9898

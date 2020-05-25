#!/usr/bin/env python
# coding: utf-8

# In[64]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# In[70]:


def build_keras(self):
    model=keras.Sequential()
    model.add(layers.Conv2D(96, (3, 3), padding="same"))
    initalizer_layer=tf.keras.initializers.Constant(value=0.1)
    act = tf.keras.layers.PReLU( alpha_initializer=initalizer_layer)
    model.add(act)
    model.add(layers.Conv2D(76, (3, 3), padding="same"))
    model.add(act)
    model.add(layers.Conv2D(65, (3, 3), padding="same"))
    model.add(act)
    model.add(layers.Conv2D(55, (3, 3), padding="same"))
    model.add(act)          
    model.add(layers.Conv2D(47, (3, 3), padding="same"))
    model.add(act)          
    model.add(layers.Conv2D(39, (3, 3), padding="same"))
    model.add(act)          
    model.add(layers.Conv2D(32, (3, 3), padding="same"))
    model.add(act)          


    model1=keras.Sequential()
    model1.add(layers.Conv2D(96, (3, 3), padding="same"))
    model1.add(act)

    model2=keras.Sequential()
    model2.add(layers.Conv2D(76, (3, 3), padding="same"))
    model2.add(act)

    model3=keras.Sequential()
    model3.add(layers.Conv2D(65, (3, 3), padding="same"))
    model3.add(act)

    model4=keras.Sequential()
    model4.add(layers.Conv2D(55, (3, 3), padding="same"))
    model4.add(act)

    model5=keras.Sequential()
    model5.add(layers.Conv2D(47, (3, 3), padding="same"))
    model5.add(act)

    model6=keras.Sequential()
    model6.add(layers.Conv2D(39, (3, 3), padding="same"))
    model6.add(act)

    model7=keras.Sequential()
    model7.add(layers.Conv2D(32, (3, 3), padding="same"))
    model7.add(act)
    modelc=tf.keras.layers.Concatenate([model, model,model2,model3,model4,model5,model6,model7])

    modela1=keras.Sequential()
    modela1.add(modelc)
    modela1.add(layers.Conv2D(64, (1, 1), padding="same"))
    modela1.add(act)


    modelb=keras.Sequential()
    modela1.add(modelc)
    modelb.add(layers.Conv2D(32, (1, 1), padding="same"))
    modelb.add(act)
    modelb.add(layers.Conv2D(32, (3, 3), padding="same"))
    modelb.add(act)

    modelf=tf.keras.layers.Concatenate([modela1,modelb])

    result=keras.Sequential()
    result.add(modelf)
    result.add(layers.Conv2D(4, (1, 1), padding="same"))
    result.add(act)
    
    return result


# In[ ]:





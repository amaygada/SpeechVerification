import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPool2D, BatchNormalization,Activation
from tensorflow.keras.models import Model
from tensorflow import keras

from config import config

model_config = config["model"]
EMBED_DIM = model_config["embed_dim"]
FF_DIM = model_config["ff_dim"]

class Embed(Model):
    def __init__(self):
        # call the parent constructor
        super(Embed, self).__init__()
        
        self.conv1 = Conv2D(32, kernel_size=(7, 7), strides=(2,2))
        self.bn1 = BatchNormalization()
        self.act1 = Activation('relu')
        self.mp1 = MaxPool2D()
        
        self.conv2 = Conv2D(32, kernel_size=(3, 3))
        self.bn2 = BatchNormalization()
        self.act2 = Activation('relu')
        
        self.conv3 = Conv2D(32, kernel_size=(3, 3), padding='same')
        self.bn3 = BatchNormalization()
        self.act3 = Activation('relu')
        
        self.conv4 = Conv2D(32, kernel_size=(3, 3), padding='same')
        self.bn4 = BatchNormalization()
        self.act4 = Activation('relu')
        
        self.conv5 = Conv2D(32, kernel_size=(3, 3), padding='same')
        self.bn5 = BatchNormalization()
        self.act5 = Activation('relu')
        
        self.mp2 = MaxPool2D()
        
        self.conv6 = Conv2D(64, kernel_size=(3, 3), strides=(2,2))
        self.bn6 = BatchNormalization()
        self.act6 = Activation('relu')
        
        self.conv7 = Conv2D(64, kernel_size=(3, 3), padding='same')
        self.bn7 = BatchNormalization()
        self.act7 = Activation('relu')
        
        self.conv8 = Conv2D(64, kernel_size=(3, 3), padding='same')
        self.bn8 = BatchNormalization()
        self.act8 = Activation('relu')
        
        self.conv9 = Conv2D(64, kernel_size=(3, 3), padding='same')
        self.bn9 = BatchNormalization()
        self.act9 = Activation('relu')
        
        self.mp3 = MaxPool2D()
        
        self.conv10 = Conv2D(128, kernel_size=(3, 3), strides=(2,2))
        self.bn10 = BatchNormalization()
        self.act10 = Activation('relu')
        
        self.conv11 = Conv2D(128, kernel_size=(3, 3), padding='same')
        self.bn11 = BatchNormalization()
        self.act11 = Activation('relu')
        
        self.conv12 = Conv2D(128, kernel_size=(3, 3), padding='same')
        self.bn12 = BatchNormalization()
        self.act12 = Activation('relu')
        
        self.conv13 = Conv2D(128, kernel_size=(3, 3), padding='same')
        self.bn13 = BatchNormalization()
        self.act13 = Activation('relu')
        
        self.mp4 = MaxPool2D()
        
        self.flatten = Flatten()
        
        self.dense = Dense(256)
        self.act14 = Activation('relu')
        
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.act1(x)
        
        x = self.mp1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        y = self.conv3(x)
        y = self.bn3(y)
        y = self.act3(y)
        
        y = self.conv4(y)
        y = self.bn4(y)
        y = self.act4(y)
        
        y = self.conv5(y)
        y = self.bn5(y)
        y = self.act5(y)
        
        x = x+y
        
        x = self.mp2(x)
        
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.act6(x)
        
        y = self.conv7(x)
        y = self.bn7(y)
        y = self.act7(y)
        
        y = self.conv8(y)
        y = self.bn8(y)
        y = self.act8(y)
        
        y = self.conv9(y)
        y = self.bn9(y)
        y = self.act9(y)
        
        x = x+y

        x = self.mp3(x)
        
        x = self.conv10(x)
        x = self.bn10(x)
        x = self.act10(x)
        
        y = self.conv11(x)
        y = self.bn11(y)
        y = self.act11(y)
        
        y = self.conv12(y)
        y = self.bn12(y)
        y = self.act12(y)
        
        y = self.conv13(y)
        y = self.bn13(y)
        y = self.act13(y)
        
        x = y+x

        x = self.flatten(x)
        x = self.dense(x)
        x = self.act14(x)
        
        return x

    
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return ffn_output
    
class TransformerModel(Model):
    def __init__(self):
        super(TransformerModel, self).__init__()
        self.t1 = TransformerBlock(EMBED_DIM, 8, FF_DIM)
        self.t2 = TransformerBlock(EMBED_DIM, 8, FF_DIM)
        self.dense = Dense(1)
        
    def call(self, inputs):
        x = self.t1(inputs)
        x = self.t2(x)
        x = self.dense(x)
        return x

def cosine_batch(x, y):
    x = tf.einsum('ijk->ikj',x)
    dot = tf.squeeze(tf.matmul(x,y),axis=-1)
    dx = tf.math.sqrt(tf.reduce_sum(tf.math.square(x),axis=-1))
    dy = tf.math.sqrt(tf.reduce_sum(tf.math.square(y),axis=1))
    ans = dot/(dx*dy)
    return ans


loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction='none')
def loss_function(real,pred):
  loss_ = loss_object(real, pred)
  return loss_, tf.reduce_mean(loss_)


def accuracy(real,pred):
  pred = list(tf.squeeze(pred))
  real = list(tf.squeeze(real))
  for i in range(len(pred)):
    if pred[i] > 0.5:
      pred[i] = 1
    else:
      pred[i] = 0
  
  for i in range(len(pred)):
    if real[i] ==0:
      real[i] = 0
    else:
      real[i] = 1
  
  s = 0
  for i in range(len(pred)):
    if real[i] == pred[i]:
      s+=1
  
  return s
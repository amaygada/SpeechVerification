import tensorflow as tf
from tensorflow.keras.models import Model

import time

from model import loss_function, accuracy, cosine_batch
from dataset import extract_data

EPOCHS = 10
total_loss_train = []
total_loss_val = []
total_acc_train = []
total_acc_val = []
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)

def train(epochs, batch_size, train_data, dev_data, train_cap, dev_cap, model_path_embed, model_path_transformer, result_path, audio_train_path, audio_dev_path, embed, transformer):
  for e in range(epochs):
    print("epoch:", str(e))

    #start timer
    st = time.time()
    
    idx = 0
    same = 0
    epoch_loss = 0
    for t in range(train_cap):
      if t%5 == 0:
        print("batch t:",str(t),end=" ")

      with tf.GradientTape() as tape:
        x,y,l,idx = extract_data(train_data, idx, batch_size, audio_train_path)
        
        x_embed1 = embed(x[0])
        x_embed2 = embed(x[1])
        x_embed3 = embed(x[2])
        x_stack = tf.stack([x_embed1, x_embed2, x_embed3],axis=-1)
        x_embed = transformer(x_stack)
        y_embed = tf.expand_dims(embed(y), -1)
        cos = cosine_batch(x_embed,y_embed)

        ll, loss = loss_function(l,cos)
        same+=accuracy(l,ll)

      epoch_loss+=loss
      variables = embed.trainable_variables + transformer.trainable_variables
      gradients = tape.gradient(loss,variables)
      optimizer.apply_gradients(zip(gradients, variables))
    total_loss_train.append(epoch_loss/train_cap)
    total_acc_train.append(same/(batch_size*train_cap))

    print()
    idx = 0
    dev_loss = 0
    same_dev = 0
    for d in range(dev_cap):
      if d%50 == 0:
        print("batch d:",str(d),end=" ")

      x,y,l,idx = extract_data(dev_data, idx, batch_size, audio_dev_path)
      
      x_embed1 = embed(x[0])
      x_embed2 = embed(x[1])
      x_embed3 = embed(x[2])
      x_stack = tf.stack([x_embed1, x_embed2, x_embed3],axis=-1)
      x_embed = transformer(x_stack)
      y_embed = tf.expand_dims(embed(y), -1)
      cos = cosine_batch(x_embed,y_embed)
      # log = terminate(cos)

      ll, loss = loss_function(l,cos)
      same_dev+=accuracy(l,ll)
      dev_loss += loss
    total_loss_val.append(dev_loss/dev_cap)
    total_acc_val.append(same_dev/(batch_size*dev_cap))

    print()
    print("train loss:", str(epoch_loss/train_cap))
    print("validation loss:", str(dev_loss/dev_cap))
    print("train accuracy:", str(same/(batch_size*train_cap)))
    print("validation accuracy:", str(same_dev/(batch_size*dev_cap)))
    print("time:",time.time()-st)
    print()

    if e%5==0:
      embed.save(model_path_embed+str(e))
      transformer.save(model_path_transformer+str(e))

      file1 = open(result_path + str(e)+ ".txt","w")
      file1.write("train loss: " + str(epoch_loss/train_cap) + "\n")
      file1.write("validation loss: " + str(dev_loss/dev_cap) + "\n")
      file1.write("train accuracy: "+ str(same/(batch_size*train_cap)*100) + "%\n")
      file1.write("validation accuracy: " + str(same_dev/(batch_size*dev_cap)*100) + "%\n")
      file1.write("time: " + str(time.time()-st) + "\n")
      file1.close()

  return embed, transformer
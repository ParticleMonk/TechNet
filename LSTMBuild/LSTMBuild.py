import tensorflow as tf
from tensorflow import keras
from keras import layers
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import ast
#import statement for gensim doc2vec loading
from gensim.models.doc2vec import Doc2Vec, TaggedDocument



model = keras.Sequential()
model.add(layers.LSTM(649, input_shape=(100, 1)))
model.add(layers.BatchNormalization())
model.add(layers.Dense(649))
print(model.summary())

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer="sgd",
    metrics=["accuracy"],
)


#load in the data with converted abstracts to nump array of floats and converted group_ids to floats
df = pd.read_table(r'LabeledToken1819PatentData.tsv', encoding="utf-8", converters={'abstract': str, 'labels': float})



#convert abstract column to list
abstract_list = df['abstract'].tolist()

#convert list of strings to list of lists
abstract_list = [ast.literal_eval(abstract) for abstract in abstract_list]

print("abstract_list length: " + str(len(abstract_list)))
print("abstract_list type: " + str(type(abstract_list)))
print("abstract_list sample: " + str(abstract_list[0]))

#load in the doc2vec model
modelDoc2Vec = Doc2Vec.load(r"doc2vec1819.model")

print("Creating abstract vectors...")
#convert the abstracts to vectors
abstracts = [modelDoc2Vec.infer_vector(abstract) for abstract in abstract_list]

print("AbstractVectors list created")

#create an numpy array of the labels
labels = df['labels'].to_numpy()

#convert list of abstract vectors to numpy array
abstracts = np.array(abstracts)

#convert labels to floats
labels = labels.astype(float)

x_train, x_test, y_train, y_test = train_test_split(abstracts, labels, test_size=0.2, random_state=42)


#reshape the data
x_train = x_train.reshape(-1, 100)
x_test = x_test.reshape(-1, 100)



#print sizes of train and test data
print("x_train shape: " + str(x_train.shape))
print("y_train shape: " + str(y_train.shape))
print("x_test shape: " + str(x_test.shape))
print("y_test shape: " + str(y_test.shape))


x_validate, y_validate = x_test[:-10], y_test[:-10]
model.fit(
    x_train, y_train, validation_data=(x_test, y_test), batch_size=64, epochs=1
)
model.fit(
    x_train, y_train, validation_data=(x_validate, y_validate), batch_size=64, epochs=10
)

for i in range(10):
    result = tf.argmax(model.predict(tf.expand_dims(x_test[i], 0)), axis=1)
    print(result.numpy(), y_test[i])

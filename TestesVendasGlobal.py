
import pandas as pd
from keras.layers import Dense, Dropout, Activation, Input
from keras.models import Model
from tensorflow.keras.optimizers import Adam


base = pd.read_csv("games.csv")


base=base.drop("Other_Sales", axis =1)
base=base.drop("NA_Sales", axis =1)
base=base.drop("EU_Sales", axis =1)
base=base.drop("JP_Sales", axis =1)
base=base.drop("Developer", axis =1)

base = base.dropna(axis=0)




name_games = base.Name
base = base.drop("Name", axis=1)

base =pd.DataFrame(base)
base =base.sample(frac=1)


inputs = base.iloc[:,[0,1,2,3,5,6,7,8,9]].values
outputs= base.iloc[:,[4]].values



from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_inputs = LabelEncoder()


inputs[:,0] = labelencoder_inputs.fit_transform(inputs[:,0])
inputs[:,2] = labelencoder_inputs.fit_transform(inputs[:,2])
inputs[:,3] = labelencoder_inputs.fit_transform(inputs[:,3])
inputs[:,5] = labelencoder_inputs.fit_transform(inputs[:,8])


onehotencoder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [0,2,3,8])],remainder='passthrough')

adam = Adam(lr=0.01)

inputs = onehotencoder.fit_transform(inputs).toarray()


from sklearn.model_selection import train_test_split


inputs_train, inputs_test,outputs_train, outputs_test = train_test_split(inputs,outputs,test_size=0.3)


input_layer = Input(shape=(303,))

hiden_layer01 = Dense(units=150, activation='selu')(input_layer)
drop_layer01 = Dropout(0.3)(hiden_layer01)
hiden_layer02 = Dense(units=150, activation='selu')(drop_layer01)
drop_layer02 = Dropout(0.3)(hiden_layer02)
hiden_layer03 = Dense(units=150, activation='selu')(drop_layer02)
drop_layer03 = Dropout(0.3)(hiden_layer03)
hiden_layer04 = Dense(units=150, activation='selu')(drop_layer03)
drop_layer04 = Dropout(0.3)(hiden_layer04)

out_layer = Dense(units=1, activation="linear")(drop_layer04)



regressor = Model(inputs=input_layer, outputs=out_layer)

regressor.compile(optimizer="adamax", loss="mse", metrics=["mean_squared_error"])

regressor.fit(inputs_train,outputs_train, epochs=5000, batch_size=500)


predicts = regressor.predict(inputs_test);

import numpy as np
outputs_test2 = [np.argmax(t) for t in outputs_test]
predicts2 = [np.argmax(t) for t in predicts]

from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(predicts2,outputs_test2)

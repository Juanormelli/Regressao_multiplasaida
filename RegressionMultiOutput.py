
import pandas as pd
from keras.layers import Dense, Dropout, Activation, Input
from keras.models import Model


base = pd.read_csv("games.csv")


base=base.drop("Other_Sales", axis =1)
base=base.drop("Global_Sales", axis =1)
base=base.drop("Developer", axis =1)


base = base.dropna(axis=0)

base =base.loc[base["NA_Sales"]>1]

base =base.loc[base["EU_Sales"]>1]

name_games = base.Name
base = base.drop("Name", axis=1)

inputs = base.iloc[:,[0,1,2,3,7,8,9,10,11]].values
output_na = base.iloc[:,[4]].values
output_eu = base.iloc[:,[5]].values
output_jp = base.iloc[:,[6]].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_inputs = LabelEncoder()


inputs[:,0] = labelencoder_inputs.fit_transform(inputs[:,0])
inputs[:,2] = labelencoder_inputs.fit_transform(inputs[:,2])
inputs[:,3] = labelencoder_inputs.fit_transform(inputs[:,3])
inputs[:,8] = labelencoder_inputs.fit_transform(inputs[:,8])


onehotencoder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [0,2,3,8])],remainder='passthrough')

inputs = onehotencoder.fit_transform(inputs).toarray()

input_layer = Input(shape=(61,))

hiden_layer01 = Dense(units=100, activation='sigmoid')(input_layer)
hiden_layer02 = Dense(units=100, activation='sigmoid')(hiden_layer01)
out_layer_na = Dense(units=1, activation="linear")(hiden_layer02)
out_layer_eu = Dense(units=1, activation="linear")(hiden_layer02)
out_layer_jp = Dense(units=1, activation="linear")(hiden_layer02)

regressor = Model(inputs=input_layer, outputs=[out_layer_na,out_layer_eu,out_layer_jp])

regressor.compile(optimizer="adam", loss="mse")

regressor.fit(inputs, [output_na,output_eu,output_jp], epochs=5000, batch_size=100)


predicts = regressor.predict(inputs);

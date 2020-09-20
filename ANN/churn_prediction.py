# IMPORT PACKAGE
import pandas as pd
from tensorflow.keras.layers import Dense, Flatten
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# IMPORT DATASET
dataset = pd.read_excel("./Datasets/churndata.xlsx")

# SPLIT DATA TESTING AND TRAINING
x=dataset.iloc[:,3:13].values
y=dataset.iloc[:,-1].values

# ENCODER LABEL & ONE HOT
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
x[:,1] = le.fit_transform(x[:,1])
x[:,2] = le.fit_transform(x[:,2])
one = ColumnTransformer([("Geography", OneHotEncoder(), [1])], remainder = 'passthrough')
x = one.fit_transform(x)

# DATA SPLIT FOR MODELLING
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size= 0.4, random_state= 1234)

# SCALING DATA
sc = StandardScaler()
X_train = sc.fit_transform(X_train)

# MODELLING AND TRAINING
model = Sequential()
model.add(Dense(500, input_dim=12, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# COMPILING THE ANN
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
model.fit(X_train, y_train, batch_size = 1000, epochs = 100,validation_data=(X_test,y_test))

y_pred = model.predict(X_test)
y_pred_1 = (y_pred > 0.5)

# Creating the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_1)

# Con
print(cm)
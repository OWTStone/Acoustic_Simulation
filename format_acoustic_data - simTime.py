from os.path import dirname, split
import glob
import struct
import numpy as np
import math
import keras
from sklearn.model_selection import train_test_split
from scipy import signal
from scipy.io import wavfile
from scipy.fftpack import fft
import matplotlib.pyplot as plt

def read_float32_from_binary(file_path):
    """Reads a 32-bit float from a binary file.

    Args:
        file_path: The path to the binary file.

    Returns:
        A float32 value, or None if an error occurs.
    """
    try:
        with open(file_path, 'rb') as file:
            binary_data = file.read(2)  # Read 2 bytes for short drone present flag
            binary_data = file.read(4)  # Read 4 bytes for float64
            #binary_data = file.read()
            #if len(binary_data) != 4:
            #  return None # Handle cases where file doesn't have enough bytes
            float32_value = struct.unpack('f', binary_data)[0]
            return float32_value
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return None
    except struct.error:
      print(f"Error: Could not unpack data as float64")
      return None

data_dir = 'C:\\Users\\ShaneStone\\source\\acoustic\\simdata\\'
dirPath = data_dir+'\\test_binary*.dat'
res = glob.glob(dirPath)

trainIdx = [0,1,2,83,84,85,3,4,5,10,11,86,87,88,12,13,14,15,89,90,91,20,21,22,92,23,24,93,25,30,94,31,32,95,33,34,96,35,97,49,41,98,42,99,43,44,100,45,50,101,51,52,102,53,54,103,55,60,61,104,62,63,105,64,65,106,70,71,107,72,73,108,74,75,109,80,110,81,111,82,112,113,114,115]
testIdx = [6,7,8,9,16,17,18,19,26,27,28,29,36,37,38,39,46,47,48,49,56,57,58,59,66,67,68,69,76,77,78,79,86,87,88,89,96,97,98,99,106,107,108,109,116,117,118,119]
file_count = 0;
sample_length = 50000
sample_rate = 50000
epochs = 500
batch_size = 32
xData = np.zeros((sample_length,160))
yData = np.zeros(120,dtype='i')
saved_model_name = data_dir + 'best_model_batch32_sigmoid_time.keras'

for filename in res:
  drone_flag = []
  print(filename)
  path_file = split(filename)   

#try:
  with open(filename, 'rb') as file:
      drone_flag = file.read(2)  # Read 2 bytes for short drone present flag
      binary_data = file.read()  # Read 4 bytes for float64
      #drone_flag = struct.unpack('h',binary_data[0:1])
      float_data = struct.unpack('f'*((len(binary_data))//4),binary_data)
            #binary_data = file.read()
            #if len(binary_data) != 4:
            #  return None # Handle cases where file doesn't have enough bytes
            #float32_value = struct.unpack('f', binary_data)[0]
#    except FileNotFoundError:
#      print(f"Error: File not found: {file_path}")
#    except struct.error:
#      print(f"Error: Could not unpack data as float64")

# Example usage:
#file_path = 'test_binary.dat' # Replace with your file path

  print("Float array size is",np.shape(float_data))
#print(binary_data)
#print(float_data)
  drone_flagg = int.from_bytes(drone_flag)  

  Ntemp = np.shape(float_data)

  print(Ntemp)
  N = sample_length #Ntemp[0]
  print("Sampling rate =",sample_rate, "Hz")
  print("Sample length =", (N-1)/sample_rate,"seconds")
  
  T = 1.0/sample_rate
  #yf = fft(samples[:,1],n=N)
  #sample_length = N//2
  #print(type(yf))
  #print(np.shape(yf))
  print("File count =",file_count)
  
  if file_count == 0:
    xData[:,0] = float_data[0:sample_length]
    xData[:,0] = xData[:,0]/max(xData[:,0]) # try normailzing data
    print(np.shape(xData))
    print(xData)
  else:
    xData[:,file_count] = float_data[0:sample_length]
    xData[:,file_count] = xData[:,file_count]/max(xData[:,file_count]) # try normailzing datav
    #xData[:][file_count] = np.column_stack(xData, yf[:N//2],1)
  audio_type = path_file[1]
  drone_flagg = int.from_bytes(drone_flag)
  if file_count < 120: #drone_flagg:
    yData = np.append(yData,1)
    print("Drone present",drone_flagg)
  else:
    yData = np.append(yData,0)
    print("No drone",drone_flagg)
    
  file_count = file_count + 1

print("XData size is",np.shape(xData))
print("YData size is",np.shape(yData))
#print(xData)
print(yData)
print(file_count)

fig, ax = plt.subplots()
#ax.plot(xf, 2.0/N * np.abs(yf[:N//2]))
ax.plot(xData[:,159])
plt.grid()
plt.show()

#X_train, X_test, y_train, y_test = train_test_split(np.transpose(xData), yData, test_size=0.33, random_state=42)
X_train = np.transpose(xData[:,trainIdx])
y_train = yData[trainIdx]
X_test = np.transpose(xData[:,testIdx])
y_test = yData[testIdx]

print(np.shape(X_train),np.shape(X_test),np.shape(y_train),np.shape(y_test))

def make_model(input_shape):
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)

    gap = keras.layers.GlobalAveragePooling1D()(conv3)

    output_layer = keras.layers.Dense(num_classes, activation="sigmoid")(gap)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)

x_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
x_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
classes = np.unique(yData, axis=0)
num_classes = len(np.unique(y_train))

model = make_model(input_shape=x_train.shape[1:])
#keras.utils.plot_model(model, show_shapes=True)


callbacks = [
    keras.callbacks.ModelCheckpoint(
        saved_model_name, save_best_only=True, monitor="val_loss"
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
    ),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
]
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["sparse_categorical_accuracy"],
)
history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_split=0.2,
    verbose=1,
)

model = keras.models.load_model(saved_model_name)

test_loss, test_acc = model.evaluate(x_test, y_test)

print("Test accuracy", test_acc)
print("Test loss", test_loss)

metric = "sparse_categorical_accuracy"
plt.figure()
plt.plot(history.history[metric])
plt.plot(history.history["val_" + metric])
plt.title("model " + metric)
plt.ylabel(metric, fontsize="large")
plt.xlabel("epoch", fontsize="large")
plt.legend(["train", "val"], loc="best")
plt.show()
plt.close()

from sklearn.metrics import confusion_matrix

#Predict
y_prediction = model.predict(x_test)

#Create confusion matrix and normalizes it over predicted (columns)
print(y_prediction, y_test)
#print(np.shape(y_prediction), np.shape(y_test))
result = confusion_matrix(y_test, np.rint(y_prediction[:,1]))
print(result)
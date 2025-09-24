from os.path import dirname, split
import glob
import numpy as np
import wave
import math
import keras
from sklearn.model_selection import train_test_split
from scipy import signal
from scipy.io import wavfile
from scipy.fftpack import fft
import matplotlib.pyplot as plt

def check_wav_channels(file_path):
    """
    Checks if a WAVE file is mono or stereo.

    Args:
        file_path (str): The path to the WAVE file.

    Returns:
        str: "mono" if the file is mono, "stereo" if it's stereo, or None if an error occurs.
    """
    try:
        with wave.open(file_path, 'rb') as wf:
            num_channels = wf.getnchannels()
            if num_channels == 1:
                return "mono"
            elif num_channels == 2:
                return "stereo"
            else:
                return None  # Unknown number of channels
    except wave.Error as e:
         print(f"Error opening or reading WAVE file: {e}")
         return None
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
	
data_dir = 'C:\\Users\\ShaneStone\\source\\acoustic\\Drone-detection-dataset-master\\Data\\Audio\\'
dirPath = data_dir+'\\*.wav'
res = glob.glob(dirPath)

file_count = 0;
sample_length = 441000
epochs = 500
batch_size = 16
#sample_length = 2**18
xData = np.zeros((sample_length,90))
yData = np.zeros(90,dtype='i')
saved_model_name = 'best_model_441000_batch8_sigmoid_time_gold.keras'

for file in res:
  print(file)
  path_file = split(file)
  channel_type = check_wav_channels(file)
  
  if channel_type:
    print(f"The WAVE file is {channel_type}.")
    if channel_type == "mono":
      data_channel = 0
    else:
      data_channel = 0
  else:
    print("Could not determine the channel type of the WAVE file.")
    data_channel = 0
    
  sample_rate, samples = wavfile.read(file)
  
  frequencies, times, spectrogram = signal.spectrogram(samples[:,data_channel], sample_rate)

  Ntemp = samples.shape

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
    xData[:,0] = samples[0:sample_length,data_channel]
    xData[:,0] = xData[:,0]/max(xData[:,0]) # try normailzing datav
    print(np.shape(xData))
    print(xData)
  else:
    xData[:,file_count] = samples[0:sample_length,data_channel]
    xData[:,file_count] = xData[:,file_count]/max(xData[:,file_count]) # try normailzing datav
  audio_type = path_file[1]

  if audio_type[:5] == "DRONE":
    yData[file_count] = 1
  else:
    yData[file_count] = 0
    
  file_count = file_count + 1

print("XData size is",np.shape(xData))
print("YData size is",np.shape(yData))
print(xData)
print(yData)
print(file_count)

#fig, ax = plt.subplots()
#ax.plot(xf, 2.0/N * np.abs(yf[:N//2]))
#ax.plot(xData)
#plt.grid()
#plt.show()

X_train, X_test, y_train, y_test = train_test_split(np.transpose(xData), yData, test_size=0.33, random_state=42)

print(np.shape(X_train),np.shape(X_test),np.shape(y_train),np.shape(y_test))

def make_model(input_shape):
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=128, kernel_size=3, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    conv3 = keras.layers.Conv1D(filters=32, kernel_size=3, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)

    conv4 = keras.layers.Conv1D(filters=16, kernel_size=3, padding="same")(conv3)
    conv4 = keras.layers.BatchNormalization()(conv4)
    conv4 = keras.layers.ReLU()(conv4)

    gap = keras.layers.GlobalAveragePooling1D()(conv4)

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
#history = model.fit(
#    x_train,
#    y_train,
#    batch_size=batch_size,
#    epochs=epochs,
#    callbacks=callbacks,
#    validation_split=0.2,
#    verbose=1,
#)

model = keras.models.load_model(saved_model_name)

test_loss, test_acc = model.evaluate(x_test, y_test)

print("Test accuracy", test_acc)
print("Test loss", test_loss)

#metric = "sparse_categorical_accuracy"
#plt.figure()
#plt.plot(history.history[metric])
#plt.plot(history.history["val_" + metric])
#plt.title("model " + metric)
#plt.ylabel(metric, fontsize="large")
#plt.xlabel("epoch", fontsize="large")
#plt.legend(["train", "val"], loc="best")
#plt.show()
#plt.close()

from sklearn.metrics import confusion_matrix
#from sklearn.metrics import plot_confusion_matrix


#class estimator:
#  _estimator_type = ''
#  classes_=[]
#  def __init__(self, model, classes):
#    self.model = model
#    self._estimator_type = 'classifier'
#    self.classes_ = classes
#  def predict(self, X):
#    y_prob= self.model.predict(X)
#    y_pred = y_prob.argmax(axis=1)
#    return y_pred

class_names = ['No Drone','Drone Present']
#classifier = estimator(model, class_names)
#plot_confusion_matrix(estimator=classifier, X=x_test, y_true=y_test)
#figsize = (12,12)
#plot_confusion_matrix(estimator=classifier, X=tx_test, y_true=y_test, cmap='Blues', normalize='true', ax=plt.subplots(figsize=figsize)[1])
#Predict
y_prediction = model.predict(x_test)

from sklearn.metrics import ConfusionMatrixDisplay

#Create confusion matrix and normalizes it over predicted (columns)
print(y_prediction, y_test)
#print(np.shape(y_prediction), np.shape(y_test))
result = confusion_matrix(y_test, np.rint(y_prediction[:,1]))
print(result)

disp = ConfusionMatrixDisplay.from_predictions(y_test,np.rint(y_prediction[:,1]),colorbar=False,display_labels=class_names)
#plt.xlabel = "Truth - Drone Present"
#plt.ylabel = "Predicted - Drone Present"
#disp.title('Predictor COnfusion Matrix')
plt.show()
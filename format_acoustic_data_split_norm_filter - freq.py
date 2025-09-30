from os.path import dirname, split
import glob
import numpy as np
import wave
import ipdb
import math
import keras
from sklearn.model_selection import train_test_split
from scipy import signal
from scipy.io import wavfile
from scipy.fftpack import fft, fftfreq
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt

scaler = MinMaxScaler()

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
	
data_dir = 'C:\\Users\\stone\\acoustic\\Data\\Audio\\'
dirPath = data_dir+'\\*.wav'
res = glob.glob(dirPath)

file_count = 0;
sample_length = 441000
epochs = 500
batch_size = 64
#sample_length = 2**18
xData = np.zeros((sample_length,1))
yData = []
saved_model_name = 'best_model_441000_epochs500_batch64_64_16_4_sigmoid_freq_presplit.keras'

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
  startCount = 0
  dstasetCount = 0
  Nsamples = samples.shape
  print("Nsamples = ",Nsamples)
  print(type(Nsamples))
  if Nsamples[0] > sample_rate: # one seconds worth of data_channel
      Ntemp = sample_rate
  else:
      Ntemp = Nsamples[0]
 
  while (startCount + Ntemp) < Nsamples[0]: # verify sufficient number of samples exist
    print("Start count =",startCount)
    audio_type = path_file[1]
    N = Ntemp

    #print("Sampling rate =",sample_rate, "Hz")
    #print("Sample length =", (N-1)/sample_rate,"seconds")
    N = 2**(math.ceil(math.log2(N)))
    #print(N,"with zero padding")

    T = 1.0/sample_rate
    yf = fft(samples[startCount:startCount+Ntemp,1],n=N)
    xf = fftfreq(N, d=1/sample_rate)
    y_fit = yf.reshape(-1,1)
    y_norm = scaler.fit_transform(np.abs(y_fit[0:len(y_fit)//2]))

    # 5. Plot the results (optional, but good for visualization)
    plt.figure(figsize=(10, 6))
    plt.semilogx(xf[0:len(xf)//2], y_norm)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('Frequency Spectrum - ' + audio_type[:-4])
    plt.grid(True)
    plt.savefig(file[:-4] + "_" + str(startCount) + "_norm.png",format='png')
    plt.close()
    #plt.show()
    #print(type(yf))
    #print(np.shape(yf))
    print("File count =",file_count)
  
    if file_count == 0 and startCount == 0:
      #xData[:,0] = np.abs(yf[:Ntemp])
      xData = y_norm
      print(np.shape(xData))
      print(xData)
    else:
      #xData[:,file_count] = np.abs(yf[:sample_length])
      xData = np.column_stack((xData,y_norm))
      print("Shape of xData is",np.shape(xData))
      #xData[:][file_count] = np.column_stack(xData, yf[:N//2],1)

    if audio_type[:5] == "DRONE":
      yData = np.append(yData,1)
    else:
      yData = np.append(yData,0)
    
    startCount = startCount + Ntemp
  
  if (Nsamples[0] - (startCount + Ntemp)) > Nsamples[0]/2:  
    print("Start count =",startCount)
    N = Nsamples[0] - (startCount + Ntemp)

    #print("Sampling rate =",sample_rate, "Hz")
    #print("Sample length =", (N-1)/sample_rate,"seconds")
    N = 2**(math.ceil(math.log2(N)))
    #print(N,"with zero padding")

    T = 1.0/sample_rate
    yf = fft(samples[startCount:Nsamples,1],n=N)
    #print(type(yf))
    #print(np.shape(yf))
    print("File count =",file_count)
  
    #xData[:,file_count] = np.abs(yf[:sample_length])
    xData = np.column_stack((xData,np.abs(yf[:N//2])))
    print("Shape of xData is",np.shape(xData))
    #xData[:][file_count] = np.column_stack(xData, yf[:N//2],1)
    audio_type = path_file[1]

    if audio_type[:5] == "DRONE":
      yData = np.append(yData,1)
    else:
      yData = np.append(yData,0)
    
    startCount = startCount + Ntemp
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


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(np.transpose(xData), yData, test_size=0.33, shuffle=True, random_state=42)


# In[8]:


print(np.shape(X_train),np.shape(X_test),np.shape(y_train),np.shape(y_test))

def make_model(input_shape):
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=128, kernel_size=3, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv1D(filters=32, kernel_size=3, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    conv3 = keras.layers.Conv1D(filters=8, kernel_size=3, padding="same")(conv2)
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
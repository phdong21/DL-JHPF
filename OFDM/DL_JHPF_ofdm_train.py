from keras.layers import Input, Dense, Dropout, Convolution2D, MaxPool2D, normalization, Lambda, regularizers
from keras.models import Model, Sequential, load_model
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import ModelCheckpoint
from numpy import *
import numpy as np
import numpy.linalg as LA
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True   #allow growth
import scipy.io as sio
import math
from keras import backend as K

cha_path=3
Ns=3
Nt=32
Nt_RF=Ns
Nr=16
Nr_RF=Ns
SNR=10.0**(10/10.0) # transmit power
P=SNR

scale=1
fre=2

channel_norm_factor=np.sqrt(6)

# QPSK modulation simple
def QPSK_modu_sim(x):
    x_mod=np.zeros((Ns,1),dtype=complex)
    for n in range(Ns):
        a1=2*x[n]-1
        x_mod[n]=1.0/np.sqrt(2)*(a1[0]+1j*a1[1])
    return x_mod

# signal flow
def process(x,A):
    y=K.batch_dot(x,A)
    return y

def exp_pha(x):
    y_real=tf.cos(x)
    y_imag = tf.sin(x)
    y=tf.cast(tf.complex(y_real, y_imag), tf.complex64)
    return y

# obtain equivalent channel
def eq_channel_fun(temp):
    H,  fRF, wRF = temp
    fRF2 = exp_pha(fRF) / np.sqrt(Nt)
    FRF = tf.reshape(fRF2, (-1, Nt, Nt_RF))
    wRF2 = exp_pha(wRF) / np.sqrt(Nr)
    WRF = tf.reshape(wRF2, (-1, Nr, Nr_RF))
    WRFH = tf.conj(tf.transpose(WRF, [0, 2, 1]))
    Heq_left = K.batch_dot(WRFH, H)
    Heq = K.batch_dot(Heq_left, FRF)
    Heq_vec=tf.reshape(Heq,(-1,Nr_RF*Nt_RF))
    y_real = tf.real(Heq_vec)
    y_imag = tf.imag(Heq_vec)
    y = tf.concat([y_real, y_imag], 1)
    return y

# signal flow function
def sig_flow(temp):
    x_sym, H, awgn, fRF, wRF, fBB_real, fBB_imag, wBB_real, wBB_imag = temp
    fBB1 = tf.cast(tf.complex(fBB_real, fBB_imag), dtype=tf.complex64)
    fBB2 = tf.reshape(fBB1, (-1, 1, Nt_RF * Ns))
    FBB = tf.reshape(fBB2, (-1, Nt_RF, Ns))
    fRF2 = exp_pha(fRF) / np.sqrt(Nt)
    FRF = tf.reshape(fRF2, (-1, Nt, Nt_RF))
    FRF_BB = K.batch_dot(FRF, FBB)
    norm_factor = np.sqrt(Ns) * tf.divide(1.0, tf.norm(FRF_BB, axis=[-2, -1], keepdims=True))
    fBB3 = K.batch_dot(norm_factor, fBB2)
    FBB = tf.reshape(fBB3, (-1, Nt_RF, Ns))
    wBB2 = tf.cast(tf.complex(wBB_real, wBB_imag), dtype=tf.complex64)
    WBB = tf.reshape(wBB2, (-1, Nr_RF, Ns))
    wRF2 = exp_pha(wRF) / np.sqrt(Nr)
    WRF = tf.reshape(wRF2, (-1, Nr, Nr_RF))
    process_FBB=K.batch_dot(FBB,x_sym)
    process_FRF = K.batch_dot(FRF, process_FBB)
    process_channel = K.batch_dot(H, process_FRF)+1.0/np.sqrt(P/Ns)*awgn
    WRFH=tf.conj(tf.transpose(WRF, [0,2,1]))
    process_WRF=K.batch_dot(WRFH, process_channel)
    WBBH = tf.conj(tf.transpose(WBB, [0, 2, 1]))
    process_WBB = K.batch_dot(WBBH, process_WRF)
    y_real=tf.real(tf.transpose(process_WBB, [0, 2, 1]))
    y_imag = tf.imag(tf.transpose(process_WBB, [0, 2, 1]))
    y=tf.concat([y_real,y_imag],2)
    y=tf.reshape(y,(-1,2*Ns))
    return y


#### Training data ####
data_num_train=1000
data_num_file=1000
# Input0: input bits
trans_bit=np.random.randint(0,2,(data_num_train,Ns,2), dtype=int)
trans_bit_label=np.reshape(trans_bit,(data_num_train,2*Ns))
mod_symb=np.zeros((data_num_train,Ns,1), dtype=complex)
for n in range(data_num_train):
    mod_symb[n]=QPSK_modu_sim(trans_bit[n])
mod_symb_T=np.reshape(mod_symb,(data_num_train,Ns))
mod_symb_label=np.hstack((np.real(mod_symb_T),np.imag(mod_symb_T)))
print(trans_bit[0],'\n',trans_bit_label[0],'\n',mod_symb[0],'\n',mod_symb_label[0])

# Input1&2: perfect CSI
H_train1=zeros((data_num_train,2*Nr*Nt), dtype=float)
sub_RF=0
filedir = os.listdir('D:\\2fre_data')   # example data
n=0
for filename in filedir:
    newname = os.path.join('D:\\2fre_data', filename)
    data = sio.loadmat(newname)
    channel = data['ChannelData_fre']
    for i in range(data_num_file):
        a=channel_norm_factor*channel[:,:,sub_RF,i]
        h=np.reshape(a,(1,Nt*Nr))
        h_re=np.real(h)
        h_im = np.imag(h)
        h_re_im=np.hstack((h_re,h_im))
        H_train1[n*data_num_file+i]=h_re_im
    n=n+1
print(n)
print(H_train1.shape)

H_train2=zeros((data_num_train,Nr,Nt), dtype=complex)
filedir = os.listdir('D:\\2fre_data')
n=0
for filename in filedir:
    newname = os.path.join('D:\\2fre_data', filename)
    data = sio.loadmat(newname)
    channel = data['ChannelData_fre']
    for i in range(data_num_file):
        sub_BB=np.random.randint(0,fre)
        a=channel_norm_factor*channel[:,:,sub_BB,i]
        H_train2[n * data_num_file + i,:,:] = np.transpose(a)
    n=n+1
print(n)
print(H_train2.shape)

# Input3: AWGN
channel_noise=1/np.sqrt(2)*np.random.randn(data_num_train,Nr,1)+1j*1/np.sqrt(2)*np.random.randn(data_num_train,Nr,1)

##### Build model #####
input_dim0 = (Ns,1)
input_dim1 = 2 * Nr * Nt
input_dim2 = (Nr, Nt)
input_dim3 = (Nr,1)
inp0 = Input(shape=input_dim0, dtype=tf.complex64)  # modulated symbol,used for self-defined signal flow function
inp1 = Input(shape=(input_dim1,))  # estimated channel(vector),used for neural network
inp2_ip = Input(shape=input_dim2, dtype=tf.complex64)  # estimated channel(matrix),used for generating equivalent channel
inp2_p = Input(shape=input_dim2, dtype=tf.complex64)  # true channel(matrix),used for self-defined signal flow function
inp3 = Input(shape=input_dim3, dtype=tf.complex64)  # AWGN,used for self-defined signal flow function

# encoder
## model 1: FRF
output_dim1=Nt_RF*Nt
x = Dense(512, activation='relu')(inp1)
x = normalization.BatchNormalization()(x)
x = Dense(256, activation='relu')(x)
x = normalization.BatchNormalization()(x)
x = Dense(128, activation='relu')(x)
x = normalization.BatchNormalization()(x)
x_FRF = Dense(output_dim1, activation='relu')(x)

## model 2: WRF
output_dim2=Nr_RF*Nr
x = Dense(512, activation='relu')(inp1)
x = normalization.BatchNormalization()(x)
x = Dense(256, activation='relu')(x)
x = normalization.BatchNormalization()(x)
x = Dense(128, activation='relu')(x)
x = normalization.BatchNormalization()(x)
x = Dense(64, activation='relu')(x)
x = normalization.BatchNormalization()(x)
x_WRF = Dense(output_dim2, activation='relu')(x)

with tf.device('/cpu:0'):
    h_eq=Lambda(eq_channel_fun, dtype=tf.float32, output_shape=(2*Nr_RF*Nt_RF,))([inp2_ip, x_FRF, x_WRF])

## model 3_1: FBB real
output_dim3=Nt_RF*Ns
x=Dense(20, activation='relu')(h_eq)
x=normalization.BatchNormalization()(x)
x = Dense(40, activation='relu')(x)
x = normalization.BatchNormalization()(x)
x = Dense(20, activation='relu')(x)
x = normalization.BatchNormalization()(x)
x_FBBreal = Dense(output_dim3, activation='linear')(x)

## model 3_2: FBB imag
output_dim3=Nt_RF*Ns
x=Dense(20, activation='relu')(h_eq)
x=normalization.BatchNormalization()(x)
x = Dense(40, activation='relu')(x)
x = normalization.BatchNormalization()(x)
x = Dense(20, activation='relu')(x)
x = normalization.BatchNormalization()(x)
x_FBBimag = Dense(output_dim3, activation='linear')(x)

## model 4_1: WBB_real
output_dim4=Nr_RF*Ns
x = Dense(20, activation='relu')(h_eq)
x = normalization.BatchNormalization()(x)
x = Dense(40, activation='relu')(x)
x = normalization.BatchNormalization()(x)
x = Dense(20, activation='relu')(x)
x = normalization.BatchNormalization()(x)
x_WBBreal = Dense(output_dim4, activation='linear')(x)

## model 4_2: WBB_imag
output_dim4=Nr_RF*Ns
x = Dense(20, activation='relu')(h_eq)
x = normalization.BatchNormalization()(x)
x = Dense(40, activation='relu')(x)
x = normalization.BatchNormalization()(x)
x = Dense(20, activation='relu')(x)
x = normalization.BatchNormalization()(x)
x_WBBimag = Dense(output_dim4, activation='linear')(x)

with tf.device('/cpu:0'):
    detected_symb= Lambda(sig_flow, dtype=tf.float32, output_shape=(2*Ns,))\
    ([inp0,inp2_p,inp3,x_FRF,x_WRF,x_FBBreal,x_FBBimag,x_WBBreal,x_WBBimag])

#decoder
x=Dense(20, activation='relu')(detected_symb)
x = normalization.BatchNormalization()(x)
x = Dense(50, activation='relu')(x)
x = normalization.BatchNormalization()(x)
x = Dense(20, activation='relu')(x)
x = normalization.BatchNormalization()(x)
x = Dense(2*Ns, activation='sigmoid')(x)

model = Model(inputs=[inp0, inp1, inp2_ip, inp2_p, inp3], outputs=x)

adam=Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer=adam, loss='binary_crossentropy')

# checkpoint
filepath='DLJHPFofdm_UMi_3path_2fre_Ns3_10dB_500ep.hdf5'

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True)
callbacks_list = [checkpoint]

model.fit(x=[mod_symb, H_train1, H_train2, H_train2, channel_noise],y=trans_bit_label,epochs=500, batch_size=256, callbacks=callbacks_list, verbose=2, shuffle=True, validation_split=0.1)


#### Testing data ####
data_num_test=1000
data_num_file=1000
# Input0: input bits
trans_bit_test=np.random.randint(0,2,(data_num_test,Ns,2), dtype=int)
trans_bit_test_label=np.reshape(trans_bit_test,(data_num_test,2*Ns))
mod_symb_test=np.zeros((data_num_test,Ns,1), dtype=complex)
for n in range(data_num_test):
    mod_symb_test[n]=QPSK_modu_sim(trans_bit_test[n])
mod_symb_test_T=np.reshape(mod_symb_test,(data_num_test,Ns))
mod_symb_test_label=np.hstack((np.real(mod_symb_test_T),np.imag(mod_symb_test_T)))
print(trans_bit_test[0],'\n',trans_bit_test_label[0],'\n',mod_symb_test[0],'\n',mod_symb_test_label[0])

# Input1&2: perfect CSI
H_test1=zeros((data_num_test,2*Nr*Nt), dtype=float)
filedir = os.listdir('D:\\2fre_data')   # example data
n=0
for filename in filedir:
    newname = os.path.join('D:\\2fre_data', filename)
    data = sio.loadmat(newname)
    channel = data['ChannelData_fre']
    for i in range(data_num_file):
        a = channel_norm_factor * channel[:, :, sub_RF, i]
        h = np.reshape(a, (1, Nt * Nr))
        h_re = np.real(h)
        h_im = np.imag(h)
        h_re_im = np.hstack((h_re, h_im))
        H_test1[n*data_num_file+i]=h_re_im
    n=n+1
print(n)
print(H_test1.shape)

H_test2=zeros((data_num_test,Nr,Nt), dtype=complex)
filedir = os.listdir('D:\\2fre_data')
n=0
for filename in filedir:
    newname = os.path.join('D:\\2fre_data', filename)
    data = sio.loadmat(newname)
    channel = data['ChannelData_fre']
    for i in range(data_num_file):
        sub_BB = np.random.randint(0, fre)
        a = channel_norm_factor * channel[:, :, sub_BB, i]
        H_test2[n * data_num_file + i,:,:] = np.transpose(a)
    n=n+1
print(n)
print(H_test2.shape)

# Input3: AWGN
channel_noise_test=1/np.sqrt(2)*np.random.randn(data_num_test,Nr,1)+1j*1/np.sqrt(2)*np.random.randn(data_num_test,Nr,1)

# load model
model.load_weights('DLJHPFofdm_UMi_3path_2fre_Ns3_10dB_500ep.hdf5')
detected_bit_test=model.predict(x=[mod_symb_test, H_test1, H_test2, H_test2, channel_noise_test], batch_size=1000)
print(detected_bit_test[0])

detected_bit_test2=detected_bit_test
detected_bit_test2[detected_bit_test2<0.5]=0
detected_bit_test2[detected_bit_test2>0.5]=1
error_bits=np.abs(detected_bit_test2-trans_bit_test_label)
print(np.sum(np.abs(detected_bit_test2-trans_bit_test_label))/(data_num_test*2*Ns))

error_bits2=error_bits
print(np.sum(np.abs(error_bits2))/((data_num_test)*2*Ns))

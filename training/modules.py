import numpy as np
import tensorflow as tf
import math
import librosa
from kapre import STFT, Magnitude, MagnitudeToDecibel

def hz_to_midi(hz):
    return 12 * (torch.log2(hz) - np.log2(440.0)) + 69

def midi_to_hz(midi):
    return 440.0 * (2.0 ** ((midi - 69.0)/12.0))

def note_to_hz(note):
    return librosa.core.note_to_hz(note)

def note_to_midi(note):
    return librosa.core.note_to_midi(note)

def hz_to_note(hz):
    return librosa.core.hz_to_note(hz)

def initialize_filterbank(sample_rate, n_harmonic, semitone_scale):
    # MIDI
    # lowest note
    low_midi = note_to_midi('C1')

    # highest note
    high_note = hz_to_note(sample_rate / (2 * n_harmonic))
    high_midi = note_to_midi(high_note)

    # number of scales
    level = (high_midi - low_midi) * semitone_scale
    midi = np.linspace(low_midi, high_midi, level + 1)
    #midi = tf.cast(midi, tf.float32)
    hz = midi_to_hz(midi[:-1])

    # stack harmonics
    harmonic_hz = []
    for i in range(n_harmonic):
        harmonic_hz = np.concatenate((harmonic_hz, hz * (i+1)))

    return harmonic_hz, level


class HarmonicSTFT(tf.keras.layers.Layer):
    def __init__(self,
                 sample_rate=16000,
                 n_fft=513,
                 win_length=None,
                 hop_length=None,
                 pad=0,
                 power=2,
                 normalized=False,
                 n_harmonic=6,
                 semitone_scale=2,
                 bw_Q=1.0,
                 learn_bw=None):
        super(HarmonicSTFT, self).__init__()

        # Parameters
        self.sample_rate = sample_rate
        self.n_harmonic = n_harmonic
        self.bw_alpha = 0.1079
        self.bw_beta = 24.7
        self.win_length=win_length
        self.hop_length=hop_length
        self.n_fft = n_fft
        #self.fft_bins = tf.linspace(0, tf.cast(self.sample_rate//2, tf.int32), self.n_fft//2 + 1)
        self.fft_bins = tf.linspace(0, self.sample_rate//2, self.n_fft//2+1)
        #self.stft = STFT(n_fft = self.n_fft, win_length=self.win_length, hop_length = self.hop_length, window_name='hann_window',  input_shape=(80000,1))
        self.stft = STFT(n_fft=self.n_fft, win_length=self.win_length, hop_length=254, window_name='hann_window', input_shape=(80000,1))
        self.magnitude = Magnitude()
        self.to_decibel = MagnitudeToDecibel()
        self.zero = tf.zeros([1,])
        # Spectrogram

        # Initialize the filterbank. Equally spaced in MIDI scale.
        harmonic_hz, self.level = initialize_filterbank(sample_rate, n_harmonic, semitone_scale)

        # Center frequncies to tensor
        self.f0 = tf.constant(harmonic_hz, dtype='float32')
        # Bandwidth parameters
        self.bw_Q = tf.constant(np.array([bw_Q]), dtype='float32')

    def get_harmonic_fb(self):
        # bandwidth
        bw = (self.bw_alpha * self.f0 + self.bw_beta) / self.bw_Q
        bw = tf.dtypes.cast(tf.expand_dims(bw, axis=0), dtype=tf.float32)
        f0 = tf.dtypes.cast(tf.expand_dims(self.f0, axis=0), dtype=tf.float32)
        fft_bins =tf.dtypes.cast(tf.expand_dims(self.fft_bins, axis=1), dtype=tf.float32)
        #bw = bw.unsqueeze(0) # (1, n_band)
        #f0 = self.f0.unsqueeze(0) # (1, n_band)
        #fft_bins = self.fft_bins.unsqueeze(1) # (n_bins, 1)

        up_slope = tf.matmul(fft_bins, (2/bw)) + 1 - (2 * f0 / bw)
        down_slope = tf.matmul(fft_bins, (-2/bw)) + 1 + (2 * f0 / bw)
        fb = tf.math.maximum(self.zero, tf.math.minimum(down_slope, up_slope))
        return fb
#https://github.com/keunwoochoi/kapre/blob/master/kapre/time_frequency.py
    
    def power_to_db(S, amin=1e-16, top_db=80.0):
        def _tf_log10(x):
            numerator = tf.math.log(x)
            denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
            return numerator / denominator

    # Scale magnitude relative to maximum value in S. Zeros in the output
    # correspond to positions where S == ref.
        ref = tf.reduce_max(S)

        log_spec = 10.0 * _tf_log10(tf.maximum(amin, S))
        log_spec -= 10.0 * _tf_log10(tf.maximum(amin, ref))

        log_spec = tf.maximum(log_spec, tf.reduce_max(log_spec) - top_db)

        return log_spec

    def call(self, input_tensor):
        print(tf.shape(input_tensor))
        waveform = tf.keras.layers.Reshape((-1, 1))(input_tensor)
        spec = self.stft(waveform)
        spec = self.magnitude(spec)
        #spec = power_to_db(spec)
        harmonic_fb = self.get_harmonic_fb()
        harmonic_fb = tf.expand_dims(harmonic_fb, axis=0)
        harmonic_fb = tf.stack([harmonic_fb, harmonic_fb, harmonic_fb, harmonic_fb, harmonic_fb, harmonic_fb,harmonic_fb,harmonic_fb,harmonic_fb,harmonic_fb,harmonic_fb,harmonic_fb,harmonic_fb,harmonic_fb,harmonic_fb,harmonic_fb])
        #harmonic_fb = tf.dtypes.cast(harmonic_fb, dtype=tf.float32)
        harmonic_spec = tf.matmul(tf.transpose(spec, perm=[0,3, 1, 2]), harmonic_fb)
        b, c, w, h = harmonic_spec.shape
        harmonic_spec = tf.keras.layers.Reshape((-1, h//self.n_harmonic, self.n_harmonic))(harmonic_spec)
        harmonic_spec = self.to_decibel(harmonic_spec)
        return harmonic_spec




class ResNet_mtat(tf.keras.layers.Layer):
    def __init__(self, input_channels, conv_channels=128):
        super(ResNet_mtat, self).__init__()
        self.num_class = 50

        # residual convolution
        self.res1 = Conv3_2d(conv_channels, 2)
        self.res2 = Conv3_2d_resmp(conv_channels, 2)
        self.res3 = Conv3_2d_resmp(conv_channels, 2)
        self.res4 = Conv3_2d_resmp(conv_channels, 2)
        self.res5 = Conv3_2d(conv_channels*2, 2)
        self.res6 = Conv3_2d_resmp(conv_channels*2, (2, 3))
        self.res7 = Conv3_2d_resmp(conv_channels*2, (2, 3))

        # fully connected
        self.fc_1 = tf.keras.layers.Dense(conv_channels * 2)
        self.bn = tf.keras.layers.BatchNormalization()
        self.fc_2 = tf.keras.layers.Dense(self.num_class)
        self.activation = tf.keras.layers.Activation('sigmoid')
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.relu = tf.keras.layers.Activation('relu')
        self.gmp = tf.keras.layers.GlobalMaxPooling2D()

    def call(self, x):
        # residual convolution
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.res6(x)
        x = self.res7(x)

        # global max pooling
        x = self.gmp(x)

        # fully connected
        x = self.fc_1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc_2(x)
        x = self.activation(x)
        return x




class Conv3_2d(tf.keras.layers.Layer):
    def __init__(self, output_channels, pooling=2):
        super(Conv3_2d, self).__init__()
        self.conv = tf.keras.layers.Conv2D(output_channels, 3, padding='same')
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.Activation('relu')
        self.mp = tf.keras.layers.MaxPooling2D(pooling)
    def forward(self, x):
        out = self.mp(self.relu(self.bn(self.conv(x))))
        return out


class Conv3_2d_resmp(tf.keras.layers.Layer):
    def __init__(self, output_channels, pooling=2):
        super(Conv3_2d_resmp, self).__init__()
        self.conv_1 = tf.keras.layers.Conv2D(output_channels, 3, padding='same')
        self.bn_1 = tf.keras.layers.BatchNormalization()
        self.conv_2 = tf.keras.layers.Conv2D(output_channels, 3, padding='same')
        self.bn_2 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.Activation('relu')
        self.mp = tf.keras.layers.MaxPooling2D(pooling)
    def forward(self, x):
        out = self.bn_2(self.conv_2(self.relu(self.bn_1(self.conv_1(x)))))
        out = x + out
        out = self.mp(self.relu(out))
        return out


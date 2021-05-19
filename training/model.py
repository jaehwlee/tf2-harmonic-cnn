# coding: utf-8
import numpy as np
import tensorflow as tf
from modules import HarmonicSTFT


class HarmonicCNN(tf.keras.Model):
    def __init__(
        self,
        conv_channels=128,
        sample_rate=16000,
        n_fft=512,
        n_harmonic=6,
        semitone_scale=2,
        learn_bw=None,
        dataset="mtat",
    ):
        super(HarmonicCNN, self).__init__()

        self.hstft = HarmonicSTFT(
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_harmonic=n_harmonic,
            semitone_scale=semitone_scale,
            learn_bw=learn_bw,
        )
        self.hstft_bn = tf.keras.layers.BatchNormalization()

        # 2D CNN
        if dataset == "mtat":
            from modules import ResNet_mtat as ResNet
        self.conv_2d = ResNet(input_channels=n_harmonic, conv_channels=conv_channels)

    def call(self, x, training=False):
        # harmonic stft
        x = self.hstft_bn(self.hstft(x, training=training), training=training)

        # 2D CNN
        logits = self.conv_2d(x, training=training)

        return logits

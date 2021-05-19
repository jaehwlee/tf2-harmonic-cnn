Tensorflow2 implementation of Data-driven Harmonic Filters for Audio Representation Learning
==


**Data-driven Harmonic Filters for Audio Representation Learning**

Minz Won, Sanghyuk Chun, Oriol Nieto, and Xavier Serra

ICASSP, 2020

Reference
--
* [pdf](https://ccrma.stanford.edu/~urinieto/MARL/publications/ICASSP2020_Won.pdf)
* [pytorch](https://github.com/minzwon/data-driven-harmonic-filters/blob/master/README.md)


Prerequisited
--
* Tensorflow (>=2.2) 
* Kapre : <code> pip install kapre</code> ,[ for more information](https://github.com/keunwoochoi/kapre)

Usage
--
* Requirements
<pre>
<code>
$ conda env create -n {ENV_NAME} --file environment.yaml
$ conda activate {ENV_NAME}
</code>
</pre>


* Preprocessing
<pre>
<code>
$ python -u preprocess.py run ../dataset
$ python -u split.py run ../dataset
</code>
</pre>

* Training
<pre>
<code>
$ python train.py
</code>
</pre>

* Options
<pre>
<code>
'--conv_channels', type=int, default=128
'--sample_rate', type=int, default=16000
'--n_fft', type=int, default=512
'--n_harmonic', type=int, default=6
'--semitone_scale', type=int, default=2
'--learn_bw', type=str, default='only_Q', choices=['only_Q', 'fix']
'--input_legnth', type=int, default=80000
'--batch_size', type=int, default=16
'--log_step', type=int, default=19
'--model_save_path', type=str, default='./../saved_models'
'--gpu', type=str, default='0'
'--data_path', type=str, default='./../../tf-harmonic-cnn/dataset'
</code>
</pre>

To be changed
--
tf.keras.optimizers.adam -> tfa.optimizers.AdamW  
tf.keras.optimizers.sgd -> tfa.optimizers.SGDW

Author
--
* Jaehwan Lee @jaehwlee
* jaehwlee@gmail.com

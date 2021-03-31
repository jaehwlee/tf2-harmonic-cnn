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
* Kapre : <code> pip install kapre</code> ,[for more information](https://github.com/keunwoochoi/kapre)

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


Author
--
* Jaehwan Lee @jaehwlee
* jaehwlee@gmail.com

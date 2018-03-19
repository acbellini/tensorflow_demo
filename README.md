# Demo per XPUG Bologna - 13 Marzo 2017

1. Installare Anaconda per python 3.6 da qui: https://www.anaconda.com/download/#linux

2. Creare un conda env

Se avete una GPU Nvidia:

```
conda create -n xpug python=3.6 jupyter notebook tensorflow-gpu
```

Se non avete una GPU Nvidia:

```
conda create -n xpug python=3.6 jupyter notebook tensorflow
```


3. Attivate l'environment

```
conda activate xpug
```

4. Clonate il progetto da git e fate partire il notebook

```
git clone https://github.com/acbellini/tensorflow_demo
cd tensorflow_demo
jupyter notebook
```

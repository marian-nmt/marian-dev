Stuff required to build module (on Linux)
```
python3 -m venv ./venv
source ./venv/bin/activate
python -mpip install sentencepiece scikit-build pybind11
```

Stuff required to run windowed demo
```
python -mpip install pyqt5 sacremoses git+https://github.com/mediacloud/sentence-splitter
```

Build the module (CPU version)
```
python setup.py build -j16 install
```

```
echo "Hello World." | python test_translate.py '--config /home/marcinjd/MTMA/decoder.yml' /home/marcinjd/MTMA/source.spm /home/marcinjd/MTMA/target.spm
```
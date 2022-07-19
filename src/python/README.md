python3 -m venv ./venv
source ./venv/bin/activate
python -mpip install sentencepiece

python setup.py build --verbose --parallel 16 install

echo "Hello World." | python test.py '--config /home/marcinjd/MTMA/decoder.yml' /home/marcinjd/MTMA/source.spm /home/marcinjd/MTMA/target.spm
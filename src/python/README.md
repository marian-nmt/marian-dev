python3 -m venv ./venv
source ./venv/bin/activate
python -mpip install numpy sentencepiece

python setup.py build --verbose --parallel 16 install

# spm is dying on my at the moment
# echo "Hello World." | spm_encode --model ~/MTMA/source.spm | python test.py | spm_decode --model ~/MTMA/target.spm
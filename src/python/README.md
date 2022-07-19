python3 -m venv ./venv
source ./venv/bin/activate
python -mpip install numpy

python setup.py build --verbose --parallel 16 install

set -eux

dir=$(pwd)
python3 kai_v2_model.py --with_kai --with_kai_v2 --mode=train


set -eux

dir=$(pwd)
python3 train.py --with_kai_v2= --mode=predict #生成计算图
python3 infer_dsl.py # 生成 leaf config
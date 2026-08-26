
set -eux

dir=$(pwd)
python3 kai_v2_model.py --with_kai_v2 --mode=predict #生成计算图

python3 dynamic_config_json.py #生成 dragonfly pipeline

python3 check_dnn_yaml.py training/dnn-plugin.yaml infer/dnn_model.yaml
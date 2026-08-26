set -eux

dir=$(pwd)
# 生成 kuiba 特征配置
python3 gen_config.py
# 将 kuiba 格式转换为 kai 格式
~/project/kuiba_to_kai_conf --kuiba_config_filename dynamic_json_config.json --kai_old_config_filename ./conf/kai_kuiba_config.json --kuiba_to_kai_use_mio_mapping=false
# 执行 数据流消费 pipeline
python3 pipeline.py
# 执行 train
# python3 train.py
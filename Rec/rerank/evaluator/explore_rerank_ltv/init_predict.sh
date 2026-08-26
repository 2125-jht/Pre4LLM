#!/bin/bash
#source ~/model/venv/bin/activate
#source ~/model/kuiba_to_kai/mio_py_env/activate.zsh
#cur_dir=$(cd "$(dirname "$0")";pwd)
#echo $cur_dir


# set -e
# cur_dir=$(cd "$(dirname "$0")";pwd)
# echo $cur_dir
# cd ~
# source mio_py_env/activate

# cd $cur_dir

# # e: 若脚本传回值非0 直接结束shell u: 只识别定义过的变量; x: 以调试的方式执行shell;
# set -eux

# dir=$(pwd)
python3 kai_v2_model.py --with_kai_v2= --mode=predict #生成计算图
cd $dir/infer_server

python3 dynamic_config_json.py #生成 dragonfly pipeline

#!/bin/sh

# 改动前务必联系wangyalong03确认！！！！！！


if [ $# -ne 3 ]; then
  echo "usage: $0 <kuiba_config_json_filename> <model_filename> <kai_job_dir>"
  exit 0;
fi

stage1="映射文件kai_kuiba_config.json"
stage2="KAI同步计算图"

function catch_error() {
  echo "❎ 生成${1} 失败,请查看执行过程中报错信息"
  exit 1
}

function finish_stage() {
  echo "✅ 成功生成$1"
}

./kuiba_to_kai_conf --kuiba_config_filename $1 --kai_old_config_filename $3/conf/kai_kuiba_config.json --kuiba_to_kai_use_mio_mapping=false || catch_error ${stage1}
finish_stage ${stage1}
python3 $2 --with_kai --text train || catch_error ${stage2}
finish_stage ${stage2}

mkdir -p $3
mkdir -p $3/conf
cp kai_kuiba_config.json $3/conf
cp ./training/conf/* $3/conf
echo "✅ 拷贝配置文件至${3}"

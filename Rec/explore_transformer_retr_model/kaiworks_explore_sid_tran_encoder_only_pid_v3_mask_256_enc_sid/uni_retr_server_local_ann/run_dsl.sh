#!/bin/bash
# e: 若脚本传回值非0 直接结束shell u: 只识别定义过的变量; x: 以调试的方式执行shell;
set -eux

CURPATH=$(cd "$(dirname "$0")"; pwd)
ROOT=${CURPATH}/../

sh gen_infer_local.sh
cd ${ROOT}/uni_retr_server_local_ann
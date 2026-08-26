print("如果报错找不到 libcuda.so.1 ，请先执行 export LD_PRELOAD=/usr/lib64/libjemallocx.so:/usr/local/cuda-11.4/compat/libcuda.so.1")
print()
# 如果你的模型描述文件是tf_graph.py，请 import tf_graph
import model
import kai_v2_model

import kai.tensorflow as config
from kai.tensorflow.compile_time.kai_ps_compile_time import KaiPsCompileTime
kpct = KaiPsCompileTime()
kai_config = config.Config()
kpct.compile(kai_config)
kpct.dump_dnn_plugin_yaml(".")
print("完成生成 dnn-plugin.yaml 及 graph.pb 等文件")

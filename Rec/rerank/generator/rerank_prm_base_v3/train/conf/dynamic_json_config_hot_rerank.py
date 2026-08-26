# -*- coding: UTF8 -*-
import json
import sys
sys.path.append("../")

from manager.loss.loss_function_hot_rerank import hotReRankFeatureconf,hotReRankLossFuction

################ 配置自己的子类，实现loss和用哪些feature
loss_function_conf = hotReRankLossFuction
feature_conf = hotReRankFeatureconf
reader_name = "kuiba_tf_hot_rerank_model_v2"
shard_num = 10
# grpc_name = "grpc_krpTfHotRerankTmodelPredictServerV1"
################


config_learner = {
    # 读者类型
    "reader_type": "kafka",
    # 对应 ps server SampleReader 的 reader_name, 是模型训练的唯一标识
    "reader_name": reader_name,
    # ps shard 数
    "shard_num": shard_num,
    # 读取线程
    "reading_threads": 4,
    # 训练线程
    "learner_threads": 8,
    # mini batch: 一次计算多少样本
    "mini_batch_size": 256,
    # network 的梯度 多少个 mini_batch 后更新回参数服务器
    "merge_size": 16,
    "fetch_sample": {
        "pass_size": 1024,
        "read_size": 128,
        "compress_sample": True,
    },
    "kafka": {
        "cluster": [
            "bjlt-reco2"
        ],
        "offset_ms_ago": 1000*3600*0.0833,
        "pass_size": 2560,
        "read_size": 256,
        "reader_name": reader_name,
        "topic": "reco_hot_context_rank_joint_log",
    },
    "bt_queue": {
        "pass_size": 1024,
        "read_size": 4096,
        "group": "hdfs_realtime_log",
        "queue_name": "acfun0",
        "reader_name": reader_name,
        "begin_time": "20190301000000",
        "compress_sample": True,
    },
    "hdfs": {
        "pass_size": 2560,
        "read_size": 2560,
        "version": 0,
        # "common_prefix": "/home/reco/rawdata/app/kuaishou/reco_new_follow_model_rank_log",
        "data_path": ["2019-10-10"],
    },
    # 自行改动
    "__SERVER_NAME": "paas_learner_server",
    "__SERVER_PART": "default_part",
    "__SERVER_SHARD": "default_shard",
    "eval_interval_seconds": 300
}

config_ps = {
    # shm 的目录
    "shm_dir": "/dev/shm/ps",
    # ps 内存上限
    "ps_memory": (1 << 30) * 200,
    # ps 参数数量上限
    "ps_capacity": 700000000,
    "max_feature_num": 600000000,
    # ps 分成几个 shard
    "shard_num": shard_num,
    "sample_reader": {
        "type_name": "BTQueueSampleReader",
        "group": "hdfs_realtime_log",
        "begin_time": "20190924183000",
        # 对应模型的名字, 作为一个训练任务的唯一标识
        "reader_name": reader_name,
        "shard_num": shard_num,
        "buffer_size": 8192,
        "queues": [],
    },
    "eval": {
        "eval_window_seconds": 300,
        "eval_window_samples": 0,
    },
    "network": {
        "shard_num": shard_num,
        "model_queue": {
            # "queue_group": "infra_mem_kuiba_follow_model_lf",
            "queue_name": reader_name,
            "shard_num": 12,
            "batch_update_period_seconds": 7200,
            "warmup": True,
            "queue_handler": "infra_btq",
            "min_feature_score": 5.0
        },
        "checkpoint": {
            "path": "viewfs:///home/reco/xuwei09/"+reader_name,
            "save_interval_seconds" : 3600*4,
            "part_key_num" : 10000000,
            # "reserve_days" : 7,
        },
        "updater_type": "ada_momentum_updater",
        "type": "TFNetwork",
        # PS(参数服务器) 配置
        "parameters": feature_conf.get_features_conf_pool(),
        # 网络 layer 配置
        "layers": {
            "default_batch_num": 1,
            "default_batch_decay": 0.9999,
            "default_move_length": 0.001,
            "default_initial_lr": 0.000005,
            "default_mom_decay_rate": 0.99,
            "default_ada_decay_rate": 0.9999,
        },
        # 损失函数配置
        "loss_functions": loss_function_conf.get_loss_function()
    },
    # "model_loader": {
    #     "consumer": "ps",
    #     "load_network_enable": False,
    #     "model_path": "viewfs:/home/reco/data/zhengjie/krp_hot-rank_rerank-v1/checkpoints/1588219200",
    #     "producer": "kuiba",
    #     "shard_num": 35,
    #     "slots": [
    #     ],
    #     "threads": 8,
    #     "user": "reco"
    # },
    "metric": {
        "{}_{}".format(name, i): [
            "count",
            "loss@real_loss",
            "rate",
            "xgauc",
            "xauc",
            "auc",
        ] for name in ['pos', 'aux'] for i in range(10)      
    },    
    "online_update": True,
    "__SERVER_NAME": "paas_ps_server",
    "__SERVER_PART": "default_part",
    "__SERVER_SHARD": "yz",
    "server_config": {
    },
    "kess_config": {
    },
    "disable_rename_model_queue":True,
    "status": "training",
}

# config_predict = {
#   "kess_config" : {
#     grpc_name : {},
#   },
#   "model_queue": {
#     "queue_handler": "infra_btq",
#     "queue_group" : "infra_mem_lf_krp_common_model2",
#     "queue_name" : reader_name,
#     "shard_num" : 12,
#     "cache_ps_type": "TFCachePs",
#     "predictor_type": "TFPredictor",
#     "warmup": True
#   },
#   "ps_memory" : (1 << 30) * 200,
#   "ps_capacity" : (1 << 31),
#   "empirical_key_num" : 100000,
#   "empirical_wait_seconds" : 300,
# }

config = {
    "krp_ps_server": config_ps,
    "krp_tf_learner_server": config_learner
}

# print(json.dumps(config_ps, indent=2))
if __name__ == "__main__":
    import sys


    if len(sys.argv) == 1:
        print(json.dumps(config, indent=2))
    else:
        if len(sys.argv[1].split("ps")) > 1:
            print(json.dumps(config_ps, indent=2))
        elif len(sys.argv[1].split("predict")) > 1:
            print(json.dumps(config_predict, indent=2))
        else:
            print(json.dumps(config_learner, indent=2))
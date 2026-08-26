# -*- coding: UTF-8 -*-
import json

shard_num = input("请将此脚本与 server_static.flags 和 dynamic_json_config.json 放在同一目录下，并输入 embedding server 的 shard_num: ")
shard_num = int(shard_num)

shm_embd_capacity = -1
with open("server_static.flags", "r") as flag_file:
  lines = flag_file.readlines()
  for line in lines:
    if line.find("shm_embd_capacity") > 0:
      shm_embd_capacity = int(line[line.find("=")+1:])
    if line.find("shm_embd_memory") > 0:
      shm_embd_memory = int(line[line.find("=")+1:])
if shm_embd_capacity == -1:
  print("未在 server_static.flags 文件中找到 shm_embd_capacity，配置翻译失败")
  exit()
if shm_embd_memory == -1:
  print("未在 server_static.flags 文件中找到 shm_embd_memory，配置翻译失败")
  exit()

with open("dynamic_json_config.json", "r") as file:
  file_content = file.read()

obj = json.loads(file_content)

static_json_data = '''
{
  "msg_queue_shards": 4
}
'''
static_json = json.loads(static_json_data)
static_json["msg_queue_shards"] = obj["queue_shard_num"]
static_data = json.dumps(static_json, indent=2, separators=(',', ': '))
print("\n生成配置如下:\n")
print("静态配置:\n" + static_data)

result_json_data = '''
{
  "biz_type": "index",
  "data_type": "embedding",
  "store_format": "Embedding",
  "default_get_format": "Embedding",
  "btq": {
    "type": "shard",
    "topic": "{topic_name}",
    "shard_num": 2,
    "pop_num": 64
  },
  "consume": {
    "type": "embedding",
    "thread_num": 64
  },
  "embedding_config": {
    "sign_format": "kuiba",
    "updater_type": "mio"
  },
  "maximum_key_count": "10000000000",
  "storage_quota": "225485783040",
  "default_expire_duration": "86400s",
  "kv_store_rpc_server_type": 2,
  "replicate_retry_wait": "90s",
  "checkpoint_path": "/home/reco_6/colossusdb_embd/reco_test",
  "task_executor_thread_num": 64
}
'''
result_json = json.loads(result_json_data)

result_json["btq"]["topic"] = obj["queue_prefix"]
result_json["btq"]["shard_num"] = obj["queue_shard_num"]
if "shard_offset" in obj:
  result_json["btq"]["shard_offset"] = obj["shard_offset"]
result_json["embedding_config"]["sign_format"] = obj["sign_format"]

def add_config_ifexist(key):
  if key in obj:
    result_json["embedding_config"][key] = obj[key]

add_config_ifexist("read_slots")
add_config_ifexist("force_expire_timet")
add_config_ifexist("missed_key_sample_rate")
add_config_ifexist("missed_key_kafka_topic")
result_json["maximum_key_count"] = shm_embd_capacity * shard_num 
result_json["storage_quota"] = shm_embd_memory
result_json["fallback_replicator_config"] = \
  '''{\"type_name\": \"BtEmbeddingFallbackReplicatorClient\", \"shard_num\": ''' \
  + str(shard_num) + ''', \"shm_embd_capacity\": ''' + str(shm_embd_capacity) \
  + ''', \"queue_prefix\":\"''' + obj["queue_prefix"] + '''\"}'''

data2 = json.dumps(result_json, indent=2, separators=(',', ': '))
print("动态配置:\n" + data2)
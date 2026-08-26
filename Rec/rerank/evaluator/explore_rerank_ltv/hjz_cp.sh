#download website:  https://halo.corp.kuaishou.com/api/cloud-storage/v1/public-objects/dev/devctl
#
chmod +x devctl
./devctl  cp hot-zhanghaijun03-01.dev.kwaidc.com:/home/zhanghaijun03/projects/fountain_ltr_model_workspace/hot_models/kai/ltvs/ui_ltv_explore/infer_server/dynamic_config_json.json ./config/dynamic_json_config.json

./bin/load.sh restart

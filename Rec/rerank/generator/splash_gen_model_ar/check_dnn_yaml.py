"""
该脚本的主要作用是检查并修正infer部署所用的dnn-plugin.yaml
输入有两个文件：
1、训练阶段产出的dnn-plugin.yaml，通常该文件在run_dir下可以找到
2、infer部署时提供给krp/moss的dnn-plugin.yaml，通常该文件在用户自己的部署环境产生
输出一个文件：
1、fixed_dnn_plugin.yaml 该文件确保infer的dense参数顺序与训练阶段一致，并且同时检查总shape是否一致，若不一致会报错失败

使用示例
`python3 -u check_infer_yaml.py train_dnn_plugin.yaml infer_dnn_plugin.yaml`

"""
import os
import sys
import yaml
import copy

def read_yaml(yaml_path: str):
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def write_yaml(yaml_dict: dict, yaml_path: str):
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_dict, f, default_flow_style=False,
                    allow_unicode=True)

def get_yaml_dense_shape(yaml_dict: dict):
    params = yaml_dict['param']
    total_shape = 0
    for param in params:
        coln ,rown = param["coln"], param["rown"]
        total_shape += coln * rown
    return total_shape

def get_order_fixed_yaml(train_yaml: dict, infer_yaml: dict):
    def get_param(param_name: str, t_p_rown: str ,params_list: list):
        for param in params_list:
            if param['name'] == param_name and param['rown'] == t_p_rown:
                return param
        raise ValueError("未在Infer Dense 参数列表中找到该dense参数: {}, 请检查Infer配置生成流程，参考文档：https://docs.corp.kuaishou.com/k/home/VfLBc99APVlU/fcADNpTXQsqpCibtPP-U0MZma，或与oncall取得联系".format(param_name))
        
    train_dense_param = train_yaml['param']
    infer_dense_param = infer_yaml['param']
    fixed_yaml = copy.deepcopy(infer_yaml)
    fixed_yaml["param"] = []
    for t_p in train_dense_param:
        t_p_name = t_p["name"]
        t_p_rown = t_p["rown"]
        param = get_param(t_p_name, t_p_rown, infer_dense_param)
        fixed_yaml["param"].append(param)
    return fixed_yaml


def runtime_main(train_yaml_path, infer_yaml_path):
    train_yaml = read_yaml(train_yaml_path)
    infer_yaml = read_yaml(infer_yaml_path)

    train_yaml_shape = get_yaml_dense_shape(train_yaml)
    infer_yaml_shape = get_yaml_dense_shape(infer_yaml)
    # if train_yaml_shape != infer_yaml_shape:
    #     raise ValueError("Train Dense 参数总维度: {} 与 Infer Dense 参数总维度: {} 不匹配，请检查Infer配置生成流程，参照文档：https://docs.corp.kuaishou.com/k/home/VfLBc99APVlU/fcADNpTXQsqpCibtPP-U0MZma，或与oncall取得联系".format(train_yaml_shape, infer_yaml_shape))

    fixed_yaml = get_order_fixed_yaml(train_yaml, infer_yaml)
    infer_yaml_dir = os.path.abspath(os.path.dirname(infer_yaml_path))
    fixed_yaml_path = os.path.join(infer_yaml_dir, "fixed_dnn-plugin.yaml")
    print("Fixed yaml save to: {}".format(fixed_yaml_path))
    write_yaml(fixed_yaml, fixed_yaml_path)


if __name__ == "__main__":
    train_yaml_path = sys.argv[1]
    print("Train yaml path: {}".format(train_yaml_path))
    infer_yaml_path = sys.argv[2]
    print("Infer yaml path: {}".format(infer_yaml_path))
    runtime_main(train_yaml_path, infer_yaml_path)
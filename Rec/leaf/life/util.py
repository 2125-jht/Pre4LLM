#!/usr/bin/env python3
# coding=utf-8

def enrich_ab_param(ab_params):
  new_ab_params = []
  for param in ab_params:
    if isinstance(param, dict):
      if "attr_name" not in param:
        param["attr_name"] = param["param_name"]
      new_ab_params.append(param)
    elif isinstance(param, tuple):
      new_param = {}
      new_param["param_name"] = param[0]
      new_param["default_value"] = param[1]
      if len(param) == 3:
        assert isinstance(param[2], str), "参数名必须是 str 类型"
        new_param["attr_name"] = param[2]
      else:
        new_param["attr_name"] = param[0]
      new_ab_params.append(new_param)
  return new_ab_params
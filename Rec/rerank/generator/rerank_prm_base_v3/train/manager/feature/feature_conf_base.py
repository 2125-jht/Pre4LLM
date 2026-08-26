# -*- coding: UTF8 -*-
import os
import sys
import json


class FeatureConfBasePool(object):
    def __init__(self, pure_feature_config, embedding_hyper_parameter_conf,lambda_feature_config=None):
        """
        :param pure_feature_config: 只含有feature {str:dict}
        :param embedding_hyper_parameter_conf: 只含有超参数配置，{str:float}
        """
        self.feature_config = self._set_key_type(pure_feature_config)
        assert self.feature_config
        if lambda_feature_config:
            self.feature_config = lambda_feature_config(self.feature_config)
        # 所有的feature input name
        self.all_input_feature = self._set_all_input_feature(self.feature_config)

        # 设置默认的参数，learning rate 需要在这里面该
        self._set_embedding_hyper_parameter(embedding_hyper_parameter_conf)

    def _print(self):
        print(json.dumps(self.feature_config, indent=2))

    def _print_inputs_feature(self):
        print(self.all_input_feature)

    def get_features_conf_pool(self):
        return self.feature_config

    def get_features_dim(self,fea_name):
        default_dim = self.feature_config['default_dim']
        return self.feature_config[fea_name].get('dim',default_dim)

    def get_features_join_limit(self,fea_name):
        # 写死 200 和kuiba 默认参数同步
        default_join_limit = 200
        return self.feature_config[fea_name].get('join_limit', default_join_limit)

    def get_features_type(self,fea_name):
        default_type = self.feature_config['default_type']
        return self.feature_config[fea_name].get('type', default_type)

    def get_inputs_feature(self):
        return self.all_input_feature

    def _set_embedding_hyper_parameter(self, embedding_hyper_parameter_conf):
        # set_default_hyper_parameter 必须在最后执行，填充缺省参数，不然会影响input_feature_list生成
        assert len(self.feature_config) != 0
        assert len(embedding_hyper_parameter_conf) != 0

        # check_set = {'default_online_push_limit',
        #              'default_type',
        #              # type 默认是 1:EMBEDDING_PARAMETER, 其它类型有 2:LR_PARAMETER 3:THIRD_PATRY_DATA 5:flat list
        #              'default_dim',
        #              'default_batch_num',
        #              'default_batch_decay',
        #              'default_move_length',
        #              'default_expire_second',
        #              'default_decay_rate',
        #              'default_initial_lr',
        #              'default_initial_g2sum',
        #              'default_init_stddev'}

        for hp_name, value in embedding_hyper_parameter_conf.items():
            # assert hp_name in check_set
            self.feature_config[hp_name] = value
            # check_set.remove(hp_name)

        # 设置的超参数 少于预期
        # assert len(check_set) == 0

    def _set_key_type(self, feature_config):
        """
        :param feature_config:
        :return:all_input_features_list，valid_feature_config
         feature_config 仅含有feature
         统一分配 key_types,从100起步，向上累加
        """
        key_index = 100
        key_sort = sorted(feature_config.keys())
        for key in key_sort:
            extractor = feature_config[key]
            assert len(extractor['attrs']) == 1
            extractor['attrs'][0]['key_type'] = key_index
            key_index += 1
        return feature_config

    def _set_all_input_feature(self, feature_config):
        """
        :param feature_config:
        :return: all_input_features_list
        feature_config 仅含有feature
        """
        all_input_features_list = []
        for k, v in feature_config.items():
            all_input_features_list.append(k)
        return all_input_features_list

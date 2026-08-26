# -*- coding: UTF8 -*-

import json

class LossFunctionBase(object):
    def __init__(self, single_loss_list, feature_conf_pool):
        default_inputs = feature_conf_pool.get_inputs_feature()
        assert default_inputs and len(default_inputs) > 0
        self.loss_function = {}
        self.feature_conf_pool = feature_conf_pool
        self.single_loss_list = single_loss_list
        for single_loss in single_loss_list:
            self.loss_function[single_loss.loss_name] = single_loss.genrate_loss_conf(default_inputs)

    def get_loss_function(self):
        return self.loss_function

    def get_hasweight_loss_list(self):
        return list(map(lambda single_loss:single_loss.loss_name, filter(lambda single_loss: single_loss.weight_name is not None,self.single_loss_list)))

    def _get_loss_2_feature(self, loss_name):
        single_loss = None
        for term in self.single_loss_list:
            if term.loss_name == loss_name:
                single_loss = term
                break
        assert single_loss, "loss_name = {}".format(loss_name)
        fea_name_2_dim = {}
        for input_fea_name in single_loss.input_feature_list:
            f_type = self.feature_conf_pool.get_features_type(input_fea_name)
            f_dim = self.feature_conf_pool.get_features_dim(input_fea_name)
            if int(f_type) == 5:
                join_limit = self.feature_conf_pool.get_features_join_limit(input_fea_name)
                fea_name_2_dim[input_fea_name] = int(f_dim) * int(join_limit)
            else:
                fea_name_2_dim[input_fea_name] = int(f_dim)
        assert fea_name_2_dim and len(fea_name_2_dim) > 0
        return fea_name_2_dim

    def gen_dryrun_input(self, loss_name):   
        import tensorflow as tf     
        res = {}
        features = self._get_loss_2_feature(loss_name)
        for feature,input_dim in features.items():
            res[feature] = tf.placeholder(tf.float32, shape=[None, input_dim])
        return res

    def _print(self):
        print(json.dumps(self.loss_function, indent=2))


class SingleLoss():
    def __init__(self, loss_name, pos_lable_name, neg_lable_name, weight_name=None, input_feature_list=None,input_feature_mmoe_list=None):
        # sample_rate：默认是1.0
        # expired_output：默认是0.0 ,
        # 实际上是定义每个 label 的采样率（sample_rate），
        # 和label映射到loss 输入时候的值（expired_output）
        # input_feature_list = None,默认用所有特征
        self.loss_name = loss_name
        self.input_feature_list = input_feature_list
        self.pos_lable_name = pos_lable_name
        self.neg_lable_name = neg_lable_name
        self.weight_name = weight_name
        self.input_feature_mmoe_list = input_feature_mmoe_list

    def genrate_loss_conf(self, default_inputs):
        labels_info = {}
        labels_info[self.neg_lable_name] = {}
        labels_info[self.pos_lable_name] = dict(sample_rate=1.0, expired_output=[1.0])
        if not self.input_feature_list:
            self.input_feature_list = default_inputs

        if not self.input_feature_mmoe_list:
            self.input_feature_mmoe_list = default_inputs

        self._check_inputs_feature(default_inputs)
        single_loss = dict(
            inputs=list(map(lambda f: 'param.' + f, self.input_feature_list)),
            labels=labels_info,
            type='LogLoss',
            auc_uid='uId',
        )
        if self.weight_name:
            single_loss['weight'] = self.weight_name
        return single_loss

    def _check_inputs_feature(self, default_inputs):
        """
        :param default_inputs:
        :return:
        check inputs_feature 是否合理
        """
        for input in self.input_feature_list:
            try:
                assert input in default_inputs
            except AssertionError:
                print("all_inputs:" + str(default_inputs))
                print("input:" + str(input))
                raise AssertionError

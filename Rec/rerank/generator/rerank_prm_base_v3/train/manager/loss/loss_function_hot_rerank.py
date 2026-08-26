# -*- coding: UTF8 -*-

from manager.feature.feature_conf_hot_rerank import hotReRankFeatureconf
from manager.loss.loss_function_base import LossFunctionBase, SingleLoss


class HotReRankLossFuction(LossFunctionBase):
    def __init__(self, feature_conf_pool):
        single_loss_list = self.set_single_loss_list()
        super(HotReRankLossFuction, self).__init__(single_loss_list, feature_conf_pool)
        # self._print()

    def set_single_loss_list(self):
        return [SingleLoss(loss_name="hot_listwise_loss", pos_lable_name="maskPos", neg_lable_name="maskNeg",
                           input_feature_list=None, weight_name=None),
                SingleLoss(loss_name="hot_ctr_loss", pos_lable_name="maskPos", neg_lable_name="maskNeg",
                           input_feature_list=None, weight_name=None),
                SingleLoss(loss_name="hot_like_loss", pos_lable_name="maskPos", neg_lable_name="maskNeg",
                           input_feature_list=None, weight_name=None),
                SingleLoss(loss_name="hot_follow_loss", pos_lable_name="maskPos", neg_lable_name="maskNeg",
                           input_feature_list=None, weight_name=None),
                SingleLoss(loss_name="hot_profile_loss", pos_lable_name="maskPos", neg_lable_name="maskNeg",
                           input_feature_list=None, weight_name=None),
                SingleLoss(loss_name="hot_longview_loss", pos_lable_name="maskPos", neg_lable_name="maskNeg",
                           input_feature_list=None, weight_name=None),
                SingleLoss(loss_name="slide_listwise_loss", pos_lable_name="slideMaskPos", neg_lable_name="slideMaskNeg",
                           input_feature_list=None, weight_name=None),
                SingleLoss(loss_name="slide_ctr_loss", pos_lable_name="slideMaskPos", neg_lable_name="slideMaskNeg",
                           input_feature_list=None, weight_name=None),
                SingleLoss(loss_name="slide_longview_loss", pos_lable_name="slideMaskPos", neg_lable_name="slideMaskNeg",
                           input_feature_list=None, weight_name=None),
                SingleLoss(loss_name="slide_interact_loss", pos_lable_name="slideMaskPos", neg_lable_name="slideMaskNeg",
                           input_feature_list=None, weight_name=None),
                SingleLoss(loss_name="slide_evtr_loss", pos_lable_name="slideMaskPos", neg_lable_name="slideMaskNeg",
                           input_feature_list=None, weight_name=None),
                # SingleLoss(loss_name="slide_loss", pos_lable_name="slideMaskPos", neg_lable_name="slideMaskNeg",
                #            input_feature_list=None, weight_name=None),
                ]


hotReRankLossFuction = HotReRankLossFuction(hotReRankFeatureconf)


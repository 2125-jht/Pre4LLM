# -*- coding: UTF8 -*-

from manager.feature.feature_conf_base import FeatureConfBasePool
from manager.utils.parameter_config import (numericExtractor, listExtractor,
                                            intExtractor, floatDiscreteExtractor)
import copy
from manager.data_source import hot_rerank_conf

class HotReRankFeatureConf(FeatureConfBasePool):
    def __init__(self):
        """
        :return: newFollowconf.get_features_conf_pool() parameters
        新的业务重新继承FeatureConfBasePool，配置特征和embeding 层的缺省超参
        """
        self.idx_len = 10
        embeding_hyper_parameter_conf = self.set_embeding_hyper_parameter()
        pure_feature_conf = self.set_feature_conf_pool()
        # print(sorted(pure_feature_conf.keys()))
        super(HotReRankFeatureConf, self).__init__(pure_feature_conf, embeding_hyper_parameter_conf,
                                                   lambda_feature_config=self.lambda_feature_config_function)

    def lambda_feature_config_function(self, feature_conf):
        # (vthanos)和老版本(v4)的特征命名区分，加统一后缀
        res = {}
        for fn, fc in feature_conf.items():
            if fn in self.get_user_feature():
                res[fn] = fc
            elif fn in self.set_slide_feature().keys():
                for idx in range(0, 6):
                    res[fn + "_idx" + str(idx)] = copy.deepcopy(fc)
                    assert len(res[fn + "_idx" + str(idx)]["attrs"]) == 1, " feature name = {},idx = {}, attrs={}".format(
                        str(fn), str(idx), str(res[fn + "_idx" + str(idx)]["attrs"]))
                    # 默认attrs只有一个attr
                    res[fn + "_idx" + str(idx)]["attrs"][0]["attr"][0]= str(res[fn + "_idx" + str(idx)]["attrs"][0]["attr"][0]) + "_idx" + str(idx)
            else:
                for idx in range(0, self.idx_len):
                    res[fn + "_idx" + str(idx)] = copy.deepcopy(fc)
                    assert len(res[fn + "_idx" + str(idx)]["attrs"]) == 1, " feature name = {},idx = {}, attrs={}".format(
                        str(fn), str(idx), str(res[fn + "_idx" + str(idx)]["attrs"]))
                    # 默认attrs只有一个attr
                    res[fn + "_idx" + str(idx)]["attrs"][0]["attr"][0]= str(res[fn + "_idx" + str(idx)]["attrs"][0]["attr"][0]) + "_idx" + str(idx)
        assert len(res) > len(feature_conf)
        # new_res = self.lambda_feature_config(res)
        return res

    def lambda_feature_config(self, feature_config):
        key_index = 500
        key_sort = sorted(feature_config.keys())

        for key in key_sort:
            extractor = feature_config[key]
            assert len(extractor['attrs']) == 1
            try:
                old_model_key_type = hot_rerank_conf["network"]["parameters"][key]["attrs"][0]["key_type"]
                extractor['attrs'][0]['key_type'] = old_model_key_type
            except KeyError as e:
                extractor['attrs'][0]['key_type'] = key_index
                key_index += 1
        return feature_config

    def get_user_feature(self):
        return self.set_user_feature().keys()
    
    def get_extra_features_listwise(self):
        return self.set_extra_feature_listwise().keys()

    def set_feature_conf_pool(self):
        res = {}
        res.update(self.set_user_feature())
        res.update(self.set_photo_feature())
        res.update(self.set_extra_feature())
        # res.update(self.set_context_features())
        res.update(self.set_slide_feature())
        return res

    def set_embeding_hyper_parameter(self):
        embeding_hyper_parameter_conf = {}
        # "online_push_limit": 0,
        embeding_hyper_parameter_conf['default_online_push_limit'] = 5
        # type 默认是 1:EMBEDDING_PARAMETER, 其它类型有 2:LR_PARAMETER 3:THIRD_PATRY_DATA 5:flat list
        embeding_hyper_parameter_conf['default_type'] = 1
        embeding_hyper_parameter_conf['default_dim'] = 8
        embeding_hyper_parameter_conf['default_batch_num'] = 1
        embeding_hyper_parameter_conf['default_batch_decay'] = 0.99
        embeding_hyper_parameter_conf['default_move_length'] = 0.01

        # expire_second 默认是 -1, 不过期
        embeding_hyper_parameter_conf['default_expire_second'] = 30 * 3600 * 24
        embeding_hyper_parameter_conf['default_decay_rate'] = 0.9999
        embeding_hyper_parameter_conf['default_initial_lr'] = 0.05
        embeding_hyper_parameter_conf['default_initial_g2sum'] = 3
        return embeding_hyper_parameter_conf


    def set_extra_feature(self):
        return dict(
            valid_click_pos_label=numericExtractor("ValidClickPosLabel", dim=1),
            pvalid_click_weight=numericExtractor("pValidClickWeightV3", dim=1),
            # valid_wtd_pos_label=numericExtractor("ValidWTDPosLabel", dim=1),
            valid_like_pos_label=numericExtractor("ValidLikePosLabel", dim=1),
            valid_follow_pos_label=numericExtractor("ValidFollowPosLabel", dim=1),
            valid_profile_pos_label=numericExtractor("ValidProfilePosLabel", dim=1),
            valid_longview_pos_label=numericExtractor("ValidLongViewPosLabel", dim=1)
        )
    
    def set_slide_feature(self):
        return dict(
            slide_evtr_pos_label=numericExtractor("SlideEvtrPosLabel", dim=1),
            pvalid_slide_click_weight=numericExtractor("SlideEvtrPosWeight", dim=1),
            slide_click_pos_label=numericExtractor("slideClickPosLabel", dim=1),
            slide_interact_pos_label=numericExtractor("slideInteractPosLabel", dim=1),
            slide_longview_pos_label=numericExtractor("slideLongViewV2PosLabel", dim=1)
        )

    def set_user_feature(self):
        return dict(
            user_id=intExtractor('uId', dim=24, expire_second=86400 * 30),
            user_device_id=intExtractor('dId', dim=24, expire_second=86400 * 30),
            # user context
            user_gender=listExtractor(['uGender']),
            user_age=listExtractor(['uAge']),
            user_age_seg=intExtractor('uAgeSeg'),
            # user_risk_level=intExtractor('uRiskLevel'),
            # user_pure_consumer=intExtractor('uPureConcumer'),
            user_provinceId=intExtractor('uProvinceId'),
            user_cityId=intExtractor('uCityId'),
            # user_clientId=intExtractor('uClientId'),
            user_mod=intExtractor('uMod'),
            user_network=intExtractor('uNetwork'),
            user_request_provinceId=intExtractor('uRequstProvinceId'),
            user_request_cityId=intExtractor('uRequstCityId'),
            # user_request_poiType=intExtractor('uRequestPoiType'),

            # 2019.8.23 新增
            # user_request_town=intExtractor('uRequestTown'),
            # user_request_city_level=intExtractor('uRequestCityLevel'),
            # user_request_commuity_type=intExtractor('uRequestCommuityType'),
            # user_freq_provinceId=intExtractor('uFreqProvinceId'),
            # user_freq_cityId=intExtractor('uFreqCityId'),
            
            # 2019.11.21 新增
            user_realtime_click_list=listExtractor(['uRealtimeClickList'], dim=16, join_limit=200),
            user_realtime_like_list=listExtractor(['uRealtimeLikeList'], dim=16, join_limit=200),
            user_realtime_follow_list=listExtractor(['uRealtimeFollowList'], dim=16, join_limit=200),
            user_realtime_forward_list=listExtractor(['uRealtimeForwardList'], dim=16, join_limit=200),
            user_realtime_negative_list=listExtractor(['uRealtimeNegativeList'], dim=16, join_limit=200),
            user_likephoto_author_list=listExtractor(['uLikePhotoAuthorList'], dim=16, join_limit=200),
            user_followphoto_author_list=listExtractor(['uFollowPhotoAuthorList'], dim=16, join_limit=200),
            user_request_hour=intExtractor('uRequestHour'),
            user_request_weekday=intExtractor('uRequestWeekday'),
            # user_follow_cnt=intExtractor('uFollowCnt'),
        )


    def set_photo_feature(self):
        return dict(
            # id, 24*4
            photo_id=intExtractor('pId', dim=24, expire_second=86400 * 3),
            author_id=intExtractor('aId', dim=24, expire_second=86400 * 30),
            
            # mc, 8*5
            mc_pctr=floatDiscreteExtractor('pMcPctr', '1,0,1,2000,-1'),
            mc_pltr=floatDiscreteExtractor('pMcPltr', '0.2,0,1,2000,-1'),
            mc_pwtr=floatDiscreteExtractor('pMcPwtr', '0.2,0,1,2000,-1'),
            mc_plvtr=floatDiscreteExtractor('pMcPlvtr', '1,0,1,2000,-1'),
            mc_psvtr=floatDiscreteExtractor('pMcPsvtr', '1,0,1,2000,-1'),

            # emp 8*7
            emp_ctr=floatDiscreteExtractor('pEmpCtr', '1,0,1,1000,0'),
            emp_ltr=floatDiscreteExtractor('pEmpLtr', '0.3,0,1,1000,0'),
            emp_wtr=floatDiscreteExtractor('pEmpWtr', '0.3,0,1,1000,0'),
            emp_ftr=floatDiscreteExtractor('pEmpFtr', '0.1,0,1,1000,0'),
            emp_ptr=floatDiscreteExtractor('pEmpPtr', '0.5,0,1,1000,0'),
            emp_cmtr=floatDiscreteExtractor('pEmpCmtr', '0.3,0,1,1000,0'),
            emp_htr=floatDiscreteExtractor('pEmpHtr', '0.001,0,1,1000,0'),

            # photo_info, 8*16
            fans_count_low=floatDiscreteExtractor('pAuthorFansCount', '100,0,100,1,0'),
            fans_count_high=floatDiscreteExtractor('pAuthorFansCount', '10000,0,100,1,0'),
            photo_age_hour=intExtractor('pAgeHour'),
            photo_duration_sec=floatDiscreteExtractor('pDurationMs', '1000,0,180,1,0'),
            photo_upload_type=intExtractor('pUploadType'),
            photo_exp_show_low=floatDiscreteExtractor('pHotShow', '200,0,100,1,0'),
            photo_exp_show_high=floatDiscreteExtractor('pHotShow', '20000,0,100,1,0'),
            photo_exp_click_low=floatDiscreteExtractor('pHotClick', '10,0,100,1,0'),
            photo_exp_click_high=floatDiscreteExtractor('pHotClick', '5000,0,100,1,0'),
            photo_exp_like_low=floatDiscreteExtractor('pHotLike', '1,0,100,1,0'),
            photo_exp_like_high=floatDiscreteExtractor('pHotLike', '500,0,200,1,0'),
            photo_exp_follow_low=floatDiscreteExtractor('pHotFollow', '1,0,100,1,0'),
            photo_exp_follow_high=floatDiscreteExtractor('pHotFollow', '250,0,200,1,0'),
            photo_exp_hate=floatDiscreteExtractor('pHotHate', '1,0,1000,1,0'),
            photo_exp_report=floatDiscreteExtractor('pHotReport', '1,0,1000,1,0'),
            # photo_rrr=floatDiscreteExtractor('pHotRRR', '0.00001,0,100,1,0'),

            # context, 1 + 8*7 + 8 * 5 + 8 *13 + 1 + 8*5
            # photo context
            living=intExtractor('pHotLiving'),
            photo_exptag=intExtractor('pHotExptag'),
            photo_provinceId=intExtractor('pProvinceId'),
            photo_cityId=intExtractor('pCityId'),
            # photo_poiType=intExtractor('pPoiType'),
            photo_levelA=intExtractor('pLevelA'),
            photo_levelB=intExtractor('pLevelB'),
            photo_contentLevel=intExtractor('pContentLevel'),
            # 2019.8.23新增
            photo_avg_watchtime=floatDiscreteExtractor('pAvgWatchtime', '1000,0,180,1,0'),
            photo_author_age_seg=intExtractor('pAuthorAgeSeg'),
            photo_author_age_gender=intExtractor('pAuthorGender'),
            photo_muisc=intExtractor('pMusic'),
            photo_mod=intExtractor('pPhoneMod'),
            # xtr, 8*11
            pctr=floatDiscreteExtractor('pPctr', '1,0,1,10000,-1'),
            pltr=floatDiscreteExtractor('pPltr', '0.5,0,1,2000,-1'),
            pwtr=floatDiscreteExtractor('pPwtr', '0.2,0,1,2000,-1'),
            pftr=floatDiscreteExtractor('pPftr', '0.2,0,1,2000,-1'),
            phtr=floatDiscreteExtractor('pPhtr', '0.2,0,1,2000,-1'),
            plvtr=floatDiscreteExtractor('pPlvtr', '1,0,1,2000,-1'),
            psvtr=floatDiscreteExtractor('pPsvtr', '1,0,1,2000,-1'),
            pvtr=floatDiscreteExtractor('pPvtr', '0.1,0,1,2000,-1'),
            pptr=floatDiscreteExtractor('pPptr', '0.2,0,1,2000,-1'),
            pcmtr=floatDiscreteExtractor('pPcmtr', '0.2,0,1,2000,-1'),
            plivingctr=floatDiscreteExtractor('pPlivingctr', '0.1,0,1,2000,-1'),

            # pLiveCtrV2=floatDiscreteExtractor('pLiveCtrV2', '0.1,0,1,2000,-1'),

            pPevtr=floatDiscreteExtractor('pPevtr', '0.2,0,1,2000,-1'),
            # pPlivingwtr=floatDiscreteExtractor('pPlivingwtr', '0.2,0,1,2000,-1'),
            pPdctr=floatDiscreteExtractor('pPdctr', '0.2,0,1,2000,-1'),
            pPtagctr=floatDiscreteExtractor('pPtagctr', '0.2,0,1,2000,-1'),
            pPtagjoin=floatDiscreteExtractor('pPtagjoin', '0.2,0,1,2000,-1'),
            pPalltagctr=floatDiscreteExtractor('pPalltagctr', '0.2,0,1,2000,-1'),
            pPmfctr=floatDiscreteExtractor('pPmfctr', '0.2,0,1,2000,-1'),
            pPcmef=floatDiscreteExtractor('pPcmef', '0.2,0,1,2000,-1'),
            pPfrScore1=floatDiscreteExtractor('pPfrScore1', '1,0,1,2000,-1'),
            pPfrScore2=floatDiscreteExtractor('pPfrScore2', '180,0,1,2000,-1'),

            hetu_level_one_list=listExtractor(['pHetuTagLevel1Id'], dim=4, join_limit=2),
            hetu_level_two_list=listExtractor(['pHetuTagLevel2Id'], dim=4, join_limit=5),
            dnn_cluster_id=intExtractor('pDnnClusterId', dim=4),
        )
    
    def set_context_features(self):
        return dict(
            # xtr 64 * 18
            maxPctr=floatDiscreteExtractor('maxPctr_context', '1,0,1,10000,-1', dim=64),
            maxPltr=floatDiscreteExtractor('maxPltr_context', '0.5,0,1,2000,-1', dim=64),
            maxPwtr=floatDiscreteExtractor('maxPwtr_context', '0.2,0,1,2000,-1', dim=64),
            maxPftr=floatDiscreteExtractor('maxPftr_context', '0.2,0,1,2000,-1', dim=64),
            # maxPhtr=floatDiscreteExtractor('maxPhtr_context', '0.2,0,1,2000,-1', dim=64),
            maxPvtr=floatDiscreteExtractor('maxPvtr_context', '1,0,1,2000,-1', dim=64),
            maxPptr=floatDiscreteExtractor('maxPptr_context', '0.2,0,1,2000,-1', dim=64),
            maxPcmtr=floatDiscreteExtractor('maxPcmtr_context', '0.2,0,1,2000,-1', dim=64),
            maxPlivingtr=floatDiscreteExtractor('maxPlivingtr_context', '0.1,0,1,2000,-1', dim=64),
            
            avgPctr=floatDiscreteExtractor('avgPctr_context', '1,0,1,10000,-1', dim=64),
            avgPltr=floatDiscreteExtractor('avgPltr_context', '0.5,0,1,2000,-1', dim=64),
            avgPwtr=floatDiscreteExtractor('avgPwtr_context', '0.2,0,1,2000,-1', dim=64),
            avgPftr=floatDiscreteExtractor('avgPftr_context', '0.2,0,1,2000,-1', dim=64),
            # avgPhtr=floatDiscreteExtractor('avgPhtr_context', '0.2,0,1,2000,-1', dim=64),
            avgPvtr=floatDiscreteExtractor('avgPvtr_context', '1,0,1,2000,-1', dim=64),
            avgPptr=floatDiscreteExtractor('avgPptr_context', '0.2,0,1,2000,-1', dim=64),
            avgPcmtr=floatDiscreteExtractor('avgPcmtr_context', '0.2,0,1,2000,-1', dim=64),
            avgPlivingtr=floatDiscreteExtractor('avgPlivingtr_context', '0.1,0,1,2000,-1', dim=64),

        )

hotReRankFeatureconf = HotReRankFeatureConf()

from ranking import CommonModule

class AutoParamModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .if_("fullrank_enable_jarvis_param > 0") \
        .gen_common_attr_by_lua(
          attr_map = {
            "user_stat_redis_key" : "' ' .. tostring(_USER_ID_)" ,
            "user_app_redis_key" : "'UserAppCart_' .. tostring(_USER_ID_)"
          }
        ) \
        .get_common_attr_from_redis(
          cluster_name = "recoHotUserStatForLtr",
          redis_params = [
            {
              "redis_key": "{{user_stat_redis_key}}",
              "output_attr_name": "user_stat_xtr_attr_from_redis"
            }
          ]
        ) \
        .get_common_attr_from_redis(
          cluster_name = "recoExploreDegradeLeaf",
          redis_params = [
            {
              "redis_key": "{{user_app_redis_key}}",
              "output_attr_name": "user_app_attr_from_redis"
            }
          ]
        ) \
        .set_default_value(
          no_overwrite=True,
          common_attrs=[
            {
                "name": "fullrank_ctr_adjust_ratio_attr",
                "type": "float",
                "value": 1.0
            },
            {
                "name": "fullrank_ltr_adjust_ratio_attr",
                "type": "float",
                "value": 1.0
            },
            {
                "name": "fullrank_wtr_adjust_ratio_attr",
                "type": "float",
                "value": 1.0
            },
            {
                "name": "fullrank_ftr_adjust_ratio_attr",
                "type": "float",
                "value": 1.0
            },
            {
                "name": "fullrank_cltr_adjust_ratio_attr",
                "type": "float",
                "value": 1.0
            },
            {
                "name": "fullrank_ptr_adjust_ratio_attr",
                "type": "float",
                "value": 1.0
            },
            {
                "name": "fullrank_cmtr_adjust_ratio_attr",
                "type": "float",
                "value": 1.0
            },
            {
                "name": "fullrank_cmef_adjust_ratio_attr",
                "type": "float",
                "value": 1.0
            },
            {
                "name": "fullrank_epstr_adjust_ratio_attr",
                "type": "float",
                "value": 1.0
            },
            {
                "name": "fullrank_fr_score1_adjust_ratio_attr",
                "type": "float",
                "value": 1.0
            },
            {
                "name": "fullrank_fr_score2_adjust_ratio_attr",
                "type": "float",
                "value": 1.0
            },
            {
                "name": "fullrank_fetr_adjust_ratio_attr",
                "type": "float",
                "value": 1.0
            },
            {
                "name": "fullrank_fountain_eff_adjust_ratio_attr",
                "type": "float",
                "value": 1.0
            },
            {
                "name": "fullrank_duration_adjust_ratio_attr",
                "type": "float",
                "value": 1.0
            },
            {
                "name": "fullrank_l2r_score_adjust_ratio_attr",
                "type": "float",
                "value": 1.0
            },
            {
                "name": "fullrank_expected_score_adjust_ratio_attr",
                "type": "float",
                "value": 1.0
            },
            {
                "name": "rerank_ctr_adjust_ratio_attr",
                "type": "float",
                "value": 1.0
            },
            {
                "name": "rerank_ltr_adjust_ratio_attr",
                "type": "float",
                "value": 1.0
            },
            {
                "name": "rerank_wtr_adjust_ratio_attr",
                "type": "float",
                "value": 1.0
            },
            {
                "name": "rerank_ftr_adjust_ratio_attr",
                "type": "float",
                "value": 1.0
            },
            {
                "name": "rerank_cltr_adjust_ratio_attr",
                "type": "float",
                "value": 1.0
            },
            {
                "name": "rerank_ptr_adjust_ratio_attr",
                "type": "float",
                "value": 1.0
            },
            {
                "name": "rerank_cmtr_adjust_ratio_attr",
                "type": "float",
                "value": 1.0
            },
            {
                "name": "rerank_cmef_adjust_ratio_attr",
                "type": "float",
                "value": 1.0
            },
            {
                "name": "rerank_epstr_adjust_ratio_attr",
                "type": "float",
                "value": 1.0
            },
            {
                "name": "rerank_fr_score1_adjust_ratio_attr",
                "type": "float",
                "value": 1.0
            },
            {
                "name": "rerank_fr_score2_adjust_ratio_attr",
                "type": "float",
                "value": 1.0
            },
            {
                "name": "rerank_fetr_adjust_ratio_attr",
                "type": "float",
                "value": 1.0
            },
            {
                "name": "rerank_fountain_eff_adjust_ratio_attr",
                "type": "float",
                "value": 1.0
            },
            {
                "name": "rerank_duration_adjust_ratio_attr",
                "type": "float",
                "value": 1.0
            },
            {
                "name": "rerank_l2r_score_adjust_ratio_attr",
                "type": "float",
                "value": 1.0
            },
            {
                "name": "rerank_ensemble_score_adjust_ratio_attr",
                "type": "float",
                "value": 1.0
            }
          ]
        ) \
        .explore_sphinx_param_enrich(
          user_info_pb_name = "user_info_ptr",
          session_id_attr = "sessionId",
          user_stat_attr = "user_stat_xtr_attr_from_redis",
          user_app_attr = "user_app_attr_from_redis",
          request_based_jarvis_enabled=True,
          jarvis_kess_service="{{fullrank_param_jarvis_service}}",
          jarvis_model_name="{{fullrank_param_jarvis_model_name}}",
          app_name="explore",
          action_type="{{fullrank_param_jarvis_action_type}}",
          jarvis_time_out=100,
          use_app_cat=True,
          use_expxtr=True,
          item_attrs={
            "author_id": "author__id",
            "pevtr": "corr_pctr",
            "plvtr": "plvtr",
            "psvr": "psvr",
            "pvtr": "pvtr",
            "pltr": "pltr",
            "phtr": "phtr",
            "pwtr": "pwtr",
            "pftr": "pftr",
            "pptr": "pptr",
            "pclr": "pcltr",
            "pdtr": "pdtr",
            "pcmtr": "pcmtr",
            "pcmef": "pcmef",
            "pepstr": "pepstr",
            "pfr_score1": "pfr_score1",
            "pfr_score2": "pfr_score2",
            "pfetr": "pfetr",
            "pfountain_eff": "pfountain_eff",
            "duration_ms": "duration_ms",
            "hetu_level_one": "hetu_tag_level_info__hetu_level_one",
            "hetu_level_two": "hetu_tag_level_info__hetu_level_two",
            "hetu_level_three": "hetu_tag_level_info__hetu_level_three"
          },
          queues=[
            {
              "name": "adjust:rl_fullrank_fr_score1",
              "param_attr": "fullrank_fr_score1_adjust_ratio_attr"
            },
            {
              "name": "adjust:rl_fullrank_fr_score2",
              "param_attr": "fullrank_fr_score2_adjust_ratio_attr"
            },
            {
              "name": "adjust:rl_fullrank_ctr",
              "param_attr": "fullrank_ctr_adjust_ratio_attr"
            },
            {
              "name": "adjust:rl_fullrank_ltr",
              "param_attr": "fullrank_ltr_adjust_ratio_attr"
            },
            {
              "name": "adjust:rl_fullrank_wtr",
              "param_attr": "fullrank_wtr_adjust_ratio_attr"
            },
            {
              "name": "adjust:rl_fullrank_ftr",
              "param_attr": "fullrank_ftr_adjust_ratio_attr"
            },
            {
              "name": "adjust:rl_fullrank_cmtr",
              "param_attr": "fullrank_cmtr_adjust_ratio_attr"
            },
            {
              "name": "adjust:rl_fullrank_ptr",
              "param_attr": "fullrank_ptr_adjust_ratio_attr"
            },
            {
              "name": "adjust:rl_fullrank_cltr",
              "param_attr": "fullrank_cltr_adjust_ratio_attr"
            },
            {
              "name": "adjust:rl_fullrank_cmef",
              "param_attr": "fullrank_cmef_adjust_ratio_attr"
            },
            {
              "name": "adjust:rl_fullrank_epstr",
              "param_attr": "fullrank_epstr_adjust_ratio_attr"
            },
            {
              "name": "adjust:rl_fullrank_fetr",
              "param_attr": "fullrank_fetr_adjust_ratio_attr"
            },
            {
              "name": "adjust:rl_fullrank_fountain_eff",
              "param_attr": "fullrank_fountain_eff_adjust_ratio_attr"
            },
            {
              "name": "adjust:rl_fullrank_l2r_score",
              "param_attr": "fullrank_l2r_score_adjust_ratio_attr"
            },
            {
              "name": "adjust:rl_fullrank_duration",
              "param_attr": "fullrank_duration_adjust_ratio_attr"
            },
            {
              "name": "adjust:rl_fullrank_expected_score",
              "param_attr": "fullrank_expected_score_adjust_ratio_attr"
            },
            {
              "name": "adjust:rl_rerank_fr_score1",
              "param_attr": "rerank_fr_score1_adjust_ratio_attr"
            },
            {
              "name": "adjust:rl_rerank_fr_score2",
              "param_attr": "rerank_fr_score2_adjust_ratio_attr"
            },
            {
              "name": "adjust:rl_rerank_ctr",
              "param_attr": "rerank_ctr_adjust_ratio_attr"
            },
            {
              "name": "adjust:rl_rerank_ltr",
              "param_attr": "rerank_ltr_adjust_ratio_attr"
            },
            {
              "name": "adjust:rl_rerank_wtr",
              "param_attr": "rerank_wtr_adjust_ratio_attr"
            },
            {
              "name": "adjust:rl_rerank_ftr",
              "param_attr": "rerank_ftr_adjust_ratio_attr"
            },
            {
              "name": "adjust:rl_rerank_cmtr",
              "param_attr": "rerank_cmtr_adjust_ratio_attr"
            },
            {
              "name": "adjust:rl_rerank_ptr",
              "param_attr": "rerank_ptr_adjust_ratio_attr"
            },
            {
              "name": "adjust:rl_rerank_cltr",
              "param_attr": "rerank_cltr_adjust_ratio_attr"
            },
            {
              "name": "adjust:rl_rerank_cmef",
              "param_attr": "rerank_cmef_adjust_ratio_attr"
            },
            {
              "name": "adjust:rl_rerank_epstr",
              "param_attr": "rerank_epstr_adjust_ratio_attr"
            },
            {
              "name": "adjust:rl_rerank_fetr",
              "param_attr": "rerank_fetr_adjust_ratio_attr"
            },
            {
              "name": "adjust:rl_rerank_fountain_eff",
              "param_attr": "rerank_fountain_eff_adjust_ratio_attr"
            },
            {
              "name": "adjust:rl_rerank_l2r_score",
              "param_attr": "rerank_l2r_score_adjust_ratio_attr"
            },
            {
              "name": "adjust:rl_rerank_duration",
              "param_attr": "rerank_duration_adjust_ratio_attr"
            },
            {
              "name": "adjust:rl_rerank_ensemble_score",
              "param_attr": "rerank_ensemble_score_adjust_ratio_attr"
            },
          ]
        ) \
      .end_if_()

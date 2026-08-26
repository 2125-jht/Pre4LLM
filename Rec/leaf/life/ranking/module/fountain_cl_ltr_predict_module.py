from ranking import CommonModule
from ranking.fountain_ranking_features import user_features_v2, photo_features, photo_pxtr_features

class FountainClLtrPredictModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .delegate_enrich(
        kess_service = "{{fountain_cl_rank_predict_kess_service}}",
        recv_item_attrs = [
          {"name": "fountain_time", "as": "fullrank_cl_score"},
          {"name": "fountain_play_time", "as": "fullrank_cl_play_time"}
        ],
        timeout_ms = 150,
        send_item_attrs = [feature["name"] for feature in photo_features if feature["name"] not in photo_pxtr_features],
        send_common_attrs = user_features_v2,
        request_type = "kai_predict",
        skip = "{{skip_fountain_cl_rank_predict_kess_service}}",
        partition_size = "{{fountain_cl_rank_predict_partition_size}}",
      ) \
      .get_kconf_params(
        skip = "{{skip_fullrank_cl_score_with_duration}}",
        kconf_configs = [{
          "kconf_key": "{{fullrank_cl_score_finish_threshold_kconf}}",
          "value_type": "list_double",
          "defult_value": [],
          "export_common_attr": "duration_finish_threshold"
        }]
      ) \
      .enrich_attr_by_lua(
        skip = "{{skip_fullrank_cl_score_with_duration}}",
        import_common_attr = [
          "fountain_fullrank_cl_time_score_weight",
          "fountain_fullrank_cl_duration_weight",
          "fountain_fullrank_cl_click_weight",
          "fountain_fullrank_cl_duration_seg",
          "fountain_fullrank_cl_duration_max",
          "fountain_fullrank_cl_enable_threshold_bias",
          "fountain_fullrank_cl_enable_threshold_bias_v2",
          "fountain_fullrank_cl_enable_duration",
          "fountain_fullrank_cl_threshold_weight",
          "duration_finish_threshold"
        ],
        import_item_attr = [
          "fullrank_cl_score",
          "duration_ms",
          "fullrank_sim_pevtr"
        ],
        export_item_attr = [
          "fullrank_cl_tran_score"
        ],
        function_for_item = "fullrank_cal_cl_score",
        lua_script_file = "life/ranking/lua/module/fountain_ranking_score__fullrank_cal_cl_score.lua",
      )
  
  def post_process(self) -> None:
    self.flow \
      .log_debug_info(
        item_attrs = [
          "fullrank_cl_score",
          "fullrank_cl_play_time"
        ],
        for_debug_request_only = True
      )


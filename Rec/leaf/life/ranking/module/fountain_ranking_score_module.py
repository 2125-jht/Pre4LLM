from ranking import CommonModule
from ranking.fountain_ranking_features import user_features_v2, photo_features, photo_pxtr_features

class FountainRankingScoreModule(CommonModule):
  def __init__(self, name: str) -> None:
      super().__init__(name)

  def process(self) -> None:
    self.flow \
      .enrich_attr_by_lua(
        import_item_attr = ["reason"],
        export_item_attr = ["reason_str"],
        function_for_item = "trans_reason_to_str",
        lua_script_file = "life/ranking/lua/module/fountain_ranking_score__trans_reason_to_str.lua"
      ) \
      .build_protobuf(
        class_name = "ks.reco.RecoPhotoInfo",
        inputs = [
          {
            "attr_name": "cascade_pctr",
            "path": "context_info.cascade_pctr"
          },
          {
            "attr_name": "cascade_pltr",
            "path": "context_info.cascade_pltr"
          },
          {
            "attr_name": "cascade_pwtr",
            "path": "context_info.cascade_pwtr"
          },
          {
            "attr_name": "cascade_plvtr",
            "path": "context_info.cascade_plvtr"
          },
          {
            "attr_name": "cascade_psvtr",
            "path": "context_info.cascade_psvr"
          },
          {
            "attr_name": "item_id",
            "path": "ar_result.pid"
          },
          {
            "attr_name": "content_safety_level_with_namespace__level_hot_online",
            "path": "ar_result.content_safety_level"
          },
          {
            "attr_name": "reason_str",
            "path": "reason"
          },
          {
            "attr_name": "cascade_pftr",
            "path": "context_info.cascade_pftr"
          },
          {
            "attr_name": "cascade_pepstr",
            "path": "context_info.cascade_pepstr"
          },
          {
            "attr_name": "cascade_pcmtr",
            "path": "context_info.cascade_pcmtr"
          },
          {
            "attr_name": "cascade_phtr",
            "path": "context_info.cascade_phtr"
          },
          {
            "attr_name": "cascade_pctr_index",
            "path": "cascade_pctr_index"
          },
          {
            "attr_name": "cascade_plvtr_index",
            "path": "cascade_plvtr_index"
          },
          {
            "attr_name": "cascade_pvtr_index",
            "path": "cascade_pvtr_index"
          },
          {
            "attr_name": "cascade_pltr_index",
            "path": "cascade_pltr_index"
          },
          {
            "attr_name": "cascade_pftr_index",
            "path": "cascade_pftr_index"
          },
          {
            "attr_name": "cascade_pwtr_index",
            "path": "cascade_pwtr_index"
          },
          {
            "attr_name": "cascade_pesptr_index",
            "path": "cascade_pesptr_index"
          },
          {
            "attr_name": "cascade_psvr_index",
            "path": "cascade_psvr_index"
          }
        ],
        is_common_attr = False,
        output_attr = "reco_photo_info",
      ) \
      .delegate_enrich(
        kess_service = "{{xlife_fountain_fullrank_sim_predict_kess_service}}",
        partition_size = "{{fountain_fullrank_sim_predict_partition_size}}",
        recv_item_attrs = [
          { "name": "ctr", "as": "fullrank_sim_out_pctr" },
          { "name": "ltr", "as": "fullrank_sim_pltr" },
          { "name": "wtr", "as": "fullrank_sim_pwtr" },
          { "name": "ftr", "as": "fullrank_sim_pftr" },
          { "name": "svr", "as": "fullrank_sim_psvr" },
          { "name": "lvtr", "as": "fullrank_sim_plvtr" },
          { "name": "cmtr", "as": "fullrank_sim_pcmtr" },
          { "name": "ptr", "as": "fullrank_sim_pptr" },
          { "name": "cmef", "as": "fullrank_sim_pcmef" },
          { "name": "htr", "as": "fullrank_sim_phtr" },
          { "name": "evtr", "as": "fullrank_sim_pevtr" },
          { "name": "vtr", "as": "fullrank_sim_pvtr" },
          { "name": "wtd_playtime", "as": "fullrank_sim_pwtd_playtime" },
          { "name": "epstr", "as": "fullrank_sim_pepstr" },
          { "name": "fintr", "as": "fullrank_sim_pfintr" },
          { "name": "cltr", "as": "fullrank_sim_pcltr" },
          { "name": "cpr", "as": "fullrank_sim_pcpr" },
          { "name": "wtd_v2_playtime", "as": "fullrank_sim_pwtd_v2_playtime" },
          { "name": "playtime_finish", "as": "fullrank_sim_playtime_finish" },
          { "name": "ordinal_playtime", "as": "fullrank_sim_ordinal_playtime" },
          # 左滑进入个人页
          { "name": "lstr", "as": "fullrank_sim_lstr" },
          { "name": "lsst", "as": "fullrank_sim_lsst" },
          { "name": "swptr", "as": "fullrank_ori_pswptr" },
          { "name": "fountain_evtr_v2", "as": "fullrank_detail_new_pevtr_v2" },
          { "name":"evtr_playtime", "as":"fullrank_sim_evtr_playtime" },
          { "name":"evtr_duration", "as":"fullrank_sim_evtr_duration" },
          { "name":"wtd_duration_score", "as":"fullrank_sim_wtd_duration" },
        ],
        request_type = "{{fountain_fullrank_sim_predict_request_type}}",
        send_common_attrs = [
          { "name": "userInfo", "as": "user_info_str" },
          { "name": "featureSourcePId", "as": "source_photo_id"  },
          { "name": "page", "as": "page_common"  },
        ],
        send_item_attrs = [
          { "name": "reco_photo_info", "as": "reco_photo_info_str" },
        ],
      ) \
      .copy_attr(
        attrs=[
          {"from_item": "fullrank_sim_out_pctr", "to_item": "pctr"},
          {"from_item": "fullrank_sim_pltr", "to_item": "pltr"},
          {"from_item": "fullrank_sim_pwtr", "to_item": "pwtr"},
          {"from_item": "fullrank_sim_pftr", "to_item": "pftr"},
          {"from_item": "fullrank_sim_psvr", "to_item": "psvtr"},
          {"from_item": "fullrank_sim_pcmtr", "to_item": "pcmtr"},
          {"from_item": "fullrank_sim_pevtr", "to_item": "pevtr"},
          {"from_item": "fullrank_sim_phtr", "to_item": "phtr"},
          {"from_item": "fullrank_sim_pcmef", "to_item": "pcmef"}
        ]
      ) \
      .enrich_attr_by_lua(
        import_item_attr = [
          "fullrank_sim_pevtr",
          "fullrank_sim_pltr",
          "fullrank_sim_pwtr",
          "fullrank_sim_pftr",
          "fullrank_sim_plvtr",
          "fullrank_sim_pvtr",
          "fullrank_sim_out_pctr",
          "fullrank_sim_pcmtr",
          "fullrank_sim_pcmef",
          "fullrank_sim_pptr",
          "fullrank_sim_pepstr",
          "fullrank_sim_phtr",
          "fullrank_sim_lstr",
          "fullrank_sim_pcltr",
          "fullrank_sim_pfintr"
        ],
        export_item_attr = [
          "fullrank_detail_pctr",
          "fullrank_detail_pltr",
          "fullrank_detail_pwtr",
          "fullrank_detail_pftr",
          "fullrank_detail_plvtr",
          "fullrank_detail_pvtr",
          "fullrank_detail_psvr",
          "fullrank_detail_pcmtr",
          "fullrank_detail_pcmef",
          "fullrank_detail_pptr",
          "fullrank_detail_pepstr",
          "fullrank_detail_phtr",
          "fullrank_final_lstr",
          "fullrank_sim_click_score",
          "fullrank_sim_like_score",
          "fullrank_sim_follow_score",
          "fullrank_sim_pcltr",
          "fullrank_detail_pwtd"
        ],
        function_for_item = "fullrank_trans_pxtr",
        lua_script_file = "life/ranking/lua/module/fountain_ranking_score__fullrank_trans_pxtr.lua",
      ) \
      .set_attr_value(
        no_overwrite=True,
        item_attrs = [
          {
            "name": "fullrank_distill_rerank_score",
            "type": "double",
            "value": 0.0
          }
        ]
      )

  def post_process(self) -> None:
    self.flow \
      .log_debug_info(
        item_attrs = [
          "fullrank_sim_pwtd_playtime",
          "fullrank_detail_phtr",
          "pctr",
          "pltr",
          "pwtr",
          "pftr",
          "psvtr",
          "pcmtr",
          "pevtr",
          "phtr",
          "pcmef"
        ],
        for_debug_request_only = True,
        item_num_limit = 10,
      )

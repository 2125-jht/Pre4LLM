from ranking import CommonModule

class RankingProducePhotoPredictModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .if_("enable_explore_rank_produce_video_predict == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "enable_explore_rank_produce_need_divuser",
            "explore_ranking_produce_real_show_photo_recent_hours",
            "explore_ranking_produce_his_zhongcao_threholds",
            "explore_ranking_produce_his_magic_face_threholds",
            "user_info_ptr",
            { "name": "enable_explore_rank_produce_need_divuser_v2", "as": "rank_produce_need_divuser_v2" },
            { "name": "uGamoraUploadDayNum30d", "as": "gamora_upload_day_num_30d" },
            { "name": "uNebulaUploadDayNum30d", "as": "nebula_upload_day_num_30d" }
          ],
          export_common_attr = [
            { "name": "ranking_need_produce_flag", "as": "ranking_need_produce_model_flag" }
          ],
          function_name = "JudgeNeedProduceModel",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .if_("ranking_need_produce_model_flag > 0") \
          .delegate_enrich(
            kess_service = "{{explore_rank_produce_video_service}}",
            send_common_attrs = [
              { "name": "userInfo", "as": "user_info_str" }
            ],
            send_item_attrs = [
              { "name": "live_photo_info__is_living", "as": "living" },
              { "name": "reco_photo_info_str", "as": "reco_photo_info_str" }
            ],
            recv_item_attrs = [
              { "name": "mtctr", "as": "produce_mtctr" },
              { "name": "twhtr", "as": "produce_twhtr" },
              { "name": "mfctr", "as": "produce_mfctr" },
              { "name": "mtcotr", "as": "produce_mtcotr" },
              { "name": "mtjtr", "as": "produce_mtjtr" },
              { "name": "mtm1", "as": "produce_mtm1" },
              { "name": "uploadw", "as": "produce_uploadw" },
              { "name": "uploads", "as": "produce_uploads" },
              { "name": "consuv", "as": "produce_consuv" },
              { "name": "consuv_v2", "as": "produce_consuv_v2"},
              { "name": "consuv_public", "as": "produce_consuv_public" },
            ],
            timeout_ms = "{{explore_rank_produce_video_predict_timeout_ms}}",
            request_type = "{{explore_rank_produce_video_request_type}}",
            partition_size = "{{explore_rank_produce_video_partition_size}}",
            for_predict = True
          ) \
        .end_() \
      .end_()

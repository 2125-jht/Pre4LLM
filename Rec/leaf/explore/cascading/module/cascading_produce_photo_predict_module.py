from cascading import CommonModule

class CascadingProducePhotoPredictModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .if_("enable_explore_cascading_produce_photo_predict == 1", to_be_delete = "date=2024-05-29;committer=liuyipeng03") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            { "name": "enable_explore_cascade_produce_need_divuser", "as": "enable_explore_rank_produce_need_divuser" },
            { "name": "explore_cascade_produce_real_show_photo_recent_hours", "as": "explore_ranking_produce_real_show_photo_recent_hours" },
            { "name": "explore_cascade_produce_his_zhongcao_threholds", "as": "explore_ranking_produce_his_zhongcao_threholds" },
            { "name": "explore_cascade_produce_his_magic_face_threholds", "as": "explore_ranking_produce_his_magic_face_threholds" },
            { "name": "user_info_ptr", "as": "user_info_ptr" },
            { "name": "enable_explore_cascade_produce_need_divuser_v2", "as": "rank_produce_need_divuser_v2" },
            { "name": "uGamoraUploadDayNum30d", "as": "gamora_upload_day_num_30d" },
            { "name": "uNebulaUploadDayNum30d", "as": "nebula_upload_day_num_30d" }
          ],
          export_common_attr = [
            { "name": "ranking_need_produce_flag", "as": "explore_cascade_need_produce_model_flag" }
          ],
          function_name = "JudgeNeedProduceModel",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .if_("explore_cascade_need_produce_model_flag > 0", to_be_delete = "date=2024-05-29;committer=liuyipeng03") \
          .delegate_enrich(
            kess_service = "{{explore_cascading_produce_photo_double_tower_service}}",
            send_common_attrs = [
              { "name": "userInfo", "as": "user_info_str" },
            ],
            recv_item_attrs = [
              { "name": "mtctr", "as": "produce_cascade_mtctr" },
              { "name": "twhtr", "as": "produce_cascade_twhtr" },
              { "name": "mtcotr", "as": "produce_cascade_mtcotr" },
              { "name": "mtjtr", "as": "produce_cascade_mtjtr" },
              { "name": "kym", "as": "produce_cascade_kym" },
              { "name": "csti", "as": "produce_cascade_csti" },
              { "name": "sjctr", "as": "produce_cascade_sjctr" },
            ],
            timeout_ms = "{{explore_cascading_produce_photo_predict_timeout_ms}}",
            request_type = "{{explore_cascading_produce_photo_double_tower_request_type}}",
            partition_size = "{{explore_cascading_produce_photo_double_tower_partition_size}}",
            infer_output_type = 2,
            for_predict = True
          ) \
        .end_() \
      .end_()
      
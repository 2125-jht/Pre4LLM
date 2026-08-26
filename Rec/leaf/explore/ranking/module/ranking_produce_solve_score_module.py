from ranking import CommonModule

class RankingProduceSolveScoreModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .if_("enable_explore_rank_produce_video_predict == 1", to_be_delete = "date=2024-05-29;committer=liuyipeng03") \
        .if_("ranking_need_produce_model_flag > 0", to_be_delete = "date=2024-05-29;committer=liuyipeng03") \
          .enrich_attr_by_light_function(
            import_item_attr = [
              { "name": "produce_uploadw", "as": "uploadw_input" },
              { "name": "produce_uploads", "as": "uploads_input" },
              { "name": "produce_consuv", "as": "consuv_input" },
              { "name": "produce_consuv_v2", "as": "consuv_v2_input"},
              { "name": "produce_consuv_public", "as": "consuv_public_input" },
            ],
            export_item_attr = [
              { "name": "produce_upload_sum_out", "as": "produce_upload_sum_score" },
              { "name": "produce_consuv_sum_out", "as": "produce_consuv_sum_score" },
            ],
            function_name = "CalProduceSocre",
            class_name = "ExploreLightFunctionSetV2",
          ) \
        .end_() \
      .end_()

  def post_process(self) -> None:
    self.flow \
      .log_debug_info(
        item_attrs = [
          "produce_mtctr",
          "produce_twhtr",
          "produce_mfctr",
          "produce_mtcotr",
          "produce_mtjtr",
          "produce_mtm1",
          "produce_uploadw",
          "produce_uploads",
          "produce_consuv",
          "produce_consuv_v2",
          "produce_consuv_public",
          "produce_upload_sum_score",
          "produce_consuv_sum_score"
        ],
        common_attrs = [
          "enable_explore_rank_produce_need_divuser",
          "explore_ranking_produce_real_show_photo_recent_hours",
          "explore_ranking_produce_his_zhongcao_threholds",
          "explore_ranking_produce_his_magic_face_threholds",
          "ranking_need_produce_model_flag",
          "enable_explore_rank_produce_video_predict"
        ],
        for_debug_request_only = True,
      )

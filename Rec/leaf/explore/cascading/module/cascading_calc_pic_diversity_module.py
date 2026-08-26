from cascading import CommonModule

class CascadingCalcPicDiversityModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .if_("enable_explore_mc_calc_pic_diversity == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "user_info_ptr",
            {"name": "uStandardRealShowPicAllIdList", "as": "realshow_pic_ids"},
            {"name": "explore_pic_mc_diversity_history_num", "as": "history_num"},
            {"name": "explore_pic_mc_diversity_time_gap_min", "as": "time_gap_min"},
            {"name": "explore_pic_mc_diversity_enable_action", "as": "enable_action"},
            {"name": "explore_pic_mc_calc_negative_history_pic_ids_by_ddp", "as": "calc_by_ddp"},
            {"name": "uStandardExploreRealshowPhotoIdList", "as": "explore_realshow_ids"},
            {"name": "uStandardExploreRealshowTimestampList", "as": "explore_realshow_timestamps"},
            {"name": "uStandardExploreRealshowLabelList", "as": "explore_realshow_labels"},
          ],
          export_common_attr = [
            "history_pic_ids",
            "embedding_source_pic_ids",
          ],
          function_name = "GetEmbeddingSourcePicIds",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {"is_picture": 1}
        ) \
        .if_("explore_pic_mc_diversity_enable_hot_mc_emb > 0", to_be_delete = "date=2024-05-29;committer=zhuwenyong") \
          .get_remote_embedding_lite(
            kess_service = "{{explore_pic_mc_diversity_embedding_service_name}}",
            shard_num = 8,
            timeout_ms = 20,
            id_converter = {"type_name": "kuibaEmbeddingIdConverter"},
            input_attr_name = "embedding_source_pic_ids",
            output_attr_name = "pic_embeddings",
            query_source_type = "common_attr",
            size = 128,
            client_side_shard = True
          ) \
          .explore_get_embedding_map_enricher(
            embedding_list_attr = "pic_embeddings",
            source_pids_list_attr = "embedding_source_pic_ids",
            dim_size = 128,
            export_common_attr = "pic_pid_embedding_map",
          ) \
          .explore_picture_diversity_enricher(
            export_item_attr = "pic_diversity_mgs_score",
            history_pic_ids_attr = "history_pic_ids",
            pid_embedding_common_attr = "pic_pid_embedding_map",
            dim_size = 128,
            dpp_diversity_mgs_topk = "{{explore_pic_mc_diversity_history_mgs_topk}}",
            enable_dpp_diversity_mgs_ratio = "{{explore_pic_mc_diversity_enable_mgs_ratio}}",
            dpp_diversity_mgs_ratio = "{{explore_pic_mc_diversity_mgs_ratio}}",
            calc_diversity_method = "{{explore_pic_mc_diversity_calc_diversity_method}}",
            enable_mgs_his_cnt_less_topk = "{{explore_pic_mc_diversity_enable_mgs_his_cnt_less_topk}}",
            need_normalize_embed = "{{explore_pic_mc_diversity_need_normalize_embed}}",
            target_item = {"is_picture": 1}
          ) \
        .else_() \
          .get_remote_embedding_lite(
            kess_service = "{{explore_pic_mc_diversity_embedding_service_name}}",
            shard_num = 4,
            id_converter = {"type_name": "kuibaEmbeddingIdConverter"},
            input_attr_name = "embedding_source_pic_ids",
            output_attr_name = "pic_embeddings",
            query_source_type = "common_attr",
            size = 64,
            client_side_shard = True
          ) \
          .explore_get_embedding_map_enricher(
            embedding_list_attr = "pic_embeddings",
            source_pids_list_attr = "embedding_source_pic_ids",
            dim_size = 64,
            export_common_attr = "pic_pid_embedding_map",
          ) \
          .explore_picture_diversity_enricher(
            export_item_attr = "pic_diversity_mgs_score",
            history_pic_ids_attr = "history_pic_ids",
            pid_embedding_common_attr = "pic_pid_embedding_map",
            dim_size = 64,
            dpp_diversity_mgs_topk = "{{explore_pic_mc_diversity_history_mgs_topk}}",
            enable_dpp_diversity_mgs_ratio = "{{explore_pic_mc_diversity_enable_mgs_ratio}}",
            dpp_diversity_mgs_ratio = "{{explore_pic_mc_diversity_mgs_ratio}}",
            calc_diversity_method = "{{explore_pic_mc_diversity_calc_diversity_method}}",
            enable_mgs_his_cnt_less_topk = "{{explore_pic_mc_diversity_enable_mgs_his_cnt_less_topk}}",
            need_normalize_embed = "{{explore_pic_mc_diversity_need_normalize_embed}}",
            target_item = {"is_picture": 1}
          ) \
        .end_() \
      .end_() \
      .if_("enable_explore_mc_calc_vid2pic_sim == 1") \
        .split_string(
          input_common_attr = "explore_vid2pic_sim_target_hetu_str",
          output_common_attr = "explore_vid2pic_sim_target_hetu_list",
          delimiters = ",",
          trim_spaces = True,
          skip_empty_tokens = True,
          parse_to_int = True
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "user_info_ptr",
            {"name": "uStandardRealShowPicAllIdList", "as": "realshow_pic_ids"},
            {"name": "explore_vid2pic_sim_play_cnt", "as": "play_cnt"},
            {"name": "explore_vid2pic_sim_interact_cnt", "as": "interact_cnt"},
            {"name": "explore_vid2pic_sim_like_cnt", "as": "like_cnt"},
            {"name": "explore_vid2pic_sim_follow_cnt", "as": "follow_cnt"},
            {"name": "explore_vid2pic_sim_comment_cnt", "as": "comment_cnt"},
            {"name": "explore_vid2pic_sim_collect_cnt", "as": "collect_cnt"},
            {"name": "explore_vid2pic_sim_forward_cnt", "as": "forward_cnt"},
            {"name": "explore_vid2pic_sim_download_cnt", "as": "download_cnt"},
            {"name": "explore_vid2pic_sim_search_click_cnt", "as": "search_click_cnt"},
            {"name": "explore_vid2pic_sim_enable_target_hetu", "as": "enable_target_hetu"},
            {"name": "explore_vid2pic_sim_target_hetu_list", "as": "target_hetu_list"},
            {"name": "explore_vid2pic_sim_enable_pic", "as": "enable_pic"},
            {"name": "explore_vid2pic_sim_time_window", "as": "time_window"},
          ],
          export_common_attr = [
            "history_video_pids",
          ],
          function_name = "GetEmbeddingSourcePidsForVid2Pic",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {"is_picture": 1}
        ) \
        .pack_item_attr(
          item_source = {
            "reco_results": True
          },
          mappings = [{
            "from_item_attr": "photo_id",
            "to_common_attr": "candicate_picture_pids",
            "aggregator": "concat"
          }],
          target_item = {"is_picture": 1}
        ) \
        .pack_common_attr(
          input_common_attrs = [
            "history_video_pids",
            "candicate_picture_pids"
          ],
          output_common_attr = "vid2pic_embedding_source_pids",
        ) \
        .get_remote_embedding_lite(
          kess_service = "{{explore_vid2pic_embedding_service_name}}",
          shard_num = 4,
          id_converter = {"type_name": "kuibaEmbeddingIdConverter"},
          input_attr_name = "vid2pic_embedding_source_pids",
          output_attr_name = "vid2pic_embeddings",
          query_source_type = "common_attr",
          size = 64,
          client_side_shard = True
        ) \
        .explore_get_embedding_map_enricher(
          embedding_list_attr = "vid2pic_embeddings",
          source_pids_list_attr = "vid2pic_embedding_source_pids",
          dim_size = 64,
          export_common_attr = "vid2pic_embedding_map",
        ) \
        .explore_picture_diversity_enricher(
          export_item_attr = "vid2pic_sim_score",
          history_pic_ids_attr = "history_video_pids",
          pid_embedding_common_attr = "vid2pic_embedding_map",
          dim_size = 64,
          dpp_diversity_mgs_topk = "{{explore_vid2pic_sim_topk}}",
          enable_dpp_diversity_mgs_ratio = "{{explore_vid2pic_sim_enable_top_ratio}}",
          dpp_diversity_mgs_ratio = "{{explore_vid2pic_sim_top_ratio}}",
          calc_diversity_method = 2,
          need_normalize_embed = "{{explore_vid2pic_sim_need_normalize_embed}}",
          target_item = {"is_picture": 1}
        ) \
      .end_()

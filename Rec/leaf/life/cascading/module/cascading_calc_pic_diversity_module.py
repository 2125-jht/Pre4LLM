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
          ],
          export_common_attr = [
            "history_pic_ids",
            "embedding_source_pic_ids",
          ],
          function_name = "GetEmbeddingSourcePicIdsLife",
          class_name = "ExploreLifeLightFunctionSet",
          target_item = {"is_picture": 1}
        ) \
        .if_("explore_pic_mc_diversity_enable_hot_mc_emb > 0") \
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
      .end_()
from retrieval import CommonModule

class SplashEntryPhotoModule(CommonModule):
  def __init__(self, 
  name: str) -> None:
    super().__init__(name)

  def process(self):
    self.flow \
      .pack_item_attr(
        item_source = {
          "reco_results": False,
          "common_attr": ["featureSourcePId"],
        },
        mappings = [
          {
            "aggregator": "copy",
            "from_item_attr": "hetu_tag_level_info_v2__hetu_level_one",
            "to_common_attr": "source_hetu_level_one_v2_original",
          },
          {
            "aggregator": "copy",
            "from_item_attr": "hetu_tag_level_info_v2__hetu_level_two",
            "to_common_attr": "source_hetu_level_two_v2_original",
          },
          {
            "aggregator": "copy",
            "from_item_attr": "hetu_tag_level_info_v2__hetu_level_three",
            "to_common_attr": "source_hetu_level_three_v2_original",
          },
          {
            "aggregator": "copy",
            "from_item_attr": "hetu_tag_level_info_v2__hetu_level_four",
            "to_common_attr": "source_hetu_level_four_v2_original",
          },
          {
            "aggregator": "copy",
            "from_item_attr": "hetu_tag_level_info_v2__hetu_tag",
            "to_common_attr": "source_hetu_tag_v2_original",
          },
          {
            "aggregator": "copy",
            "from_item_attr": "hetu_tag_level_info_v2__hetu_face_id",
            "to_common_attr": "source_hetu_face_id_v2_original",
          },
          {
            "aggregator": "copy",
            "from_item_attr": "hetu_tag_level_info_v2__hetu_cluster_id",
            "to_common_attr": "source_hetu_cluster_id_v2_original",
          },
          {
            "aggregator": "copy",
            "from_item_attr": "hetu_tag_level_info__hetu_level_one",
            "to_common_attr": "source_hetu_level_one",
          },
          {
            "aggregator": "copy",
            "from_item_attr": "hetu_tag_level_info__hetu_level_two",
            "to_common_attr": "source_hetu_level_two",
          },
          {
            "aggregator": "copy",
            "from_item_attr": "hetu_tag_level_info__hetu_level_three",
            "to_common_attr": "source_hetu_level_three",
          },
          {
            "aggregator": "copy",
            "from_item_attr": "hetu_tag_level_info__hetu_level_four",
            "to_common_attr": "source_hetu_level_four",
          },
          {
            "aggregator": "copy",
            "from_item_attr": "hetu_tag_level_info__hetu_level_five",
            "to_common_attr": "source_hetu_level_five",
          },
          {
            "aggregator": "copy",
            "from_item_attr": "hetu_tag_level_info__hetu_tag",
            "to_common_attr": "source_hetu_tag_level_info_hetu_tag",
          },
          {
            "aggregator": "copy",
            "from_item_attr": "hetu_tag_level_info__hetu_face_id",
            "to_common_attr": "source_hetu_face_ids",
          },
          {
            "aggregator": "copy",
            "from_item_attr": "hetu_tag_level_info__hetu_cluster_id",
            "to_common_attr": "source_hetu_cluster_ids",
          },
          {
            "aggregator": "copy",
            "from_item_attr": "author__category_detail__third_level_id",
            "to_common_attr": "source_author_third_level_id",
          },
          {
            "aggregator": "copy",
            "from_item_attr": "mmu_content_ids_3",
            "to_common_attr": "source_mmu_content_ids_3",
          },
          {
            "aggregator": "copy",
            "from_item_attr": "mmu_content_ids_8",
            "to_common_attr": "source_mmu_content_ids_8",
          },
          {
            "aggregator": "copy",
            "from_item_attr": "mmu_content_ids_10",
            "to_common_attr": "source_mmu_content_ids_10",
          },
          {
            "aggregator": "copy",
            "from_item_attr": "mmu_content_ids_15",
            "to_common_attr": "source_mmu_content_ids_15",
          },
          {
            "aggregator": "copy",
            "from_item_attr": "mmu_content_ids_16",
            "to_common_attr": "source_mmu_content_ids_16",
          },
          {
            "aggregator": "copy",
            "from_item_attr": "mmu_content_ids_17",
            "to_common_attr": "source_mmu_content_ids_17",
          },
          {
            "aggregator": "copy",
            "from_item_attr": "hetu_sim_cluster_id",
            "to_common_attr": "source_hetu_sim_cluster_id",
          },
          {
            "aggregator": "copy",
            "from_item_attr": "user_hash_tag_id",
            "to_common_attr": "source_user_hash_tag_id",
          },
          {
            "aggregator": "copy",
            "from_item_attr": "author_circle_v2",
            "to_common_attr": "source_author_circle_v2",
          },
        ]
      ) \
      .log_debug_info(
        common_attrs = ["source_hetu_level_five"]
      ) \
      .enrich_attr_by_lua(
        import_common_attr = [
          "photoTagBucket",
          "featureUId",
          "topSubdivisionHetuBucket",
          "skip_fountain_top_subdivision_nn_retrieval_tag_splash",
          "fountain_retrieval_skip_top_subdivision_nn_retrieval",
          "skip_fountain_reco_emb_hetu_retrieval_splash",
          "topSubdivisionBucket",
          "featureSourcePId",
          "fountainHetuTagBucket",
          "fountain_skip_reco_emb_u2i_retr_splash",
          "fountain_skip_gcse_u2i_retrieval_splash",
          "source_hetu_level_one_v2_original",
          "source_hetu_level_two_v2_original",
          "source_hetu_level_three_v2_original",
          "source_hetu_level_four_v2_original",
          "source_hetu_tag_v2_original",
          "source_hetu_face_id_v2_original",
          "source_hetu_cluster_id_v2_original",
          "sourceMovieIp",
          "fountain_enable_first_page_skip_u2i_retrieval",
          "fountain_skip_ip2tag2ip_retr_splash",
          "fountain_enable_ip2tag2ip_retr_opt",
          "fountain_ip2tag2ip_retr_movie2movie_level",
          "currentTimeMs",
          "skip_fountain_icf_splash_retr",
          "skip_fountain_icf_splash_retr_mobile",
        ],
        export_common_attr = [
          "skip_fountain_top_subdivision_nn_retrieval_tag_splash",
          "fountain_retrieval_skip_top_subdivision_nn_retrieval",
          "skip_fountain_reco_emb_hetu_retrieval_splash",
          "fountain_swing_retr_redis_key",
          "fountain_skip_reco_emb_u2i_retr_splash",
          "fountain_skip_gcse_u2i_retrieval_splash",
          "source_hetu_level_one_v2",
          "source_hetu_level_two_v2",
          "source_hetu_level_three_v2",
          "source_hetu_level_four_v2",
          "source_hetu_tag_v2",
          "source_hetu_face_id_v2",
          "source_hetu_cluster_id_v2",
          "source_movie_related_ips_key",
          "source_movie_ip_extends_key",
          "fountain_skip_ip2tag2ip_retr_splash",
          "fountain_enable_ip2tag2ip_retr_opt",
          "fountain_relation_interaction_retr_redis_key",
          "skip_fountain_icf_splash_retr"
        ],
        function_for_common = "retrieval_splash_control",
        lua_script_file = "fountain/retrieval/lua/module/splash_entry_photo__retrieval_splash_control.lua"
      ) \
      .log_debug_info(
          common_attrs = [
            "source_author_third_level_id", 
            "source_hetu_face_ids", 
            "source_hetu_level_four", 
            "source_hetu_level_one", 
            "source_hetu_level_three", 
            "source_hetu_level_two", 
            "source_hetu_tag_level_info_hetu_tag",
            "source_mmu_content_ids_3",
            "source_mmu_content_ids_8",
            "source_mmu_content_ids_10",
            "source_mmu_content_ids_15",
            "source_mmu_content_ids_16",
            "source_mmu_content_ids_17",
            "source_hetu_sim_cluster_id",
            "source_user_hash_tag_id",
          ],
          for_debug_request_only = True
      ) \
      .log_debug_info(
        common_attrs = [
          "skip_fountain_top_subdivision_nn_retrieval_tag_splash",
          "fountain_retrieval_skip_top_subdivision_nn_retrieval",
          "skip_fountain_reco_emb_hetu_retrieval_splash",
          "fountain_swing_retr_redis_key",
          "fountain_skip_reco_emb_u2i_retr_splash",
          "fountain_skip_gcse_u2i_retrieval_splash",
          "source_hetu_level_one_v2",
          "source_hetu_level_two_v2",
          "source_hetu_level_three_v2",
          "source_hetu_level_four_v2",
          "source_hetu_tag_v2",
          "source_hetu_face_id_v2",
          "source_hetu_cluster_id_v2",
          "source_movie_related_ips_key",
          "source_movie_ip_extends_key",
          "fountain_skip_ip2tag2ip_retr_splash",
          "fountain_enable_ip2tag2ip_retr_opt",
          "fountain_relation_interaction_retr_redis_key",
          "skip_fountain_icf_splash_retr"
        ],
        for_debug_request_only = True,
      ) \
      .enrich_attr_by_lua(
        import_common_attr = [
          "fountain_splash_increase_quota_time_window"
        ],
        export_common_attr = ["increase_quota_status"],
        function_for_common = "splash_increase_quota_status",
        lua_script_file = "fountain/retrieval/lua/module/increase_quota_attrs__increase_quota.lua")

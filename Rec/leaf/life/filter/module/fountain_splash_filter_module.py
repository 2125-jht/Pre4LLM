from filter import CommonModule


class FountainSplashFilterModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  @property
  def item_attr_map(self) -> dict:
    attr_map = {
      "explore_server_show_attr": "explore_stat__show_count",
    }
    attr_map.update(self.flow.base_item_attr_map)
    return attr_map

  @property
  def filters(self) -> list:
    filter_list = [
      {
        "name": "source_aid",
        "enable": "{{fountain_enable_source_aid_filter}}",
        "source_aid_attr": "sourcePidAuthorId",
      },
      {
        "name": "source_dup_content_id_filter",
        "enable": "{{fountain_enable_source_content_filter}}",
        "source_pid_attr": "featureSourcePId",
        "source_content_type_list_attr": "fountain_source_content_filter_ids",
      },
    ]
    filter_list += self.flow.base_filters
    return filter_list

  @property
  def sec_tab_truncation_map(self) -> dict:
    return {
    }

  @property
  def truncation_map(self) -> dict:
    return {
      "196": 360,
      "6300": 350,
    }

  def process(self):
    self.flow \
      .base_params() \
      .split_string(
        input_common_attr = "fountain_source_content_filter_ids_str",
        output_common_attr = "fountain_source_content_filter_ids",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
        parse_to_int = True,
      ) \
      .explore_life_retrieval_filter(
        user_info_ptr_attr="user_info_ptr",
        item_attr_map=self.item_attr_map,
        filters=self.filters,
        truncation_map={
          "default": 5000,
        },
      )
    self.process_splash_similarity()

  def process_splash_similarity(self):
    self.flow\
      .pack_item_attr(
        item_source={
          "reco_results": False,
          "common_attr": ["featureSourcePId"],
        },
        mappings=[
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
            "from_item_attr": "hetu_sim_cluster_id",
            "to_common_attr": "source_hetu_sim_cluster_id",
          }
        ]
      ) \
      .enrich_attr_by_lua(
        import_common_attr=[
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
        export_common_attr=[
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
        function_for_common="retrieval_splash_control",
        lua_script_file="life/filter/lua/module/fountain_splash_filter__similarity.lua"
      ) \
      .if_("enable_fountain_related_score_calc_v3 == 1") \
        .get_kconf_params(
          kconf_configs = [
            {
              "kconf_key": "reco.fountain.relatedScoreConfig",
              "json_path": "hetu_conf_key",
              "default_value": [-1],
              "export_common_attr": "splash_related_score_v3_hetu_conf_key_list"
            },
            {
              "kconf_key": "reco.fountain.relatedScoreConfig",
              "json_path": "hetu_conf_value",
              "default_value": [2,3,4,5],
              "export_common_attr": "splash_related_score_v3_hetu_conf_value_list"
            },
            {
              "kconf_key": "reco.fountain.relatedScoreConfig",
              "json_path": "tag_element_conf",
              "default_value": 4,
              "export_common_attr": "splash_related_score_v3_tag_element_score"
            },
            {
              "kconf_key": "reco.fountain.relatedScoreConfig",
              "json_path": "tag_content_conf",
              "default_value": 3,
              "export_common_attr": "splash_related_score_v3_tag_content_score"
            },
            {
              "kconf_key": "reco.fountain.relatedScoreConfig",
              "json_path": "ip_conf",
              "default_value": 6,
              "export_common_attr": "splash_related_score_v3_ip_score"
            },
            {
              "kconf_key": "reco.fountain.relatedScoreConfig",
              "json_path": "cid_conf",
              "default_value": 6,
              "export_common_attr": "splash_related_score_v3_cid_score"
            },
            {
              "kconf_key": "reco.fountain.relatedScoreConfig",
              "json_path": "aid_conf",
              "default_value": 2,
              "export_common_attr": "splash_related_score_v3_aid_score"
            },
          ]
        ) \
        .explore_related_score_enricher_v2(
          # 相关门槛分
          source_hetu_attr_list = ["source_hetu_level_one_v2", "source_hetu_level_two_v2", "source_hetu_level_three_v2", "source_hetu_level_four_v2"],
          source_author_str_list = ["sourcePidFourthLevelCategory", "sourcePidThirdLevelCategory"],
          source_face_id_attr = "source_hetu_face_id_v2",
          source_hetu_tag_attr = "source_hetu_tag_v2",
          source_cluster_id_attr = "source_hetu_cluster_id_v2",
          source_aid_attr = "sourcePidAuthorId",
          source_hetu_cid_attr = "source_hetu_sim_cluster_id",
          target_hetu_attr_list = ["hetu_tag_level_info_v2__hetu_level_one", "hetu_tag_level_info_v2__hetu_level_two", "hetu_tag_level_info_v2__hetu_level_three", "hetu_tag_level_info_v2__hetu_level_four"],
          target_author_str_list = ["author__category_detail__fourth_level_id", "author__category_detail__third_level_id"],
          target_face_id_attr = "hetu_tag_level_info_v2__hetu_face_id",
          target_hetu_tag_attr = "hetu_tag_level_info_v2__hetu_tag",
          target_cluster_id_attr = "hetu_tag_level_info_v2__hetu_cluster_id",
          target_aid_attr = "author__id",
          target_hetu_cid_attr = "hetu_sim_cluster_id",
          save_score_to_attr = "fountain_related_score_v2_splash",
          save_score_detail_to_attr = "fountain_related_score_v2_detail_splash",
          hetu_conf_score_key_list = "splash_related_score_v3_hetu_conf_key_list",
          hetu_conf_score_value_list = "splash_related_score_v3_hetu_conf_value_list",
          tag_element_conf_score = "splash_related_score_v3_tag_element_score",
          tag_content_conf_score = "splash_related_score_v3_tag_content_score",
          ip_conf_score = "splash_related_score_v3_ip_score",
          cid_conf_score = "splash_related_score_v3_cid_score",
          aid_conf_score = "splash_related_score_v3_aid_score",
          enable_use_author = "{{enable_fountain_related_score_calc_v2_use_author}}",
          enable_v3 = "{{enable_fountain_related_score_calc_v3}}",
        ) \
      .else_() \
        .explore_related_score_enricher_v2(
          source_hetu_attr_list=["source_hetu_level_one_v2", "source_hetu_level_two_v2", "source_hetu_level_three_v2", "source_hetu_level_four_v2"],
          source_author_str_list=["sourcePidFourthLevelCategory", "sourcePidThirdLevelCategory"],
          source_face_id_attr="source_hetu_face_id_v2",
          source_hetu_tag_attr="source_hetu_tag_v2",
          source_cluster_id_attr="source_hetu_cluster_id_v2",
          target_hetu_attr_list=[
            "hetu_tag_level_info_v2__hetu_level_one",
            "hetu_tag_level_info_v2__hetu_level_two",
            "hetu_tag_level_info_v2__hetu_level_three",
            "hetu_tag_level_info_v2__hetu_level_four"
          ],
          target_author_str_list=["author__category_detail__fourth_level_id", "author__category_detail__third_level_id"],
          target_face_id_attr="hetu_tag_level_info_v2__hetu_face_id",
          target_hetu_tag_attr="hetu_tag_level_info_v2__hetu_tag",
          target_cluster_id_attr="hetu_tag_level_info_v2__hetu_cluster_id",
          save_score_to_attr="fountain_related_score_v2_splash",
          hetu_level_one_set=[6, 39, 32, 7, 31, 15, 18, 12, 25],
        ) \
      .end_() \
      .filter_by_attr_with_perf(
        attr_name="fountain_related_score_v2_splash",
        remove_if="==",
        compare_to=0,
        remove_if_attr_missing=False,
        cancel_num="{{fountain_related_score_filter_cancel_num}}",
      )\
      .log_debug_info(
        common_attrs=[
          "request_type",
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
          # output
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
          "skip_fountain_icf_splash_retr",
          # temp log
          'source_author_third_level_id', 'source_hetu_cluster_ids', 'source_hetu_face_ids', 'source_hetu_level_five', 'source_hetu_level_four', 'source_hetu_level_one',
          'source_hetu_level_three', 'source_hetu_level_two', 'source_hetu_tag_level_info_hetu_tag',
          # 临时
          'enable_fountain_video_filter',
          "fountain_source_content_filter_ids"
        ],
        item_attrs=[
          "hetu_tag_level_info_v2__hetu_level_one",
          "hetu_tag_level_info_v2__hetu_level_two",
          "hetu_tag_level_info_v2__hetu_level_three",
          "hetu_tag_level_info_v2__hetu_level_four",
          'fountain_related_score_v2_splash',
          "hetu_tag_level_info_v2__hetu_face_id",
          "hetu_tag_level_info_v2__hetu_tag",
          "fountain_related_score_v2_detail_splash"
        ],
        for_debug_request_only=True,
        item_num_limit=20,
      ) \
      .if_("fountain_skip_trans_hetu_tag_item_attr_new == 0") \
        .explore_transform_hetu_tag(
          output_attrs = ["hetu_level_one_v2", "hetu_level_two_v2",  "hetu_level_three_v2", "hetu_level_four_v2", "hetu_tag_v2", "hetu_face_id_v2"],
          hetu_tag_attrs = ["hetu_tag_level_info_v2__hetu_level_one", "hetu_tag_level_info_v2__hetu_level_two", "hetu_tag_level_info_v2__hetu_level_three", "hetu_tag_level_info_v2__hetu_level_four", "hetu_tag_level_info_v2__hetu_tag", "hetu_tag_level_info_v2__hetu_face_id"]
        ) \
      .end_if_() \
      .if_("enable_fountain_related_score_calc_v3 == 1") \
        .fountain_calc_related_score_v2(
          # 相关排序分
          enable_cal_photo_sim_by_intersect = "{{enable_fountain_related_score_calc_v3}}",
          diversity_dim_weight = "{{source_related_dim_weight_v3}}",
          save_score_to_attr = "source_related_score",
          int_source_attrs = [
            "source_hetu_sim_cluster_id", "source_hetu_cluster_id_v2",
            "sourcePidMmuImgClusterV3", "sourcePidMmuTextCluster", 
            "sourcePidAuthorId", "sourcePidFirstLevelCategory",
            "sourcePidSecondLevelCategory", "sourcePidThirdLevelCategory",
            "sourcePidTagId", "sourcePidUploadType",
          ],
          int_list_source_attrs = [
            "source_hetu_level_one_v2", "source_hetu_level_two_v2",
            "source_hetu_level_three_v2", "source_hetu_level_four_v2",
            "source_hetu_tag_v2", "source_hetu_face_id_v2"
          ],
          int_item_attrs = [
            "hetu_sim_cluster_id", "hetu_tag_level_info_v2__hetu_cluster_id",
            "mmu_img_cluster_v3", "mmu_text_cluster",
            "author__id", "author__category_detail__first_level_id",
            "author__category_detail__second_level_id", "author__category_detail__third_level_id",
            "tag", "upload_type",
          ],
          int_list_item_attrs = [
            "hetu_level_one_v2", "hetu_level_two_v2",
            "hetu_level_three_v2", "hetu_level_four_v2",
            "hetu_tag_v2", "hetu_face_id_v2",
          ],
        ) \
      .end_() \

  def post_process(self) -> None:
    self.flow \
      .log_debug_info(
        common_attrs=[
          'rank_neg_photo_id_list_str',
          'rerank_neg_photo_id_list_str',
          'explore_nearline_user_update_flag',
        ],
        item_attrs=[
          'audit_cold_review_level',
          'audit_hot_high_subdivision_level',
          'audit_risk_immd_tag',
          'audit_user_experiment_level',
          'author__explore_report_thirtyday__low_report_count',
          'author__explore_report_thirtyday__total_report_count',
          'auto_audit_black_exempt_level_v1',
          'cuckoo_info__author_type',
          'data_set_tags_bit',
          'dup_cluster_id',
          'enable_download',
          'explore_operation_c_review_level',
          'explore_stat__report_detail__low_report_count',
          'final_cross_section_first_class_id',
          'fountain_stats__like_count',
          'fountain_stats__real_show_count',
          'hetu_tag_level_info_v2__hetu_tag',
          'hetu_tag_level_info_v3__hetu_level_one',
          'high_hot_audit_tag_v2',
          'is_jianguan_risk_photo',
          'is_sirius_punish',
          'kuaishan_id',
          'mmu_content_ids_10',
          'mmu_content_ids_14',
          'mmu_content_ids_15',
          'mmu_content_ids_16',
          'mmu_content_ids_17',
          'mmu_content_ids_3',
          'mmu_content_ids_8',
          'mmu_content_ids_9',
          'mmu_low_quality_model_score_104',
          'mmu_low_quality_model_score_123',
          'mmu_low_quality_model_score_143',
          'mmu_low_quality_model_score_145',
          'mmu_low_quality_model_score_150',
          'mmu_low_quality_model_score_151',
          'mmu_low_quality_model_score_163',
          'mmu_low_quality_model_score_164',
          'mmu_low_quality_model_score_40',
          'mmu_low_quality_model_score_42',
          'mmu_low_quality_model_score_46',
          'mmu_low_quality_model_score_52',
          'mmu_low_quality_model_score_63',
          'mmu_low_quality_model_score_64',
          'mmu_low_quality_model_score_90',
          'nebula_stats__like_count',
          'nebula_stats__real_show_count',
          'ocr_cover_text_evil_level',
          'outer_material_id',
          'photo_category_info__ecom_intent_score',
          'photo_status', 'pic_and_selfdup_id',
          'review_pass_level_b',
          'risk_level',
          'sim_remove_dup_id',
          'thanos_stats__like_count',
          'thanos_stats__real_show_count',
          'title_evil_level',
          'topk_audit_tag',
          'video_cold_start_info__photo_dynamic_xtrs_str',
          'video_quality_assessment_flag',
          'young_inc_tags',
        ],
        for_debug_request_only=True
      )

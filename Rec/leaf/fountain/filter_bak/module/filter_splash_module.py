from filter import CommonModule
from . import common_filter

class FilterSplashModule(CommonModule):
  ITEM_ATTR_MAP = {
    "author_fans_count_attr": "author__fans_count",
    "explore_server_show_attr": "explore_stat__show_count",
  }

  FILTERS = [
    {
      "name": "source_aid",
      "enable": True,
      "source_aid_attr": "sourcePidAuthorId",
    },
    {
      "name": "follow_author",
      "enable": True,
      "follow_author_filter_timegap_attr": "fountain_follow_author_filter_timegap",
      "author_id_attr": "author__id",
      "upload_time_attr": "upload_time",
    },
    {
      "name": "low_fans_lite",
      "enable": "{{fountain_enable_low_fans_lite_filter_splash}}",
      "count_threshold_attr": "fountain_author_fans_low_bound_splash",
    },
    {
      "name": "low_server_show_lite",
      "enable": "{{fountain_enable_low_server_show_lite_filter_splash}}",
      "count_threshold_attr": "fountain_show_cnt_low_bound_splash",
    },
  ]

  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self):
    self.flow \
      .enrich_attr_by_lua(
        import_common_attr = [
          "photoTagBucket",
          "featureUId",
          "topSubdivisionHetuBucket",
          "topSubdivisionBucket",
          "featureSourcePId",
          "fountainHetuTagBucket",
          "source_hetu_level_one_v2_original",
          "source_hetu_level_two_v2_original",
          "source_hetu_level_three_v2_original",
          "source_hetu_level_four_v2_original",
          "source_hetu_tag_v2_original",
          "source_hetu_face_id_v2_original",
          "source_hetu_cluster_id_v2_original",
          "sourceMovieIp",
          "fountain_ip2tag2ip_retr_movie2movie_level",
          "currentTimeMs",
        ],
        export_common_attr = [
          "source_hetu_level_one_v2",
          "source_hetu_level_two_v2",
          "source_hetu_level_three_v2",
          "source_hetu_level_four_v2",
          "source_hetu_tag_v2",
          "source_hetu_face_id_v2",
          "source_hetu_cluster_id_v2",
        ],
        function_for_common = "retrieval_splash_control",
        lua_script_file = "fountain/filter/lua/filter_splash__control.lua"
      ) \
      .enrich_attr_by_lua(
        import_common_attr = [
          "sourcePidAuthorId",
          "source_hetu_level_one_v2",
          "enable_cal_information_score_init"
        ],
        export_common_attr = [
          "sourcePidAuthorId",
          "enable_cal_information_score_splash"
        ],
        function_for_common = "calculate",
        lua_script_file = "fountain/filter/lua/filter_splash__control.lua") \
      .if_("enable_cal_information_score_splash == 1") \
        .get_kconf_params(
          kconf_configs=[{
          "kconf_key": "reco.fountain.informationHetuTagId",
          "value_type": "list_int64",
          "default_value": [],
          "export_common_attr": "information_hetu_tag_id"
          }]
        ) \
      .end_if_()

    common_filter.append_prepare_processors(self.flow)

    self.flow \
      .if_("fountain_enable_splash_retr_filter_limit == 1") \
        .explore_retrieval_filter(
          name = "explore_retr_filter_splash_limit",
          traceback = True,
          user_info_ptr_attr = "userInfoPb",
          item_attr_map = {**common_filter.ITEM_ATTR_MAP, **FilterSplashModule.ITEM_ATTR_MAP},
          filters = common_filter.FILTERS + FilterSplashModule.FILTERS,
        ) \
      .else_() \
        .explore_retrieval_filter(
          name = "explore_retr_filter_splash",
          traceback = True,
          user_info_ptr_attr = "userInfoPb",
          item_attr_map = {**common_filter.ITEM_ATTR_MAP, **FilterSplashModule.ITEM_ATTR_MAP},
          filters = common_filter.FILTERS + FilterSplashModule.FILTERS,
          truncation_map = {
            "default": 5000,
          },
        ) \
      .end_() \
      .enrich_attr_by_lua(
        import_common_attr = [
          "source_hetu_level_one_v2",
          "source_hetu_level_two_v2",
          "source_hetu_level_three_v2",
          "source_hetu_level_four_v2",
          "sourcePidFourthLevelCategory",
          "sourcePidThirdLevelCategory",
          "source_hetu_face_id_v2",
          "source_hetu_tag_v2",
          "source_hetu_cluster_id_v2",
          "fountain_skip_filter_photo_by_not_related_reason_splash_v2",
          "fountain_skip_filter_photo_by_not_related_information_splash"
        ],
        export_common_attr = [
          "fountain_skip_filter_photo_by_not_related_reason_splash_v2",
          "fountain_skip_filter_photo_by_not_related_information_splash"
        ],
        function_for_common = "calculate",
        lua_script_file = "fountain/filter/lua/filter_splash__skip_empty_source_hetu.lua",
        skip = "{{fountain_skip_empty_source_hetu_filter}}") \
      .if_("enable_fountain_use_hetu_v1_related_score_calc_v2_splash == 1") \
        .explore_related_score_enricher_v2(
          source_hetu_attr_list = ["source_hetu_level_one", "source_hetu_level_two", "source_hetu_level_three", "source_hetu_level_four"],
          source_author_str_list = ["sourcePidFourthLevelCategory", "sourcePidThirdLevelCategory"],
          source_face_id_attr = "source_hetu_face_ids",
          source_hetu_tag_attr = "source_hetu_tag_level_info_hetu_tag",
          source_cluster_id_attr = "source_hetu_cluster_ids",
          target_hetu_attr_list = ["hetu_tag_level_info__hetu_level_one", "hetu_tag_level_info__hetu_level_two", "hetu_tag_level_info__hetu_level_three", "hetu_tag_level_info__hetu_level_four"],
          target_author_str_list = ["author__category_detail__fourth_level_id", "author__category_detail__third_level_id"],
          target_face_id_attr = "hetu_tag_level_info__hetu_face_id",
          target_hetu_tag_attr = "hetu_tag_level_info__hetu_tag",
          target_cluster_id_attr = "hetu_tag_level_info__hetu_cluster_id",
          save_score_to_attr = "fountain_related_score_v2",
          enable_hetu_v1 = True,
          target_reason = [
            10002, 10038, 10046, 10071, 10082, 10083, 10084, 10088, 10098, 10135, 10143, 10147,
            10149, 10150, 10300, 10301, 10308, 10310, 10311, 10312, 10317, 10318, 10324, 10325,
            10326, 10328, 10329, 10400, 10401, 10402, 10403, 10405, 10407, 10408, 10424, 10302,
            10426, 10788, 10790, 11207, 11208, 11501, 11502, 10136, 10417, 13020, 13021, 10303,
            10401, 10151, 10152, 10313, 10314, 13017, 13026, 10409, 10414, 10415, 10239
          ]
        ) \
      .end_() \
      .if_("enable_cal_information_score_splash == 1") \
        .explore_information_related_score_enricher(
          information_related_score = "information_related_score",
          source_hetu_attr_list = ["source_hetu_level_three_v2", "source_hetu_level_two_v2"],
          source_author_str_list = ["sourcePidFourthLevelCategory", "sourcePidThirdLevelCategory"],
          source_face_id_attr = "source_hetu_face_id_v2",
          source_hetu_tag_attr = "source_hetu_tag_v2",
          source_cluster_id_attr = "source_hetu_cluster_id_v2",
          target_hetu_attr_list = ["hetu_tag_level_info_v2__hetu_level_three", "hetu_tag_level_info_v2__hetu_level_two"],
          target_author_str_list = ["author__category_detail__fourth_level_id", "author__category_detail__third_level_id"],
          target_face_id_attr = "hetu_tag_level_info_v2__hetu_face_id",
          target_hetu_tag_attr = "hetu_tag_level_info_v2__hetu_tag",
          target_cluster_id_attr = "hetu_tag_level_info_v2__hetu_cluster_id",
          information_hetu_tag_id_attr = "information_hetu_tag_id",
          use_hetu_v3 = True,
        ) \
      .end_if_() \
      .if_("fountain_splash_use_emb_similarity_score_filter == 1") \
        .pack_item_attr(
          target_item = {"fountain_related_score_v2": 0},
          item_source = {
            "reco_results": True,
          },
          mappings = [{
            "from_item_attr": "photo_id",
            "to_common_attr": "zero_related_score_photos",
          }]
        ) \
        .pack_common_attr(
          input_common_attrs = ["zero_related_score_photos", "featureSourcePId"],
          output_common_attr = "embedding_source_pids",
        ) \
        .get_remote_embedding_lite(
          kess_service = "grpc_MMUHetuContentEmbeddingV2",
          shard_num = 4,
          timeout_ms = 30,
          id_converter = {"type_name": "kuibaEmbeddingIdConverter"},
          input_attr_name = "embedding_source_pids",
          output_attr_name = "mmu_embeddings",
          query_source_type = "common_attr",
          size = 64,
          client_side_shard = True
        ) \
        .explore_custom_embedding_score_enricher(
          target_item = {"fountain_related_score_v2": 0},
          enable_fix_low_hit_rate = True,
          user_info_ptr_attr = "userInfoPb",
          embedding_list_attr = "mmu_embeddings",
          source_pids_list_attr = "embedding_source_pids",
          calc_type = "single_dot",
          export_item_attr = "splash_retr_similary_score",
          dim_size = 64,
          source_photo_id = "{{featureSourcePId}}"
        ) \
        .set_attr_value( # 给一个默认值让规则过滤掉, 主要针对emb获取失败的photo, 没有这个分数 
          no_overwrite = True,
          item_attrs = [
            {
              "name": "emb_similary_photo",
              "type": "int",
              "value": 0
            }
          ]
        ) \
        .transform_item_attr(
          mappings = [
            {
              "check_attr_name": "splash_retr_similary_score",
              "check_attr_type": "double",
              "output_attr_name": "emb_similary_photo",
              "output_attr_type": "int",
              "output_default_value": 0,
              "rules": [
                {
                  "check_range": {
                    "lower_bound": "{{fountain_splash_emb_similarity_score_threshold}}", # 包含，可缺省
                  },
                  "output_value": 1
                },
              ]
            }
          ]
        ) \
        .filter_by_attr(
          target_item = {"emb_similary_photo": 0},
          attr_name = "fountain_related_score_v2",
          remove_if = "==",
          compare_to = 0,
          remove_if_attr_missing = False,
          cancel_num = "{{fountain_related_score_filter_cancel_num}}"
        ) \
      .else_() \
        .if_("fountain_skip_filter_photo_by_not_related_reason_splash_v2 == 0") \
          .filter_by_attr(
            attr_name = "fountain_related_score_v2",
            remove_if = "==",
            compare_to = 0,
            remove_if_attr_missing = False,
            cancel_num = "{{fountain_related_score_filter_cancel_num}}"
          ) \
        .end_() \
      .end_() \
      .filter_by_attr(
        attr_name = "information_related_score",
        remove_if = "==",
        compare_to = 0,
        remove_if_attr_missing = False,
        cancel_num = "{{fountain_related_score_filter_cancel_num}}",
        skip = "{{fountain_skip_filter_photo_by_not_related_information_splash}}") \
      .copy_item_meta_info(
        save_reason_to_attr = "reason") \
      .transform_item_attr(
        mappings = [
          {
            "check_attr_name": "author__id",
            "check_attr_type": "int",
            "output_attr_name": "is_photo_author_followed",
            "output_attr_type": "int",
            # 检查规则
            "rules": [{
              # 当 author__id 在 followAuthors 内时
              "check_values": ["{{followAuthors}}"],
              "output_value": 1,
            }]
          },
        ]
      ) \
      .set_attr_value(
        item_attrs=[
          {
            "name": "is_follow_author", # 关注作者建议使用此字段
            "type": "int",
            "value": 1
          }
        ],
        target_item = {
          "is_photo_author_followed": 1
        },
      ) \
      .count_item_attr(
        counters = [{
          "check_attr_name": "hetu_tag_level_info__hetu_level_one",
          "check_values": [
            "{{source_hetu_level_one}}"
          ],
          "output_attr_name": "is_photo_same_hetu_level_one"
        },
        {
          "check_attr_name": "author__category_detail__third_level_id",
          "check_values": [
            "{{source_author_third_level_id}}"
          ],
          "output_attr_name": "is_photo_same_author_third_level_id"
        },
        {
          "check_attr_name": "hetu_tag_level_info__hetu_level_two",
          "check_values": [
            "{{source_hetu_level_two}}"
          ],
          "output_attr_name": "is_photo_same_hetu_level_two"
        },
        {
          "check_attr_name": "hetu_tag_level_info__hetu_level_three",
          "check_values": [
            "{{source_hetu_level_three}}"
          ],
          "output_attr_name": "is_photo_same_hetu_level_three"
        },
        {
          "check_attr_name": "hetu_tag_level_info__hetu_level_four",
          "check_values": [
            "{{source_hetu_level_four}}"
          ],
          "output_attr_name": "is_photo_same_hetu_level_four"
        },
        {
          "check_attr_name": "hetu_tag_level_info__hetu_tag",
          "check_values": [
            "{{source_hetu_tag_level_info_hetu_tag}}"
          ],
          "output_attr_name": "is_photo_same_hetu_tag"
        },
        {
          "check_attr_name": "hetu_tag_level_info__hetu_face_id",
          "check_values": [
            "{{source_hetu_face_ids}}"
          ],
          "output_attr_name": "is_photo_same_hetu_face_id"
        }]
      ) \
      .if_("fountain_skip_trans_hetu_tag_item_attr_new == 0") \
        .enrich_attr_by_lua(
          import_item_attr = [
            "hetu_tag_level_info__hetu_level_one",
            "hetu_tag_level_info__hetu_level_two",
          ],
          export_item_attr = [
            "hetu_level_one",
            "hetu_level_two",
          ],
          function_for_item = "calculate",
          lua_script_file = "fountain/filter/lua/filter_splash__trans_hetu_tagv2.lua") \
        .explore_transform_hetu_tag(
          output_attrs = ["hetu_level_one_v2"],
          hetu_tag_attrs = ["hetu_tag_level_info_v2__hetu_level_one"],
        ) \
      .end_if_() \
      .count_reco_result(save_count_to="fountain_splash_item_num_after_filter") \
      .count_reco_result(
        save_count_to="backup_retr_item_num",
        target_reason=[10406]  # 兜底召回源 
      ) \
      .if_("skip_fountain_filter_backup_retr_splash == 0 and fountain_splash_item_num_after_filter - backup_retr_item_num >= fountain_splash_backup_retr_num_threshold") \
        .filter_by_attr(
          attr_name = "reason",
          remove_if = "==",
          compare_to = 10406,
          remove_if_attr_missing = False
        )\
      .end_if_() \
      .perflog_reason_count(
        check_point = "filter_finish"
      )
      
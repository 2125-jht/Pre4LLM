from rerank import CommonModule

class McDistillSampleModule(CommonModule):
    def __init__(self, name: str) -> None:
      super().__init__(name)

    def process(self) -> None:
      self.flow \
        .truncate(
          name = "explore_rr_truncate",
          traceback = True,
          size_limit = "{{max_rpc_fr_result_num}}",
        ) \
        .copy_item_meta_info(
          save_item_seq_to_attr = "rerank_final_index",
        ) \
        ._dump_attr_to_kafka(
          stage_name = "rerank", 
          dump_item_attr_list = [
            "explore_fr_ensemble_score"
          ]
        ) \
        .pack_item_attr(  # 保存 rerank 正样本
          item_source = {
            "reco_results": True,
            "total_limit": "{{mc_distill_sample_num}}",
          },
          mappings = [{
            "aggregator": "concat",
            "from_item_attr": "item_key",
            "to_common_attr": "rerank_pos_sample_list",
          }],
        ) \
        .pack_item_attr(  # 保存 rerank 结束后的结果集
          item_source = {
            "reco_results": True
          },
          mappings = [{
            "aggregator": "concat",
            "from_item_attr": "item_key",
            "to_common_attr": "rerank_output_item_key_list",
          }],
        ) \
        .if_("skip_mc_distill_sample == 0") \
          .if_("_REQ_TYPE_ == \"life\"") \
            .set_attr_value(
              no_overwrite = False,
              common_attrs = [
                {
                  "name": "mc_distill_tab_id",
                  "type": "int",
                  "value": 100
                }
              ]
            ) \
          .else_() \
            .set_attr_value(
              no_overwrite = False,
              common_attrs = [
                {
                  "name": "mc_distill_tab_id",
                  "type": "int",
                  "value": 0
                }
              ]
            ) \
          .end_() \
          .explore_mc_distill_sample_enrich(  #粗排级联模型样本采样：粗排负样本
            candidate_list_attr = "cascade_input_item_key_list",
            excludes_attr = "cascade_output_item_key_list",
            sample_num = "{{mc_distill_sample_num}}",
            save_sample_result_to = "cascade_neg_sample_list",
          ) \
          .explore_mc_distill_sample_enrich(  #粗排级联模型样本采样：精排负样本
            candidate_list_attr = "cascade_output_item_key_list",
            excludes_attr = "rerank_output_item_key_list",
            sample_num = "{{mc_distill_sample_num}}",
            save_sample_result_to = "ranking_neg_sample_list",
          ) \
          .explore_mc_distill_sample_enrich(  #粗排级联模型样本采样：rerank负样本
            candidate_list_attr = "rerank_output_item_key_list",
            excludes_attr = "rerank_pos_sample_list",
            sample_num = "{{mc_distill_sample_num}}",
            save_sample_result_to = "rerank_neg_sample_list",
          ) \
          .explore_mc_distill_sample_reco_log_enrich(  # 粗排级联模型样本采样：填充最终样本(recoLog)
            sample_list_names = ["cascade_neg_sample_list", "ranking_neg_sample_list", "ranking_pos_sample_list", "rerank_neg_sample_list", "rerank_pos_sample_list"],
            user_info_attr = "user_info_ptr",
            send_photo_optional = 0,
            send_user_optional = 0,
            tab = "{{mc_distill_tab_id}}",
            save_result_to = "mc_distill_sample_message",
            label_type_list = ["cas_neg", "rank_neg", "rank_pos", "final_neg", "final_pos"],
          ) \
          .send_with_kafka(
            common_attr = "mc_distill_sample_message",
            topic_name = "mc_distill_samples_dfeed",
          ) \
        .end_() \
        .if_("enable_full_link_sample_package == 1") \
          .copy_item_meta_info(  # 保存 rerank 最后的 index
            save_item_seq_to_attr = "final_index",
          ) \
          .pack_item_attr(  # 保存 rerank 样本，约定 60
            item_source = {
              "reco_results": True,
              "total_limit": 60,
            },
            mappings = [{
              "aggregator": "concat",
              "from_item_attr": "item_key",
              "to_common_attr": "final_output_sample_list",
            }],
          ) \
          .explore_mc_distill_sample_enrich(  #粗排级联模型样本采样：粗排负样本
            candidate_list_attr = "cascade_input_item_key_list",
            excludes_attr = "cascade_output_item_key_list",
            sample_num = "{{explore_full_link_mc_distill_sample_num}}",
            save_sample_result_to = "full_link_cascade_neg_sample_list",
          ) \
          .explore_mc_distill_sample_enrich(  #粗排级联模型样本采样：精排负样本
            candidate_list_attr = "cascade_output_item_key_list",
            excludes_attr = "final_output_sample_list",
            sample_num = "{{explore_full_link_rank_distill_sample_num}}",
            save_sample_result_to = "full_link_ranking_neg_sample_list",
          ) \
          .explore_mc_distill_sample_enrich(  #粗排级联模型样本采样：rerank负样本
            candidate_list_attr = "final_output_sample_list",
            excludes_attr = "rerank_pos_sample_list",
            sample_num = "{{explore_full_link_rarank_distill_sample_num}}",
            save_sample_result_to = "full_link_rerank_neg_sample_list",
          ) \
          .explore_full_link_distill_sample_reco_log_enricher(
            sample_list_names = ["full_link_cascade_neg_sample_list", "full_link_ranking_neg_sample_list", "full_link_rerank_neg_sample_list", "rerank_pos_sample_list"],
            user_info_attr = "user_info_ptr",
            load_attr = "full_link_reco_log_message",
            save_result_to = "full_link_reco_log_message_final",
            enable_set_user_info = "{{explore_enable_full_link_set_user_info}}",
            final_index = "final_index",
            rank_index = "rank_index_before_rerank",
            pctr = "pctr",
            pltr = "pltr",
            pwtr = "pwtr",
            pftr = "pftr",
            pptr = "pptr",
            pcmtr = "pcmtr",
            plvtr = "plvtr",
            pvtr = "pvtr",
            cascade_pctr = "cascade_pctr",
            cascade_pltr = "cascade_pltr",
            cascade_pwtr = "cascade_pwtr",
            cascade_pftr = "cascade_pftr",
            cascade_pptr = "cascade_ptr",
            cascade_pcmtr = "cascade_pcmtr",
            cascade_plvtr = "cascade_plvtr",
            cascade_pvtr = "cascade_pwatch_time",
            label_type_list = ["cas_neg", "rank_neg", "final_neg", "final_pos"],
          ) \
          .send_with_kafka(
            common_attr = "full_link_reco_log_message_final",
            topic_name = "full_link_samples",
          ) \
        .end_() \
        .if_("explore_enable_rerank_write_rerank_neg_result_to_redis == 1") \
          .if_("explore_enable_rerank_select_rerank_neg_result == 1") \
            .enrich_attr_by_light_function(
              import_common_attr = [
                {"name": "explore_rerank_neg_photo_target_ratio", "as": "target_ratio"},
              ],
              import_item_attr = [
                {"name": "rank_final_index", "as": "before_index"},
                {"name": "rerank_final_index", "as": "after_index"},
                "photo_id"
              ],
              export_common_attr = [
                {"name": "target_pids", "as": "rerank_neg_photo_id_list"},
              ],
              function_name = "SelectRecoNegPids",
              class_name = "ExploreLightFunctionSetV2",
              range_start = "{{explore_rerank_neg_photo_index}}",
              range_end = "{{explore_rerank_neg_photo_end_index}}"
            ) \
          .else_() \
            .pack_item_attr(
              item_source = {
                "reco_results": True
              },
              mappings = [{
                "from_item_attr": "item_key",
                "to_common_attr": "rerank_neg_photo_id_list",
                "aggregator": "concat"
              }],
              range_start = "{{explore_rerank_neg_photo_index}}",
              range_end = "{{explore_rerank_neg_photo_end_index}}"
            ) \
          .end_() \
          .pack_common_attr(
            input_common_attrs = [
              "rerank_neg_photo_id_list",
              "rerank_neg_photo_id_filter_list"
            ],
            output_common_attr = "rerank_neg_photo_id_list",
            deduplicate = True,
            limit_num = "{{explore_rerank_neg_photo_size}}",
          ) \
          .write_to_redis(
            kcc_cluster = "recoExploreNegPhoto",
            timeout = 10,
            expire_second = "{{explore_rerank_neg_photo_redis_expire_seconds}}",
            key_prefix = "{{explore_rerank_neg_photo_key_prefix}}",
            key = "{{_DEVICE_ID_}}",
            value = "{{rerank_neg_photo_id_list}}"
          ) \
        .end_() \
        .if_("explore_enable_rerank_write_rerank_pos_result_to_redis == 1") \
          .if_("explore_enable_rerank_select_rerank_pos_result == 1") \
            .enrich_attr_by_light_function(
              import_common_attr = [
                {"name": "explore_select_rerank_pos_result_throw_thres", "as": "throw_thres"},
                {"name": "explore_select_rerank_pos_result_enable_sort", "as": "enable_sort"},
                {"name": "explore_select_rerank_pos_result_ctr_power_weight", "as": "ctr_power_weight"},
                {"name": "explore_select_rerank_pos_result_target_count", "as": "target_count"},
                {"name": "explore_select_rerank_pos_result_filter_map_str", "as": "filter_map_str"}
              ],
              import_item_attr = [
                "photo_id",
                {"name": "corr_pctr", "as": "pctr"},
                {"name": "fr_score2", "as": "pwatch_time"},
                "audit_hot_cover_level",
                {"name": "hetu_tag_level_info_v2__hetu_level_one", "as": "hetu_level_one"}
              ],
              export_common_attr = [
                {"name": "target_pids", "as": "rerank_pos_photo_id_list"}
              ],
              function_name = "ExploreSelectRerankPosPids",
              class_name = "ExploreLightFunctionSetV2",
              range_start = "{{explore_rerank_pos_photo_start_index}}",
              range_end = "{{explore_rerank_pos_photo_end_index}}"
            ) \
          .else_() \
            .pack_item_attr(
              item_source = {
                "reco_results": True
              },
              mappings = [{
                "from_item_attr": "item_key",
                "to_common_attr": "rerank_pos_photo_id_list",
                "aggregator": "concat"
              }],
              range_start = "{{explore_rerank_pos_photo_start_index}}",
              range_end = "{{explore_rerank_pos_photo_end_index}}"
            ) \
          .end_() \
          .enrich_attr_by_light_function(
            import_common_attr = [
              {"name": "rerank_pos_photo_id_list", "as": "universal_set_list"},
              {"name": "explore_rerank_pos_photo_id_retrieval_list", "as": "sub_set_list"}
            ],
            export_common_attr = [
              {"name": "difference_list", "as": "rerank_pos_photo_id_list"}
            ],
            function_name = "GetDifferenceSet",
            class_name = "ExploreLightFunctionSetV2",
          ) \
          .pack_common_attr(
            input_common_attrs = [
              "rerank_pos_photo_id_list",
              "explore_rerank_pos_photo_id_retrieval_list"
            ],
            output_common_attr = "rerank_pos_photo_id_list",
            deduplicate = True,
            limit_num = "{{explore_rerank_pos_photo_size}}",
          ) \
          .write_to_redis(
            kcc_cluster = "recoExploreNegPhoto",
            timeout = 10,
            expire_second = "{{explore_rerank_pos_photo_redis_expire_seconds}}",
            key_prefix = "{{explore_rerank_pos_photo_key_prefix}}",
            key = "{{_USER_ID_}}",
            value = "{{rerank_pos_photo_id_list}}"
          ) \
        .end_()
      return self


    def calc_result_count_to_ab_metric(self):
        return self.flow \
          .count_reco_result(
            save_count_to = "rerank_top60_pic_result_count",
            target_item = {"is_picture": 1},
            range_end = 60
          ) \
          .send_abtest_metrics(
            metrics = [
              "rerank_top60_pic_result_count"
            ],
            metric_name_prefix = "explore_reco_leaf_",
          )

    def post_process(self) -> None:
        self.calc_result_count_to_ab_metric()

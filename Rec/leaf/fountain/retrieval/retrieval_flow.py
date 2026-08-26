from dragonfly.ext.explore.explore_api_mixin import ExploreApiMixin
from dragonfly.ext.retrieval.retrieval_api_mixin import RetrievalApiMixin
from dragonfly.ext.gsu.gsu_api_mixin import GsuApiMixin
from dragonfly.ext.embed_calc.embed_calc_api_mixin import EmbedCalcApiMixin
from dragonfly.ext.kgnn.kgnn_api_mixin import KgnnApiMixin
from dragonfly.ext.tdm.tdm_api_mixin import TDMApiMixin
from dragonfly.ext.subdivision.subdivision_api_mixin import subdivisionApiMixin
from dragonfly.ext.kuiba.kuiba_api_mixin import KuibaApiMixin

from common import CommonRecoFlow
from retrieval.module.retrieval_perf_module import RetrievalPerfModule
from retrieval.module.photo_info_fetching_module import PhotoInfoFetchingModule
from retrieval.module.pagesize_dedup_module import PageSizeDedupModule

from dump_attr_to_kafka import dump_attr_to_kafka

class RetrievalFlow(CommonRecoFlow, ExploreApiMixin, RetrievalApiMixin, GsuApiMixin, EmbedCalcApiMixin, KgnnApiMixin, subdivisionApiMixin, KuibaApiMixin, TDMApiMixin):
  def __init__(self, name: str, is_sub_flow: bool = False) -> None:
    super().__init__(name, "fountain", "retrieval", "config", "module", "config/module", "lua/module", is_sub_flow)

  def _get_deduplicate_processor_name(self) -> str:
    if self.name == "retrieval_default":
      return "fountain_retr_dedup"
    elif self.name == "retrieval_splash":
      return "fountain_splash_retr_dedup"
    raise ValueError("Invalid name, candidates: \"retrieval_default\", \"retrieval_splash\"", self.name)

  def _dump_attr_to_kafka(self, stage_name : str, dump_item_attr_list : list):
    """
    dump item attr to kafka
    """
    dump_attr_to_kafka(self, stage_name, dump_item_attr_list)
    return self

  def _flow_end(self):
    self \
      .do_nothing(  # 这个 processor 是为了先知打点出所有召回的原始结果
        name = "fountain_retr",
        traceback = True,
      ) \
      ._perf_result(
        step_name = "origin",
        perf_sampling_attr = "_IS_PERF_SAMPLING_REQUEST_",
      ) \
      .get_abtest_params(
        biz_name = "RECO_RPC",
        ab_params = [
          ("enable_fountain_dump_attrs_to_kafka", 0, "enable_dump_attrs_to_kafka"),
          ("skip_explore_item_reason_score_enricher", 1),
          ("fountain_item_reason_score_smoothing", 1),
          ("fountain_enable_low_pass_rate_global_filter", False),
          ("enable_fountain_bad_item_list_similarity_score", False),
          ("fountain_retr_only_cover_bad_item_save_count", 100),
          ("fountain_retr_only_sense_bad_item_save_count", 100),
        ],
      ) \
    
    if self.name == "retrieval_default":
      self\
        .get_abtest_params(
          biz_name = "RECO_RPC",
          ab_params = [
            ("fountain_enable_similary_neg_feedback_limit", False),
            ("enable_fountain_retr_fix_low_hit_rate", False),
            ("fountain_retr_emb_kess_name_for_similary_neg_feedback", "grpc_MMUHetuContentEmbedding"),
            ("retr_hate_similary_score_not_click_hour_limit", 0.0),
            ("retr_hate_similary_score_play_stat_hour_limit", 0.0),
            ("retr_hate_similary_score_extra_not_click_hour_limit", 0.0),
            ("retr_similary_neg_score_not_click_weight", 1.0),
            ("retr_similary_neg_score_short_view_weight", 1.0),
            ("retr_similary_neg_score_extra_not_click_weight", 1.0),
            ("retr_similary_neg_score_short_view_threshold", 3000),
            ("retr_enable_fountain_play_stat", False),
            ("retr_similary_neg_score_enable_avg_pooling", False),
            ("fountain_enable_similary_neg_feedback_filter_by_rank", False),
            ("fountain_similary_neg_feedback_limit_percent", 1.0),
            ("fountain_similary_neg_score_thres", 100.0),
            ("fountain_retr_reason_limit_num_fast", 7500, "fountain_retr_reason_limit_num"), # 默认值
            # 填充默认值
            ("fountain_retr_reason_limit_reason_weight_str_fast", "317:1.0;336:1.0;314:0.7;415:0.6;322:0.5;405:0.4;436:0.4;326:0.4;414:1.0;417:2.0;310:1.5;406:1.5;419:1.0;330:1.3;327:1.3;343:1.1;3899:1.5;3900:1.5;344:0.8", "fountain_retr_reason_limit_reason_weight_str"),
            ("fountain_retr_reason_limit_skip_fillback_result_fast", False, "fountain_retr_reason_limit_skip_fillback_result"),
          ],
        )
    elif self.name == "retrieval_splash":
      self\
        .get_abtest_params(
          biz_name = "RECO_RPC",
          ab_params = [
            ("fountain_retr_reason_limit_num_splash", 7500, "fountain_retr_reason_limit_num"),
            ("fountain_retr_reason_limit_reason_weight_str_splash", "317:1.0;336:1.0;314:0.7;415:0.6;322:0.5;405:0.4;436:0.4;326:0.4;414:1.0;417:2.0;310:1.5;406:1.5;419:1.0;330:1.3;327:1.3;343:1.1;3899:1.5;3900:1.5;344:0.8", "fountain_retr_reason_limit_reason_weight_str"),
            ("fountain_retr_reason_limit_skip_fillback_result_splash", False, "fountain_retr_reason_limit_skip_fillback_result"),
          ],
        )
        
    self \
      .explore_item_reason_score_enricher(
        mappings = [{
          "reason": 414,
          "to_item_attr": "comirec_rank_score",
        }],
        smoothing = "{{fountain_item_reason_score_smoothing}}",
        skip = "{{skip_explore_item_reason_score_enricher}}") \
      .log_debug_info(
        item_attrs = ["comirec_rank_score"],
      ) \
      .copy_item_meta_info(
        save_item_id_to_attr = "item_id"
      ) \
      .deduplicate(
        name = self._get_deduplicate_processor_name(),
        traceback = True,
        on_item_attr = "item_id",
        save_dup_count_to = "retrieval_dup_count",
      ) \
      .filter_by_browse_set(
        name = "fountain_retr_browse_set_filter",
        traceback = True,
        check_id_in_attr = "item_id",
        item_type_of_checked_id = 0) \
      .if_("swing_user_hate_i2i_list ~= nil and #swing_user_hate_i2i_list > 0") \
        .filter_by_common_attr(
          name = "swing_i2i_neg_photo_id_list_filter",
          traceback = True,
          common_attr = ["swing_user_hate_i2i_list"]
        ) \
      .end_() \
      .if_("mc_i2i_neg_pid_list ~= nil and #mc_i2i_neg_pid_list > 0") \
        .filter_by_common_attr(
          name = "fountain_mc_i2i_neg_pid_list_filter",
          traceback = True,
          common_attr = ["mc_i2i_neg_pid_list"]
        ) \
      .end_() \
      .if_("fountain_enable_low_pass_rate_global_filter == 1", to_be_delete = "date=2024-05-29;committer=denghong") \
        .explore_memory_data_enrich(
          data_key = "fountain_low_pass_rate_photo_set",
          data_type = "uint64_set",
          save_data_ptr_to_attr = "fountain_low_pass_rate_photo_set"
        ) \
        .explore_memory_data_ptr_filter(
          memory_data_ptr_attr = "fountain_low_pass_rate_photo_set"
        ) \
      .end_() \
    
    if self.name == "retrieval_default":
      self \
        .if_("fountain_enable_similary_neg_feedback_limit == 1") \
          .explore_embedding_candidates_attr_enricher(
            trans_type = "fountain_candidates",
            enable_fix_low_hit_rate = "{{enable_fountain_retr_fix_low_hit_rate}}",
            user_info_ptr_attr = "userInfoPb",
            export_common_attr = "embedding_source_pids" 
          ) \
          .get_remote_embedding_lite(
            kess_service = "{{fountain_retr_emb_kess_name_for_similary_neg_feedback}}",
            shard_num = 4,
            timeout_ms = 25,
            id_converter = {"type_name": "kuibaEmbeddingIdConverter"},
            input_attr_name = "embedding_source_pids",
            output_attr_name = "mmu_embeddings",
            query_source_type = "common_attr",
            size = 64,
            client_side_shard = True
          ) \
          .explore_custom_embedding_score_enricher(
            enable_fountain_version = True,
            enable_fix_low_hit_rate = "{{enable_fountain_retr_fix_low_hit_rate}}",
            user_info_ptr_attr = "userInfoPb",
            embedding_list_attr = "mmu_embeddings",
            source_pids_list_attr = "embedding_source_pids",
            calc_type = "action_bucket_dot",
            not_click_limit_hour = "{{retr_hate_similary_score_not_click_hour_limit}}",
            play_stat_limit_hour = "{{retr_hate_similary_score_play_stat_hour_limit}}",
            extra_not_click_limit_hour = "{{retr_hate_similary_score_extra_not_click_hour_limit}}",
            not_click_weight = "{{retr_similary_neg_score_not_click_weight}}",
            short_view_weight = "{{retr_similary_neg_score_short_view_weight}}",
            extra_not_click_weight = "{{retr_similary_neg_score_extra_not_click_weight}}",
            short_view_threshold = "{{retr_similary_neg_score_short_view_threshold}}",
            enable_fountain_play_stat = "{{retr_enable_fountain_play_stat}}",
            enable_avg_pooling = "{{retr_similary_neg_score_enable_avg_pooling}}",
            export_item_attr = "retr_similary_neg_score",
            dim_size = 64,
            check_point_ = "fountain_retrieval",
          ) \
          .if_("fountain_enable_similary_neg_feedback_filter_by_rank == 1", to_be_delete = "date=2024-05-29;committer=denghong") \
            .sort(
              score_from_attr = "retr_similary_neg_score",
              desc = False
            ) \
            .count_reco_result(
              save_count_to = "fountain_retr_v23_item_num"
            ) \
            .gen_common_attr_by_lua(
              attr_map = {
                "fountain_similary_neg_feedback_limit_num": "math.floor(fountain_retr_v23_item_num * fountain_similary_neg_feedback_limit_percent)"
              }
            ) \
            .limit(
              size = "{{fountain_similary_neg_feedback_limit_num}}"
            ) \
          .else_() \
            .filter_by_attr(
              attr_name = "retr_similary_neg_score",
              remove_if = ">=",
              compare_to = "{{fountain_similary_neg_score_thres}}",
              remove_if_attr_missing = False,
            ) \
          .end_() \
        .end_() \
      
    self \
      .explore_retr_reason_limit(
        name = "fountain_retr_quota_limit",
        traceback = True,
        size_limit = "{{fountain_retr_reason_limit_num}}",
        default_weight = 1.0,
        reason_weight_str = "{{fountain_retr_reason_limit_reason_weight_str}}",
        skip_fillback_result = "{{fountain_retr_reason_limit_skip_fillback_result}}",
      )
      
    photo_info_module_name = self.config["photo_info_module"]
    self.namespace_(ns = photo_info_module_name, nest = True)
    photo_info_module = PhotoInfoFetchingModule(photo_info_module_name)
    photo_info_module.set_flow(self)
    photo_info_module.process()
    self.namespace_()

    pagesize_dedup_module_name = self.config["pagesize_dedup_module"]
    self.namespace_(ns = pagesize_dedup_module_name, nest = True)
    photo_info_module = PageSizeDedupModule(pagesize_dedup_module_name)
    photo_info_module.set_flow(self)
    photo_info_module.process()
    self.namespace_()

    self \
      .if_("enable_fountain_bad_item_list_similarity_score == 1") \
        .pack_item_attr(  # 保存进入召回封面劣质结果集，计算相似度
          item_source = {
            "reco_results": True,
            "total_limit" : "{{fountain_retr_only_cover_bad_item_save_count}}",
          },
          mappings = [{
            "aggregator": "concat",
            "from_item_attr": "photo_id",
            "to_common_attr": "fountain_retr_only_bad_cover_input_item_key_list",
          }],
          target_item = { "audit_hot_cover_level": [2023746] }
        ) \
        .pack_item_attr(  # 保存进入召回观感劣质结果集，计算相似度
          item_source = {
            "reco_results": True,
            "total_limit" : "{{fountain_retr_only_sense_bad_item_save_count}}",
          },
          mappings = [{
            "aggregator": "concat",
            "from_item_attr": "photo_id",
            "to_common_attr": "fountain_retr_only_bad_sense_input_item_key_list",
          }],
          target_item = { "content_safety_level_with_namespace__level_hot_online": [0, 1] }
        ) \
        .pack_common_attr(
          input_common_attrs = [
            "fountain_retr_only_bad_cover_input_item_key_list",
            "fountain_retr_only_bad_sense_input_item_key_list"
          ],
          output_common_attr = "fountain_retr_bad_input_item_key_list",
          deduplicate = True  
        ) \
      .end_() \

    self \
      .pack_item_attr(
        item_source = {
          "reco_results": True,
        },
        mappings = [
          {
            "aggregator": "sum",
            "from_item_attr": "retrieval_dup_count",
            "default_val": 0,
            "to_common_attr": "retrieval_dup_count",
          }
        ]
      ) \
      .perflog_attr_value(
        check_point = "retrieval",
        common_attrs = [
          "retrieval_dup_count",
        ]
      ) \
      ._perf_result(
        perf_sampling_attr = "_IS_PERF_SAMPLING_REQUEST_",
      )

    if self.name == "retrieval_default":
      self \
        ._dump_attr_to_kafka(
          stage_name = "retr",
          dump_item_attr_list = ["i2i_trigger_id"],
        )
    else:
      self \
        ._dump_attr_to_kafka(
          stage_name = "retr",
          dump_item_attr_list = [],
        )

    super()._flow_end()

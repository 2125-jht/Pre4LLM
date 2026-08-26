from dragonfly.decorators import apply
from dragonfly.ext.explore.explore_api_mixin import ExploreApiMixin
from dragonfly.ext.embedding.embedding_api_mixin import EmbeddingApiMixin
from dragonfly.common_leaf_dsl import LeafFlow, LeafService

from rerank.rerank_base import rerank_features_v3, gen_photo_features_for_all_position_v3,rerank_gen_model_send_user_feas,rerank_gen_model_send_item_feas,rerank_gen_model_send_common_feas,rerank_flash_eval_model_send_item_feas,rerank_flash_eval_model_send_common_feas,rerank_flash_eval_model_send_user_feas
from rerank.rerank_es_queue import es_add_queues, es_mul_queues, rerank_hetu_fusion_queues

class RerankV3Mixin(ExploreApiMixin, EmbeddingApiMixin):
  def rerank_v3(self):
    self \
      .rerank_predict() \
      .rerank_get_embedding() \
      .rerank_gen_item_attr() \
      .rerank_gen_candidate_list() \
      .rerank_gen_list() \
      .rerank_list_predict() \
      .sort(
        name = "fountain_rr_v3",
        traceback = True,
        stable_sort = True,
        score_from_attr = "rerank_list_score",
      ) \
      .rerank_gen_reason_metrics()

    return self
      
  def rerank_get_embedding(self):
    self \
      .get_remote_embedding_lite(
        kess_service = "grpc_hotMcEmbed",
        shard_num = 8,
        id_converter = {"type_name": "kuibaEmbeddingIdConverter"},
        output_attr_name = "rerank_mc_emb",
        size = 128,
        client_side_shard = True,
      ) \
      .get_remote_embedding_lite(
        kess_service= "grpc_hotRerankMmuEmbServerV3",
        shard_num = 4,
        id_converter = {"type_name": "kuibaEmbeddingIdConverter"},
        output_attr_name = "rerank_mmu_emb",
        size = 64,
        client_side_shard = True,
      ) \
      .fetch_remote_embedding(
        protocol=1,
        colossusdb_embd_model_name="explore_reco_user_pid_author_sid_v1",
        colossusdb_embd_table_name="emb_sid_v1",
        query_source_type="user_id",
        output_attr_name ="user_sid",
        slot=0,
        id_converter = {"type_name": "mioEmbeddingIdConverter"},
        raw_data_type = "int32",
        timeout_ms=100,
        is_raw_data=True,
        size=48,
      ) \
      .fetch_remote_embedding(
        protocol=1,
        colossusdb_embd_model_name="explore_reco_user_pid_author_sid_v1",
        colossusdb_embd_table_name="emb_sid_v1",
        query_source_type="item_id",
        output_attr_name ="photo_sid",
        slot=1,
        id_converter = {"type_name": "mioEmbeddingIdConverter"},
        raw_data_type = "int32",
        timeout_ms=100,
        is_raw_data=True,
        size=8,
      ) \
      .fetch_remote_embedding(
        protocol=1,
        colossusdb_embd_model_name="explore_reco_user_pid_author_sid_v1",
        colossusdb_embd_table_name="emb_sid_v1",
        query_source_type="item_attr",
        input_attr_name="author__id",
        output_attr_name ="author_sid",
        slot=2,
        id_converter = {"type_name": "mioEmbeddingIdConverter"},
        raw_data_type = "int32",
        timeout_ms=100,
        is_raw_data=True,
        size=3,
      ) \
      .if_("fountain_rerank_enable_import_hetu_emb_v4 == 1") \
        .switch_("fountain_rerank_hetu_emb_switch") \
          .case_(1) \
            .get_remote_embedding_lite(
              protocol = 1,
              shard_num = 8,
              colossusdb_embd_service_name = "explore_reco_hetu_emb_v4",
              colossusdb_embd_table_name = "explore_reco_hetu_emb_v4",
              id_converter = {"type_name": "mioEmbeddingIdConverter"},
              output_attr_name = "rerank_hetu_emb",
              size = 128,
              client_side_shard = True,
            ) \
          .case_(2) \
            .get_remote_embedding_lite(
              protocol = 1,
              shard_num = 8,
              colossusdb_embd_service_name = "explore_hetu_emb_server_v51",
              colossusdb_embd_table_name = "explore_hetu_emb_server_v51",
              id_converter = {"type_name": "mioEmbeddingIdConverter"},
              output_attr_name = "rerank_hetu_emb",
              size = 128,
              client_side_shard = True,
            ) \
        .end_() \
      .end_() \
      .log_debug_info(
        log_tag="jht",
        common_attrs=["user_sid"],
        item_attrs=["photo_sid", "author_sid"],
        item_num_limit=1000,
        for_debug_request_only = False,
        respect_sample_logging = False
      )

    return self

  def rerank_predict(self):
    self \
      .if_("fountain_rerank_enable_gen_model_beam == 1") \
        .fountain_rerank_gen_model_beam() \
      .end_() \
      .if_("fountain_rerank_enable_gen_model_nar_beam == 1") \
        .fountain_rerank_gen_model_nar_beam() \
      .end_() \
      .if_("fountain_rerank_enable_gen_model_ar_beam == 1") \
        .fountain_rerank_gen_model_ar_beam() \
      .end_() \
      .if_("fountain_rerank_enable_gen_model_rankmixer == 1") \
        .fountain_rerank_gen_model_rankmixer() \
      .end_() \
      .if_("fountain_rerank_enable_gen_model_ar_pinrec == 1") \
        .fountain_rerank_gen_model_ar_pinrec() \
      .end_() \

    return self

  def calc_user_custom_ensemble_weight(self):
    self \
      .adjust_forward_social_params() \
      .gen_common_attr_by_lua(
        attr_map = {
          "fountain_rerank_ensemble_power_weight_fullrank_ltr_emp": "fountain_rerank_ensemble_power_weight_fullrank_ltr_emp * user_group_emp_ltr",
          "fountain_rerank_ensemble_power_weight_fullrank_wtr_emp": "fountain_rerank_ensemble_power_weight_fullrank_wtr_emp * user_group_emp_wtr",
          "fountain_rerank_ensemble_power_weight_fullrank_ftr_emp": "fountain_rerank_ensemble_power_weight_fullrank_ftr_emp * user_group_emp_ftr",
          "fountain_rerank_ensemble_power_weight_fullrank_cmtr_emp": "fountain_rerank_ensemble_power_weight_fullrank_cmtr_emp * user_group_emp_cmtr",
          "fountain_rerank_ensemble_power_weight_fullrank_ptr_emp": "fountain_rerank_ensemble_power_weight_fullrank_ptr_emp * user_group_emp_ptr",
        },
      ) \
      .pack_item_attr(
        item_source = {
          "reco_results": True,
        },
        mappings = [
          {
            "aggregator": "avg",
            "from_item_attr": "fullrank_sim_pevtr",
            "to_common_attr": "pevtr_avg"
          },
          {
            "aggregator": "avg",
            "from_item_attr": "fullrank_sim_pltr",
            "to_common_attr": "pltr_avg"
          },
          {
            "aggregator": "avg",
            "from_item_attr": "fullrank_sim_pwtr",
            "to_common_attr": "pwtr_avg"
          },
          {
            "aggregator": "avg",
            "from_item_attr": "fullrank_sim_pftr",
            "to_common_attr": "pftr_avg"
          },
          {
            "aggregator": "avg",
            "from_item_attr": "fullrank_sim_pcmtr",
            "to_common_attr": "pcmtr_avg"
          },
          {
            "aggregator": "avg",
            "from_item_attr": "fullrank_sim_pptr",
            "to_common_attr": "pptr_avg"
          },
          {
            "aggregator": "avg",
            "from_item_attr": "fullrank_sim_pepstr",
            "to_common_attr": "pepstr_avg"
          },
          {
            "aggregator": "avg",
            "from_item_attr": "fullrank_sim_plvtr",
            "to_common_attr": "plvtr_avg"
          },
          {
            "aggregator": "avg",
            "from_item_attr": "fullrank_sim_pfintr",
            "to_common_attr": "pfintr_avg"
          },
        ],
      ) \
      .enrich_attr_by_lua(
        import_common_attr = [
          "fountain_rerank_ensemble_power_weight_adjust_ratio_min",
          "fountain_rerank_ensemble_power_weight_adjust_ratio_max",
          "fountain_rerank_ensemble_power_weight_adjust_request_ratio",
          "fountain_rerank_ensemble_power_weight_fullrank_ltr_emp",
          "fountain_rerank_ensemble_power_weight_fullrank_wtr_emp",
          "fountain_rerank_ensemble_power_weight_fullrank_cmtr_emp",
          "fountain_rerank_ensemble_power_weight_fullrank_ptr_emp",
          "fountain_rerank_ensemble_power_weight_fullrank_ftr_emp",
          "fountain_rerank_ensemble_power_weight_fullrank_epstr_emp",
          "fountain_rerank_ensemble_power_weight_fullrank_evtr_emp",
          "fountain_rerank_ensemble_power_weight_fullrank_lvtr_emp",
          "fountain_rerank_ensemble_power_weight_fullrank_fintr_emp",
          "userExpLtr",
          "userExpWtr",
          "userExpCmtr",
          "userExpPtr",
          "userExpFtr",
          "userExpEptr",
          "user_emp_evtr",
          "user_emp_lvtr",
          "user_emp_watch_time",
          "pltr_avg",
          "pwtr_avg",
          "pftr_avg",
          "pcmtr_avg",
          "pptr_avg",
          "pepstr_avg",
          "pevtr_avg",
          "plvtr_avg",
          "pfintr_avg",
        ],
        export_common_attr = [
          "emp_ltr_factor",
          "emp_wtr_factor",
          "emp_cmtr_factor",
          "emp_ptr_factor",
          "emp_ftr_factor",
          "emp_epstr_factor",
          "emp_evtr_factor",
          "emp_lvtr_factor",
          "emp_watchtime_factor",
        ],
        function_for_common = "cal_user_emp_ada_weight_factor",
        lua_script_file = "fountain/rerank/lua/all_lua.lua",
      ) \
      .gen_common_attr_by_lua(
        attr_map = {
          "rerank_es_fullrank_like_score": "emp_ltr_factor * rerank_es_fullrank_like_score",
          "rerank_es_fullrank_follow_score": "emp_wtr_factor * rerank_es_fullrank_follow_score",
          "rerank_es_fullrank_comment_score": "emp_cmtr_factor * rerank_es_fullrank_comment_score",
          "rerank_es_fullrank_profile_score": "emp_ptr_factor * rerank_es_fullrank_profile_score",
          "rerank_es_fullrank_forward_score": "emp_ftr_factor * rerank_es_fullrank_forward_score",
          "rerank_es_fullrank_pepstr_score": "emp_epstr_factor * rerank_es_fullrank_pepstr_score",
          "rerank_es_fullrank_click_score": "emp_evtr_factor * rerank_es_fullrank_click_score",
          "rerank_es_fullrank_lvtr_ori_score": "emp_lvtr_factor * rerank_es_fullrank_lvtr_ori_score",
          "rerank_es_fullrank_pfintr_score": "emp_watchtime_factor * rerank_es_fullrank_pfintr_score",
          "rerank_es_fullrank_like_score_addAndMul": "emp_ltr_factor * rerank_es_fullrank_like_score_addAndMul",
          "rerank_es_fullrank_follow_score_addAndMul": "emp_wtr_factor * rerank_es_fullrank_follow_score_addAndMul",
          "rerank_es_fullrank_comment_score_addAndMul": "emp_cmtr_factor * rerank_es_fullrank_comment_score_addAndMul",
          "rerank_es_fullrank_profile_score_addAndMul": "emp_ptr_factor * rerank_es_fullrank_profile_score_addAndMul",
          "rerank_es_fullrank_pepstr_score_addAndMul": "emp_epstr_factor * rerank_es_fullrank_pepstr_score_addAndMul",
        },
      ) \
      .if_("fountain_enable_rerank_source_related_score_weight_user_custom == 1") \
        .gen_common_attr_by_lua(
          attr_map = {
            "rerank_es_source_related_score_weight": "rerank_es_source_related_score_weight * fullrank_user_intn_rate",
            "rerank_es_source_related_score_weight_mul": "rerank_es_source_related_score_weight_mul * fullrank_user_intn_rate",
          }
        ) \
        .if_("enable_rerank_source_related_score_weight_adjust_by_page == 1") \
          .gen_common_attr_by_lua(
            attr_map = {
              "rerank_es_source_related_score_weight": "rerank_es_source_related_score_weight * fountain_related_weight_page_decay",
              "rerank_es_source_related_score_weight_mul": "rerank_es_source_related_score_weight_mul * fountain_related_weight_page_decay",
            }
          ) \
        .end_() \
      .end_() \
      .if_("fountain_enable_rerank_trend_weight_user_custom_pvtr == 1") \
        .gen_common_attr_by_lua(
          attr_map = {
            "rerank_es_fullrank_watchtime_score": "rerank_es_fullrank_watchtime_score * fountain_watchtime_trend_weight",
            "rerank_es_fullrank_watchtime_score_addAndMul": "rerank_es_fullrank_watchtime_score_addAndMul * fountain_watchtime_trend_weight",
          }
        ) \
      .end_() \
      .if_("fountain_enable_rerank_trend_weight_user_custom_plvtr == 1") \
        .gen_common_attr_by_lua(
          attr_map = {
            "rerank_es_fullrank_lvtr_ori_score": "rerank_es_fullrank_lvtr_ori_score * fountain_watchtime_trend_weight",
            "rerank_es_fullrank_lvtr_ori_score_addAndMul": "rerank_es_fullrank_lvtr_ori_score_addAndMul * fountain_watchtime_trend_weight",
          }
        ) \
      .end_() \
      .if_("fountain_enable_rerank_ten_group_weight_user_custom_pfintr == 1") \
        .gen_common_attr_by_lua(
          attr_map = {
            "rerank_es_fullrank_pfintr_score": "rerank_es_fullrank_pfintr_score * user_group_emp_playtime",
            "rerank_es_fullrank_pfintr_score_addAndMul": "rerank_es_fullrank_pfintr_score_addAndMul * user_group_emp_playtime",
          }
        ) \
      .end_() \
      .if_("fountain_enable_rerank_ten_group_weight_user_custom_pvtr == 1") \
        .gen_common_attr_by_lua(
          attr_map = {
            "rerank_es_fullrank_watchtime_score": "rerank_es_fullrank_watchtime_score * user_group_emp_playtime",
            "rerank_es_fullrank_watchtime_score_addAndMul": "rerank_es_fullrank_watchtime_score_addAndMul * user_group_emp_playtime",
          }
        ) \
      .end_() \
      .if_("fountain_enable_rerank_ten_group_weight_user_custom_plvtr == 1") \
        .gen_common_attr_by_lua(
          attr_map = {
            "rerank_es_fullrank_lvtr_ori_score": "rerank_es_fullrank_lvtr_ori_score * user_group_emp_playtime",
            "rerank_es_fullrank_lvtr_ori_score_addAndMul": "rerank_es_fullrank_lvtr_ori_score_addAndMul * user_group_emp_playtime",
          }
        ) \
      .end_() \
      .duration_longview_adjust_watchtime_queues() \
      .duration_longview_adjust_interaction_queues() \
      .playtime_trend_adjust_watchtime_queues()

    return self

  def rerank_gen_item_attr(self):
    self \
      .copy_item_meta_info(
        save_item_seq_to_attr = "item_seq",
      ) \
      .enrich_attr_by_light_function(
        import_item_attr = [
          {"name": "hetu_tag_level_info__hetu_level_one", "as": "extract_hetu_tag_list"},
        ],
        export_item_attr = [
          {"name": "first_hetu_tag", "as": "first_hetu_l1_tag"},
        ],
        function_name = "ExtractFirstHetuTag",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .enrich_attr_by_light_function(
        import_item_attr = [
          "explore_stat__real_show_count",
          "explore_stat__click_count",
          "explore_stat__like_count",
          "explore_stat__follow_count",
          "explore_stat__forward_count",
          "explore_stat__long_play_count",
          "explore_stat__short_play_count",
          "explore_stat__profile_enter_count",
          "explore_stat__negative_count",
          "explore_stat__comment_count",
          "explore_stat__view_length_sum",
          "is_picture",
        ],
        export_item_attr = [
          "empirical_ctr",
          "empirical_ltr",
          "empirical_wtr",
          "empirical_ftr",
          "empirical_ptr",
          "empirical_htr",
          "empirical_cmtr",
          {"name": "empirical_watch_time", "as": "empirical_watchtime"}
        ],
        function_name = "McCalEmpiricalXtr",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "fountain_rerank_cal_quality_score_sub_coeff", "as": "sub_coeff"},
        ],
        import_item_attr = [
          {"name": "fountain_fullrank_bad_item_similary_score", "as": "score"},
        ],
        export_item_attr = [
          {"name": "score", "as": "rerank_good_item_similary_score"},
        ],
        function_name = "CalExploreDoubleMinusDouble",
        class_name = "ExploreLightFunctionSetV2",
      )

    return self

  def rerank_calc_ensemble_hetu_fusion_score(self):
    self \
      .fountain_calc_ensemble_score(
        save_score_to_attr = "fountain_rerank_hetu_fusion_ensemble_score",
        user_power_calc_v2 = "{{fountain_enable_rerank_hetu_fusion_score_use_power_calc_v2}}",
        user_new_proportion = "{{fountain_enable_rerank_hetu_fusion_score_user_new_proportion}}",
        user_info_ptr_attr = "userInfoPb",
        queue_head_boost_index = "{{fountain_fullrank_hetu_fusion_head_boost_index}}",
        queue_tail_discount_index = "{{fountain_fullrank_hetu_fusion_tail_discount_index}}",
        queues = rerank_hetu_fusion_queues,
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          { "name": "fountain_rerank_hetu_fusion_beam_size", "as": "topk_shuffle_beam_size" },
          { "name": "fountain_rerank_hetu_fusion_shuffle_topk", "as": "topk" },
          { "name": "fountain_enable_rerank_hetu_fusion_padding_hetu", "as": "enable_padding_hetu" },
          { "name": "fountain_rerank_hetu_fusion_calc_score_mode", "as": "calc_mode" },
          { "name": "fountain_rerank_hetu_fusion_calc_score_hetu_weight", "as": "hetu_weight" },
          { "name": "fountain_rerank_hetu_fusion_calc_score_original_weight", "as": "score_weight" },
          { "name": "fountain_rerank_hetu_fusion_user_degree_bucket_weights_str", "as": "user_degree_bucket_weights_str" },
          { "name": "fountain_rerank_hetu_fusion_enable_personalized_adjust", "as": "enable_personalized_adjust" },
          { "name": "fountain_rerank_hetu_fusion_photo_cnt_bucket_weights_str", "as": "photo_cnt_bucket_weights_str" },
          "find_user_active_degree",
        ],
        import_item_attr = [
          { "name": "fountain_rerank_hetu_fusion_ensemble_score", "as": "score" },
          { "name": "first_hetu_l1_tag", "as": "hetu_level_one" },
          { "name": "explore_stat__click_count", "as": "photo_cnt" },
        ],
        export_item_attr = [
          { "name": "output_score_list", "as": "rerank_hetu_fusion_score_list" },
        ],
        function_name = "CalcRerankHetuScoreList",
        class_name = "ExploreLightFunctionSetV2",
      )
    return self

  def rerank_gen_candidate_list(self):
    self \
      .if_("fountain_enable_ensemble_hetu_fusion_score == 1") \
        .rerank_calc_ensemble_hetu_fusion_score() \
      .end_() \
      .if_("fountain_enable_ensemble_fullrank_original_list == 1") \
        .enrich_attr_by_light_function(
          import_item_attr = [
            { "name": "fullrank_ensemble_score", "as": "score" },
          ],
          export_item_attr = [
            { "name": "rank_score", "as": "rerank_fullrank_ensemble_rank_score" },
          ],
          function_name = "CalcRerankSingleRankScore",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .explore_rerank_collect_single_score_candidate(
        candidates = [
          {
            "source": "gen_model",
            "enable": "{{fountain_enable_dpp_gen_model}}",
            "score_attr": "rerank_gen_score_list",
            "list_size": "{{fountain_rerank_gen_model_beam_size}}",
            "save_candidate_to_attr": "rerank_gen_candidate",
          },
          {
            "source": "hetu_fusion",
            "enable": "{{fountain_enable_ensemble_hetu_fusion_score}}",
            "score_attr": "rerank_hetu_fusion_score_list",
            "list_size": "{{fountain_rerank_hetu_fusion_beam_size}}",
            "save_candidate_to_attr": "rerank_hetu_fusion_candidate",
          },
          {
            "source": "fr_ensemble",
            "enable": "{{fountain_enable_ensemble_fullrank_original_list}}",
            "score_attr": "rerank_fullrank_ensemble_rank_score",
            "list_size": "{{fountain_ensemble_fullrank_original_list_beam_size}}",
            "save_candidate_to_attr": "rerank_fr_ensemble_candidate",
          },
          {
            "source": "gen_ar_model",
            "enable": "{{fountain_enable_dpp_gen_ar_model}}",
            "score_attr": "rerank_gen_ar_score_list",
            "list_size": "{{fountain_rerank_gen_ar_model_size}}",
            "save_candidate_to_attr": "rerank_gen_ar_candidate",
          },
          {
            "source": "gen_nar_model",
            "enable": "{{fountain_enable_dpp_gen_nar_model}}",
            "score_attr": "rerank_gen_nar_score_list",
            "list_size": "{{fountain_rerank_gen_nar_model_size}}",
            "save_candidate_to_attr": "rerank_gen_nar_candidate",
          },
          {
            "source": "gen_rankmixer_model",
            "enable": "{{fountain_enable_dpp_gen_rankmixer_model}}",
            "score_attr": "rerank_gen_rankmixer_score_list",
            "list_size": "{{fountain_rerank_gen_rankmixer_model_size}}",
            "save_candidate_to_attr": "rerank_gen_rankmixer_candidate",
          },
          {
            "source": "gen_ar_pinrec_model",
            "enable": "{{fountain_enable_dpp_gen_ar_pinrec_model}}",
            "score_attr": "rerank_gen_ar_pinrec_score_list",
            "list_size": "{{fountain_rerank_gen_ar_pinrec_model_size}}",
            "save_candidate_to_attr": "rerank_gen_ar_pinrec_candidate",
          },
        ],
      ) \
      .calc_user_custom_ensemble_weight() \
      .if_("enable_fountain_rerank_es_topk_resort == 1 or enable_fountain_rerank_es_topk_resort_mul == 1") \
        .calc_weighted_sum(
          formula_version = 0,
          channels = [
            { "name": "fullrank_sim_pevtr", "weight": "{{fountain_rerank_es_topk_resort_pevtr_weight}}" },
            { "name": "fullrank_sim_pcpr", "weight": "{{fountain_rerank_es_topk_resort_pcpr_weight}}" },
            { "name": "fullrank_sim_pfintr", "weight": "{{fountain_rerank_es_topk_resort_pfintr_weight}}" },
            { "name": "fullrank_sim_pvtr", "weight": "{{fountain_rerank_es_topk_resort_pvtr_weight}}" },
            { "name": "fullrank_sim_plvtr", "weight": "{{fountain_rerank_es_topk_resort_plvtr_weight}}" },
            { "name": "fullrank_sim_pltr", "weight": "{{fountain_rerank_es_topk_resort_pltr_weight}}" },
            { "name": "fullrank_sim_pcmtr", "weight": "{{fountain_rerank_es_topk_resort_pcmtr_weight}}" },
            { "name": "fullrank_sim_pwtr", "weight": "{{fountain_rerank_es_topk_resort_pwtr_weight}}" },
            { "name": "fullrank_sim_psvr", "weight": "{{fountain_rerank_es_topk_resort_psvr_weight}}" },  # 2026-1-8 by dengyingjie03
          ],
          output_item_attr = "rerank_es_topk_resort_score",
        ) \
      .end_() \
      .fountain_rerank_es_playtime_adjust() \
      .explore_rerank_gen_random_ensemble_candidate(
        # 最后 list 数量等于: es 扰动数量 + topk shuffle 数量 + topk resort 数量，乘法 es 同理
        source = "es_add",
        target_candidate_num = "{{rerank_dpp_diversity_max_sequence_num}}",
        ensemble_algorithm = 1,
        queues = es_add_queues,
        save_candidate_to_attr = "rerank_es_add_candidate",
        enable_topk_shuffle = "{{enable_fountain_rerank_es_topk_shuffle}}",
        topk_shuffle_window_size = "{{fountain_rerank_es_topk_shuffle_window_size}}",
        enable_topk_resort = "{{enable_fountain_rerank_es_topk_resort}}",
        topk_resort_window_size = "{{fountain_rerank_es_topk_resort_window_size}}",
        topk_shuffle_target_candidate_num = "{{fountain_rerank_es_topk_shuffle_target_candidate_num}}",
        topk_resort_target_candidate_num = "{{fountain_rerank_es_topk_resort_target_candidate_num}}",
        topk_resort_score_attr = "rerank_es_topk_resort_score",
      ) \
      .explore_rerank_gen_random_ensemble_candidate(
        source = "es_mul",
        target_candidate_num = "{{rerank_dpp_sequence_num_multiple}}",
        ensemble_algorithm = 2,
        queues = es_mul_queues,
        save_candidate_to_attr = "rerank_es_mul_candidate",
        enable_topk_shuffle = "{{enable_fountain_rerank_es_topk_shuffle_mul}}",
        topk_shuffle_window_size = "{{fountain_rerank_es_topk_shuffle_window_size_mul}}",
        enable_topk_resort = "{{enable_fountain_rerank_es_topk_resort_mul}}",
        topk_resort_window_size = "{{fountain_rerank_es_topk_resort_window_size_mul}}",
        topk_shuffle_target_candidate_num = "{{fountain_rerank_es_topk_shuffle_target_candidate_num_mul}}",
        topk_resort_target_candidate_num = "{{fountain_rerank_es_topk_resort_target_candidate_num_mul}}",
        topk_resort_score_attr = "rerank_es_topk_resort_score",
      )
      
    return self

  def rerank_gen_list(self):
    self \
      .if_("fountain_enable_dpp_gen_list == 1") \
        .if_("fountain_enable_user_need_break_cocoon_rerank == 1 and user_need_break_cocoon_flag == 1") \
          .enrich_attr_by_light_function(
            import_common_attr = [
              {"name": "fountain_rerank_dpp_relevance_score_theta", "as": "value"},
              {"name": "fountain_rerank_dpp_relevance_score_theta_user_need_break_cocoon_adjust_coef", "as": "weight"},
            ],
            export_common_attr = [
              {"name": "new_value", "as": "fountain_rerank_dpp_relevance_score_theta"},
            ],
            function_name = "CalExploreDoubleMultiDouble",
            class_name = "ExploreLightFunctionSetV2",
          ) \
        .end_() \
        .if_("fountain_rerank_dpp_enable_diversity_increase_by_page == 1") \
          .enrich_attr_by_light_function(
            import_common_attr = [
              {"name": "fountain_rerank_dpp_relevance_score_theta", "as": "theta"},
              {"name": "fountain_rerank_dpp_diversity_increase_by_page_weight", "as": "page_diversity_weight"},
              {"name": "page", "as": "page_index"},
            ],
            export_common_attr = [
              {"name": "new_theta", "as": "fountain_rerank_dpp_relevance_score_theta"},
            ],
            function_name = "FountainRerankDppDiversityIncreaseByPage",
            class_name = "ExploreLightFunctionSetV2",
          ) \
        .end_() \
        .if_("fountain_rerank_enable_import_hetu_emb_v4 == 1 and fountain_rerank_dpp_diversity_enable_hetu_emb_v4 == 1") \
          .calc_explore_rerank_gen_list_by_dpp("rerank_hetu_emb", 128) \
        .else_() \
          .calc_explore_rerank_gen_list_by_dpp() \
        .end_() \
      .end_() \
      .if_("fountain_enable_ssd_gen_list == 1") \
        .if_("fountain_rerank_ssd_enable_diversity_increase_by_page == 1") \
          .enrich_attr_by_light_function(
            import_common_attr = [
              {"name": "fountain_ssd_rank_score_weight", "as": "ssd_diversity_weight"},
              {"name": "fountain_rerank_ssd_diversity_increase_by_page_weight", "as": "page_diversity_weight"},
              {"name": "page", "as": "page_index"}
            ],
            export_common_attr = [
              {"name": "new_ssd_diversity_weight", "as": "fountain_ssd_rank_score_weight"},
            ],
            function_name = "FountainRerankSsdDiversityIncreaseByPage",
            class_name = "ExploreLightFunctionSetV2",
          ) \
        .end_() \
        .explore_rerank_gen_list_by_ssd(
          source = "ssd",
          candidate_attrs = [
            "rerank_es_add_candidate",
            "rerank_es_mul_candidate",
            "rerank_fr_ensemble_candidate",
            "rerank_hetu_candidate",
            "rerank_hetu_fusion_candidate",
            "rerank_gen_candidate",
            "rerank_gen_ar_candidate",
            "rerank_gen_nar_candidate",
            "rerank_gen_ar_pinrec_candidate",
            "rerank_gen_rankmixer_candidate"
          ],
          list_size = "{{fountain_rerank_dpp_diversity_list_size}}",
          hetu_l1_attr = "first_hetu_l1_tag",
          max_hetu_l1_num = "{{fountain_rerank_ssd_max_hetu1_regular_num}}",
          rank_score_weight = "{{fountain_ssd_rank_score_weight}}",
          mc_emb_attr = "rerank_mc_emb",
          mc_emb_dim = 128,
          save_list_to_attr = "ssd_list_collection",
        ) \
      .end_() \
      .if_("fountain_rerank_enable_import_hetu_emb_v4 == 1 and fountain_rerank_enable_select_list_hetu_emb_v4 == 1") \
        .explore_rerank_select_list(
          list_attrs = [
            "dpp_list_collection",
            "ssd_list_collection",
          ],
          enable_set_dedup = "{{fountain_dpp_use_set_filter}}",
          max_set_dedup_cnt = "{{fountain_dpp_set_max_filter_cnt}}",
          mmu_emb_attr = "rerank_hetu_emb",
          mmu_emb_dim = 128,
          limit_num = "{{fountain_ssd_filter_final_cnt}}",
        ) \
      .else_() \
        .explore_rerank_select_list(
          list_attrs = [
            "dpp_list_collection",
            "ssd_list_collection",
          ],
          enable_set_dedup = "{{fountain_dpp_use_set_filter}}",
          max_set_dedup_cnt = "{{fountain_dpp_set_max_filter_cnt}}",
          mmu_emb_attr = "rerank_mmu_emb",
          mmu_emb_dim = 64,
          limit_num = "{{fountain_ssd_filter_final_cnt}}",
        ) \
      .end_() \
  
    return self

  def rerank_list_predict(self):
    return self \
      .rerank_retrieve_list_index() \
      .if_("enable_fountain_rerank_flash_eval_model_predict == 1") \
        .rerank_flash_eval_model() \
      .else_() \
        .rerank_eval_list_predict() \
      .end_() \
      .rerank_eval_sort() \

  def rerank_retrieve_list_index(self):
    self \
      .explore_rerank_list_retriever(
        list_attrs = [
          "dpp_list_collection",
          "ssd_list_collection",
        ],
        save_list_to_attr = "origin_result_list",
        save_key_list_to_attr = "origin_result_key_list",
        save_list_source_to_attr = "list_source",
        save_candidate_list_source_to_attr = "candiate_list_source",
        reset_existing_item_attrs = False,
        item_table = "rerank_list",
      ) \
      .pack_item_attr(
        item_source = {
          "reco_results": True,
        },
        mappings = [
          {
            "from_item_attr": "origin_result_key_list",
            "to_common_attr": "rerank_list_item_key_flat_list",
          },
        ],
        item_table = "rerank_list",
      ) \
      .enrich_attr_by_light_function(
        item_list_from_attr = "rerank_list_item_key_flat_list",
        export_item_attr = [
          "item_seq",
        ],
        function_name = "EmptyFunction",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .pack_item_attr(
        item_source = {
          "reco_results": False,
          "common_attr": [
            "rerank_list_item_key_flat_list",
          ],
        },
        mappings = [
          {
            "from_item_attr": "item_seq",
            "to_common_attr": "rerank_list_item_idx_flat_list",
          },
        ],
      ) \

    return self

  # flash eval 架构下将融分步骤放到 infer 服务 , 输入候选集 + list index , 直接返回最终 list score
  def rerank_flash_eval_model(self):
    return self \
      .explore_custom_trim_user_info(
        user_info_attr = "userInfo",
        save_trimed_user_info_to_attr = "rerank_deep_ltr_trimmed_user_info",
        trim_user_info = rerank_flash_eval_model_send_user_feas,
      ) \
      .delegate_enrich(
        kess_service = "{{fountain_rerank_flash_eval_model_kess_service}}",
        recv_common_attrs = [
          "eval_list_scores"
        ],
        timeout_ms = 100,
        send_item_attrs = rerank_flash_eval_model_send_item_feas,
        send_common_attrs = rerank_flash_eval_model_send_common_feas,
        request_type = "{{fountain_rerank_flash_eval_model_request_type}}",
        partition_size = "{{fountain_rerank_flash_eval_model_partition_size}}",
      ) \
      .dispatch_common_attr(
        from_common_attr = "eval_list_scores",
        to_item_attr = "es_score",
        item_table = "rerank_list",
      )

  @apply(item_table = "rerank_list")
  def rerank_eval_list_predict(self):
    self \
      .explore_common_user_feature_enricher(
        user_info_attr = "userInfoPb",
        user_action_list_long_version_attr = "uActionListLongVersion",
        user_view_hetu1_attr = "uViewHetu1ListV1",
        user_view_hetu2_attr = "uViewHetu2ListV1",
        ft_ev_pids_attr = "uViewEffectivePidListV1",
        ft_lv_pids_attr = "uViewLongPidListV1",
        ft_sv_pids_attr = "uViewShortPidListV1",
      ) \
      .explore_rerank_gen_list_feature(
        origin_result_table = "",
        origin_result_list_attr = "origin_result_list",
        origin_result_list_size = 6,
        feature_attrs = rerank_features_v3,
      ) \
      .common_predict(
        kess_service = "{{fountain_rerank_kess_service_v2_produce_v4}}",
        timeout_ms = 100,
        loss_function_name = ["pos" + str(i) for i in range(6)] + ["next" + str(i) for i in range(6)] + ["play" + str(i) for i in range(6)],
        loss_default_value = 0.0,
        extra_common_attrs = [
          "uViewHetu1ListV1",
          "uViewHetu2ListV1",
          "uViewEffectivePidListV1",
          "uViewLongPidListV1",
          "uViewShortPidListV1",
          "basic_info_gender",
        ],
        item_attrs = gen_photo_features_for_all_position_v3(6),
        attr_name_transform_map = {
          "featureUId": "uId",
          "featureDeviceId": "dId",
          "featureAgeSegment": "uBasicAge",
          "featureCityId": "uCityId",
          "featureFountainProfileClickPidList": "featureFountainProfileClikPidList",
          "featureUserProfileV1ClickPidList": "uClickPhotoList",
          "featureUserProfileV1FollowPidList": "uFollowPhotoList",
          "featureUserProfileV1LikePidList": "uLikePhotoList",
          "featureUserProfileV1CommentPidList": "uCommentPhotoList",
          "featureRealtimeClickList": "uRealtimeClickList",
          "featureRealtimeFollowList": "uRealtimeFollowList",
          "featureRealtimeLikeList": "uRealtimeLikeList",
          "featureRealtimeForwardList": "uRealtimeForwardList",
          "basic_info_gender": "uBasicGender",
        },
      ) \
      .get_kconf_params(
        kconf_configs = [
          {
            "kconf_key": "{{fountain_rr_play_watch_table_kconf}}",
            "value_type": "json",
            "json_path": "durationQuantile",
            "export_common_attr": "rerank_duration_buckets",
          },
          {
            "kconf_key": "{{fountain_rr_play_watch_table_kconf}}",
            "value_type": "json",
            "json_path": "playQ0",
            "export_common_attr": "rerank_wtd_table_0",
          },
          {
            "kconf_key": "{{fountain_rr_play_watch_table_kconf}}",
            "value_type": "json",
            "json_path": "playQ1",
            "export_common_attr": "rerank_wtd_table_1",
          },
          {
            "kconf_key": "{{fountain_rr_play_watch_table_kconf}}",
            "value_type": "json",
            "json_path": "playQ2",
            "export_common_attr": "rerank_wtd_table_2",
          },
          {
            "kconf_key": "{{fountain_rr_play_watch_table_kconf}}",
            "value_type": "json",
            "json_path": "playQ3",
            "export_common_attr": "rerank_wtd_table_3",
          },
          {
            "kconf_key": "{{fountain_rr_play_watch_table_kconf}}",
            "value_type": "json",
            "json_path": "playQ4",
            "export_common_attr": "rerank_wtd_table_4",
          },
          {
            "kconf_key": "{{fountain_rr_play_watch_table_kconf}}",
            "value_type": "json",
            "json_path": "playQ5",
            "export_common_attr": "rerank_wtd_table_5",
          },
          {
            "kconf_key": "{{fountain_rr_play_watch_table_kconf}}",
            "value_type": "json",
            "json_path": "playQ6",
            "export_common_attr": "rerank_wtd_table_6",
          },
          {
            "kconf_key": "{{fountain_rr_play_watch_table_kconf}}",
            "value_type": "json",
            "json_path": "playQ7",
            "export_common_attr": "rerank_wtd_table_7",
          },
          {
            "kconf_key": "{{fountain_rr_play_watch_table_kconf}}",
            "value_type": "json",
            "json_path": "playQ8",
            "export_common_attr": "rerank_wtd_table_8",
          },
        ],
      ) \
      .set_attr_value(
        no_overwrite = False,
        common_attrs = [
          {
            "name": "rerank_origin_score_attrs",
            "type": "string_list",
            "value": ["play" + str(i) for i in range(6)],
          },
          {
            "name": "rerank_duration_attrs",
            "type": "string_list",
            "value": ["pDurationMs_idx" + str(i) for i in range(6)],
          },
          {
            "name": "rerank_wtd_table_atts",
            "type": "string_list",
            "value": ["rerank_wtd_table_" + str(i) for i in range(9)],
          },
          {
            "name": "rerank_wtd_score_attrs",
            "type": "string_list",
            "value": ["wtd" + str(i) for i in range(6)],
          },
          {
            "name": "rerank_list_es_score_attrs",
            "type": "string_list",
            "value": ["pos_score", "next_score", "wtd_score"],
          },
          {
            "name": "rerank_list_es_weight_attrs",
            "type": "string_list",
            "value": ["rerank_list_pos_weight", "rerank_list_next_weight", "rerank_list_wtd_weight"],
          },
          {
            "name": "rerank_list_pos_weight",
            "type": "double",
            "value": 1.0,
          },
          {
            "name": "rerank_list_next_weight",
            "type": "double",
            "value": 1.0,
          },
          {
            "name": "rerank_list_wtd_weight",
            "type": "double",
            "value": 0.6,
          },
        ],
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          { "name": "rerank_duration_buckets", "as": "duration_buckets" },
          { "name": "rerank_origin_score_attrs", "as": "origin_score_attrs" },
          { "name": "rerank_duration_attrs", "as": "duration_attrs" },
          { "name": "rerank_wtd_table_atts", "as": "trans_score_list_attrs" },
          { "name": "rerank_wtd_score_attrs", "as": "wtd_score_attrs" },
        ] + ["rerank_wtd_table_" + str(i) for i in range(9)],
        import_item_attr = ["play" + str(i) for i in range(6)] + ["pDurationMs_idx" + str(i) for i in range(6)],
        export_item_attr = ["wtd" + str(i) for i in range(6)],
        function_name = "CalcRerankWtdScore",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .calc_weighted_sum(
        channels = [{ "name": "pos" + str(i), "weight": 1.0 } for i in range(6)],
        output_item_attr = "pos_score",
      ) \
      .calc_weighted_sum(
        channels = [{ "name": "next" + str(i), "weight": 1.0 } for i in range(6)],
        output_item_attr = "next_score",
      ) \
      .calc_weighted_sum(
        channels = [{ "name": "wtd" + str(i), "weight": 1.0 } for i in range(6)],
        output_item_attr = "wtd_score",
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          { "name": "rerank_list_es_score_attrs", "as": "score_attrs" },
          { "name": "rerank_list_es_weight_attrs", "as": "weight_attrs" },
          "rerank_list_pos_weight",
          "rerank_list_next_weight",
          "rerank_list_wtd_weight",
        ],
        import_item_attr = [
          "pos_score",
          "next_score",
          "wtd_score",
        ],
        export_item_attr = [
          "es_score",
        ],
        function_name = "CalcRerankListESScore",
        class_name = "ExploreLightFunctionSetV2",
      ) \

    return self
      
  @apply(item_table = "rerank_list")
  def rerank_eval_sort(self):
    self \
      .pack_item_attr(
        item_source = {
          "reco_results": True,
        },
        mappings = [
          {
            "aggregator": "concat",
            "from_item_attr": "es_score",
            "to_common_attr": "rerank_list_score_list",
          },
        ],
      ) \
      .sort(
        score_from_attr = "es_score",
      ) \
      .explore_rerank_dispatch_list_score(
        origin_result_table = "",
        origin_result_list_attr = "origin_result_list",
        save_list_score_to_attr = "rerank_list_score",
      ) \
      .pack_item_attr(
        item_source = {
          "reco_results": True,
        },
        mappings = [
          {
            "aggregator": "copy",
            "from_item_attr": "list_source",
            "to_common_attr": "selected_list_source",
          },
          {
            "aggregator": "copy",
            "from_item_attr": "candiate_list_source",
            "to_common_attr": "selected_candiate_list_source",
          },
          {
            "aggregator": "avg",
            "from_item_attr": "es_score",
            "to_common_attr": "rerank_list_score_avg",
          },
          {
            "aggregator": "max",
            "from_item_attr": "es_score",
            "to_common_attr": "rerank_list_score_max",
          },
        ],
      )

    return self

  def rerank_gen_reason_metrics(self):
    candidate_list_source_list = [
      "es_add",
      "es_mul",
      "hetu",
      "hetu_fusion",
      "gen_model",
      "fr_ensemble",
      "gen_ar_model",
      "gen_nar_model",
      "gen_rankmixer_model",
      "gen_ar_pinrec_model"
    ]

    list_source_list = [
      "dpp",
      "ssd",
    ]

    self \
      .set_attr_value(
        common_attrs = [
          {
            "name": "candidate_list_source_list", 
            "type": "string_list",
            "value": candidate_list_source_list,
          },
          {
            "name": "list_source_list", 
            "type": "string_list",
            "value": list_source_list,
          },
        ],
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          "selected_candiate_list_source",
          "selected_list_source",
          "candidate_list_source_list",
          "list_source_list",
        ],
        export_common_attr = [
          "rerank_list_source_" + candidate_list_source + "_" + list_source
          for candidate_list_source in candidate_list_source_list
            for list_source in list_source_list
        ],
        function_name = "GenRerankListReasonMetrics",
        class_name = "ExploreLightFunctionSetV2",
      )

    return self

  def adjust_forward_social_params(self):
    self \
    .if_("fountain_rerank_enable_adjust_forward_social_params > 0 and bid_follow_num > 0") \
      .gen_common_attr_by_lua(
      attr_map={
        "rerank_es_fullrank_forward_score": "rerank_es_fullrank_forward_score * dpp_fountain_rerank_gen_seed_ensemble_fullrank_forward_score_social_coeff",
      }
    ) \
    .end_()
    return self
  
  def duration_longview_adjust_watchtime_queues(self):
    self \
    .if_("fountain_enable_rerank_duration_longview_adjust_weight_user_custom_pfintr == 1") \
      .gen_common_attr_by_lua(
        attr_map = {
          "rerank_es_fullrank_pfintr_score": "rerank_es_fullrank_pfintr_score * fountain_duration_longview_adjust_weight",
          "rerank_es_fullrank_pfintr_score_addAndMul": "rerank_es_fullrank_pfintr_score_addAndMul * fountain_duration_longview_adjust_weight",
        }
      ) \
    .end_() \
    .if_("fountain_enable_rerank_duration_longview_adjust_weight_user_custom_pvtr == 1") \
      .gen_common_attr_by_lua(
        attr_map = {
          "rerank_es_fullrank_watchtime_score": "rerank_es_fullrank_watchtime_score * fountain_duration_longview_adjust_weight",
          "rerank_es_fullrank_watchtime_score_addAndMul": "rerank_es_fullrank_watchtime_score_addAndMul * fountain_duration_longview_adjust_weight",
        }
      ) \
    .end_() \
    .if_("fountain_enable_rerank_duration_longview_adjust_weight_user_custom_plvtr == 1") \
      .gen_common_attr_by_lua(
        attr_map = {
          "rerank_es_fullrank_lvtr_ori_score": "rerank_es_fullrank_lvtr_ori_score * fountain_duration_longview_adjust_weight",
          "rerank_es_fullrank_lvtr_ori_score_addAndMul": "rerank_es_fullrank_lvtr_ori_score_addAndMul * fountain_duration_longview_adjust_weight",
        }
      ) \
    .end_() \
    .if_("fountain_enable_rerank_duration_longview_adjust_weight_user_custom_pcpr == 1") \
      .gen_common_attr_by_lua(
        attr_map = {
          "rerank_es_fullrank_pcpr_score": "rerank_es_fullrank_pcpr_score * fountain_duration_longview_adjust_weight",
          "rerank_es_fullrank_pcpr_score_addAndMul": "rerank_es_fullrank_pcpr_score_addAndMul * fountain_duration_longview_adjust_weight",
        }
      ) \
    .end_()
    return self
  
  def playtime_trend_adjust_watchtime_queues(self):
    self \
    .if_("fountain_enable_rerank_playtime_trend_adjust_weight_user_custom_pfintr == 1") \
      .gen_common_attr_by_lua(
        attr_map = {
          "rerank_es_fullrank_pfintr_score": "rerank_es_fullrank_pfintr_score * fountain_playtime_trend_adjust_weight",
          "rerank_es_fullrank_pfintr_score_addAndMul": "rerank_es_fullrank_pfintr_score_addAndMul * fountain_playtime_trend_adjust_weight",
        }
      ) \
    .end_() \
    .if_("fountain_enable_rerank_playtime_trend_adjust_weight_user_custom_pvtr == 1") \
      .gen_common_attr_by_lua(
        attr_map = {
          "rerank_es_fullrank_watchtime_score": "rerank_es_fullrank_watchtime_score * fountain_playtime_trend_adjust_weight",
          "rerank_es_fullrank_watchtime_score_addAndMul": "rerank_es_fullrank_watchtime_score_addAndMul * fountain_playtime_trend_adjust_weight",
        }
      ) \
    .end_() \
    .if_("fountain_enable_rerank_playtime_trend_adjust_weight_user_custom_plvtr == 1") \
      .gen_common_attr_by_lua(
        attr_map = {
          "rerank_es_fullrank_lvtr_ori_score": "rerank_es_fullrank_lvtr_ori_score * fountain_playtime_trend_adjust_weight",
          "rerank_es_fullrank_lvtr_ori_score_addAndMul": "rerank_es_fullrank_lvtr_ori_score_addAndMul * fountain_playtime_trend_adjust_weight",
        }
      ) \
    .end_() \
    .if_("fountain_enable_rerank_playtime_trend_adjust_weight_user_custom_pcpr == 1") \
      .gen_common_attr_by_lua(
        attr_map = {
          "rerank_es_fullrank_pcpr_score": "rerank_es_fullrank_pcpr_score * fountain_playtime_trend_adjust_weight",
          "rerank_es_fullrank_pcpr_score_addAndMul": "rerank_es_fullrank_pcpr_score_addAndMul * fountain_playtime_trend_adjust_weight",
        }
      ) \
    .end_()
    return self
  
  def duration_longview_adjust_interaction_queues(self):
    self \
    .if_("fountain_enable_rerank_duration_longview_adjust_weight_like_score == 1") \
      .gen_common_attr_by_lua(
        attr_map = {
          "rerank_es_fullrank_like_score": "rerank_es_fullrank_like_score / fountain_duration_longview_adjust_weight",
          "rerank_es_fullrank_like_score_addAndMul": "rerank_es_fullrank_like_score_addAndMul / fountain_duration_longview_adjust_weight",
        }
      ) \
    .end_() \
    .if_("fountain_enable_rerank_duration_longview_adjust_weight_follow_score == 1") \
      .gen_common_attr_by_lua(
        attr_map = {
          "rerank_es_fullrank_follow_score": "rerank_es_fullrank_follow_score / fountain_duration_longview_adjust_weight",
          "rerank_es_fullrank_follow_score_addAndMul": "rerank_es_fullrank_follow_score_addAndMul / fountain_duration_longview_adjust_weight",
        }
      ) \
    .end_() \
    .if_("fountain_enable_rerank_duration_longview_adjust_weight_comment_score == 1") \
      .gen_common_attr_by_lua(
        attr_map = {
          "rerank_es_fullrank_comment_score": "rerank_es_fullrank_comment_score / fountain_duration_longview_adjust_weight",
          "rerank_es_fullrank_comment_score_addAndMul": "rerank_es_fullrank_comment_score_addAndMul / fountain_duration_longview_adjust_weight",
        }
      ) \
    .end_() \
    .if_("fountain_enable_rerank_duration_longview_adjust_weight_forward_score == 1") \
      .gen_common_attr_by_lua(
        attr_map = {
          "rerank_es_fullrank_forward_score": "rerank_es_fullrank_forward_score / fountain_duration_longview_adjust_weight",
          "fountain_rerank_es_fullrank_forward_weight_mul": "fountain_rerank_es_fullrank_forward_weight_mul / fountain_duration_longview_adjust_weight",
        }
      ) \
    .end_() \
    .if_("fountain_enable_rerank_duration_longview_adjust_weight_collect_score == 1") \
      .gen_common_attr_by_lua(
        attr_map = {
          "rerank_es_fullrank_collect_score": "rerank_es_fullrank_collect_score / fountain_duration_longview_adjust_weight",
          "rerank_es_fullrank_collect_score_addAndMul": "rerank_es_fullrank_collect_score_addAndMul / fountain_duration_longview_adjust_weight",
        }
      ) \
    .end_()
    return self

  # 封装dpp生成序列的函数,减少代码冗余
  def calc_explore_rerank_gen_list_by_dpp(self, emb_attr = "rerank_mmu_emb", dim_size = 64):
    return self \
      .explore_rerank_gen_list_by_dpp(
        source = "dpp",
        candidate_attrs = [
          "rerank_es_add_candidate",
          "rerank_es_mul_candidate",
          "rerank_fr_ensemble_candidate",
          "rerank_hetu_candidate",
          "rerank_hetu_fusion_candidate",
          "rerank_gen_candidate",
          "rerank_gen_ar_candidate",
          "rerank_gen_nar_candidate",
          "rerank_gen_ar_pinrec_candidate",
          "rerank_gen_rankmixer_candidate"
        ],
        list_size = "{{fountain_rerank_dpp_diversity_list_size}}",
        beam_size = "{{fountain_dpp_beam_size}}",
        sim_matrix_norm_type = "{{fountain_rerank_dpp_sim_matrix_norm_type}}",
        sim_matrix_mix_param = "{{fountain_rerank_dpp_sim_matrix_alpha}}",
        relevance_score_theta = "{{fountain_rerank_dpp_relevance_score_theta}}",
        hetu_l1_attr = "first_hetu_l1_tag",
        max_hetu_l1_num = "{{fountain_rerank_dpp_max_hetu1_regular_num}}",
        mmu_emb_attr = emb_attr,
        mmu_emb_dim = dim_size,
        mc_emb_attr = "rerank_mc_emb",
        mc_emb_dim = 128,
        save_list_to_attr = "dpp_list_collection",
      )

  def fountain_rerank_gen_model_beam(self):
    return self \
      .explore_custom_trim_user_info(
        user_info_attr = "userInfo",
        save_trimed_user_info_to_attr = "rerank_deep_ltr_trimmed_user_info",
        trim_user_info = [
          "active_days",
          "basic_info.age_segment",
          "location.city_id",
          "location.region_type",
          "client_id",
          "device_id",
          "gender",
          "infer_gender",
          "true_gender",
          "request_location.poi_type",
          "request_location.province_id",
          "request_location.city_id",
          "visit_mod",
          "user_profile.exp_stat.exp_click",
          "user_profile.exp_stat.exp_like",
          "user_profile.exp_stat.exp_follow",
          "user_profile.exp_stat.exp_realshow",
          "user_profile.exp_stat.exp_long_view",
          "user_profile.user_level",
          "realtime_click_list",
          "realtime_follow_list",
          "realtime_forward_list",
          "realtime_like_list",
          "fountain_reco_user_profile.follow_list.author_id",
          "fountain_reco_user_profile.follow_list.photo_id",
          "fountain_reco_user_profile.like_list.author_id",
          "fountain_reco_user_profile.like_list.photo_id",
          "fountain_reco_user_profile.video_play_stat.photo_id",
          "fountain_reco_user_profile.video_play_stat.author_id",
          "fountain_reco_user_profile.video_play_stat.video_duration",
          "fountain_reco_user_profile.video_play_stat.playing_time",
        ],
      ) \
      .delegate_enrich(
        kess_service = "{{fountain_rerank_gen_model_beam_kess_service}}",
        recv_item_attrs = [
          {"name": "rerank_gen_score", "as": "rerank_gen_score_list"}
        ],
        timeout_ms = 50,
        send_item_attrs = rerank_gen_model_send_item_feas,
        send_common_attrs = rerank_gen_model_send_common_feas,
        request_type = "{{fountain_rerank_gen_model_request_type}}",
        partition_size = "{{fountain_rerank_gen_model_partition_size}}",
      ) \

  def fountain_rerank_gen_model_nar_beam(self):
    return self \
      .explore_custom_trim_user_info(
        user_info_attr = "userInfo",
        save_trimed_user_info_to_attr = "rerank_deep_ltr_trimmed_user_info",
        trim_user_info = [
          "active_days",
          "basic_info.age_segment",
          "location.city_id",
          "location.region_type",
          "client_id",
          "device_id",
          "gender",
          "infer_gender",
          "true_gender",
          "request_location.poi_type",
          "request_location.province_id",
          "request_location.city_id",
          "visit_mod",
          "user_profile.exp_stat.exp_click",
          "user_profile.exp_stat.exp_like",
          "user_profile.exp_stat.exp_follow",
          "user_profile.exp_stat.exp_realshow",
          "user_profile.exp_stat.exp_long_view",
          "user_profile.user_level",
          "realtime_click_list",
          "realtime_follow_list",
          "realtime_forward_list",
          "realtime_like_list",
          "fountain_reco_user_profile.follow_list.author_id",
          "fountain_reco_user_profile.follow_list.photo_id",
          "fountain_reco_user_profile.like_list.author_id",
          "fountain_reco_user_profile.like_list.photo_id",
          "fountain_reco_user_profile.video_play_stat.photo_id",
          "fountain_reco_user_profile.video_play_stat.author_id",
          "fountain_reco_user_profile.video_play_stat.video_duration",
          "fountain_reco_user_profile.video_play_stat.playing_time",
        ],
      ) \
      .delegate_enrich(
        kess_service = "{{fountain_rerank_gen_model_beam_nar_kess_service}}",
        recv_item_attrs = [
          {"name": "rerank_gen_score", "as": "rerank_gen_nar_score_list"}
        ],
        timeout_ms = 50,
        send_item_attrs = rerank_gen_model_send_item_feas,
        send_common_attrs = rerank_gen_model_send_common_feas,
        request_type = "{{fountain_rerank_gen_model_nar_request_type}}",
        partition_size = "{{fountain_rerank_gen_model_nar_partition_size}}",
      ) \

  def fountain_rerank_gen_model_ar_beam(self):
    return self \
      .explore_custom_trim_user_info(
        user_info_attr = "userInfo",
        save_trimed_user_info_to_attr = "rerank_deep_ltr_trimmed_user_info",
        trim_user_info = [
          "active_days",
          "basic_info.age_segment",
          "location.city_id",
          "location.region_type",
          "client_id",
          "device_id",
          "gender",
          "infer_gender",
          "true_gender",
          "request_location.poi_type",
          "request_location.province_id",
          "request_location.city_id",
          "visit_mod",
          "user_profile.exp_stat.exp_click",
          "user_profile.exp_stat.exp_like",
          "user_profile.exp_stat.exp_follow",
          "user_profile.exp_stat.exp_realshow",
          "user_profile.exp_stat.exp_long_view",
          "user_profile.user_level",
          "realtime_click_list",
          "realtime_follow_list",
          "realtime_forward_list",
          "realtime_like_list",
          "fountain_reco_user_profile.follow_list.author_id",
          "fountain_reco_user_profile.follow_list.photo_id",
          "fountain_reco_user_profile.like_list.author_id",
          "fountain_reco_user_profile.like_list.photo_id",
          "fountain_reco_user_profile.video_play_stat.photo_id",
          "fountain_reco_user_profile.video_play_stat.author_id",
          "fountain_reco_user_profile.video_play_stat.video_duration",
          "fountain_reco_user_profile.video_play_stat.playing_time",
        ],
      ) \
      .delegate_enrich(
        kess_service = "{{fountain_rerank_gen_model_beam_ar_kess_service}}",
        recv_item_attrs = [
          {"name": "rerank_gen_score", "as": "rerank_gen_ar_score_list"}
        ],
        timeout_ms = 50,
        send_item_attrs = rerank_gen_model_send_item_feas,
        send_common_attrs = rerank_gen_model_send_common_feas,
        request_type = "{{fountain_rerank_gen_model_ar_request_type}}",
        partition_size = "{{fountain_rerank_gen_model_ar_partition_size}}",
      )
  
  def fountain_rerank_gen_model_rankmixer(self):
    return self \
      .explore_custom_trim_user_info(
        user_info_attr = "userInfo",
        save_trimed_user_info_to_attr = "rerank_deep_ltr_trimmed_user_info",
        trim_user_info = rerank_gen_model_send_user_feas,
      ) \
      .delegate_enrich(
        kess_service = "{{fountain_rerank_gen_model_beam_rankmixer_kess_service}}",
        recv_item_attrs = [
          {"name": "rerank_gen_score", "as": "rerank_gen_rankmixer_score_list"}
        ],
        timeout_ms = 50,
        send_item_attrs = rerank_gen_model_send_item_feas,
        send_common_attrs = rerank_gen_model_send_common_feas,
        request_type = "{{fountain_rerank_gen_model_rankmixer_request_type}}",
        partition_size = "{{fountain_rerank_gen_model_rankmixer_partition_size}}",
      )

  def fountain_rerank_gen_model_ar_pinrec(self):
    return self \
      .explore_custom_trim_user_info(
        user_info_attr = "userInfo",
        save_trimed_user_info_to_attr = "rerank_deep_ltr_trimmed_user_info",
        trim_user_info = rerank_gen_model_send_user_feas,
      ) \
      .delegate_enrich(
        kess_service = "{{fountain_rerank_gen_model_beam_ar_pinrec_kess_service}}",
        recv_item_attrs = [
          {"name": "rerank_gen_score", "as": "rerank_gen_ar_pinrec_score_list"}
        ],
        timeout_ms = 50,
        send_item_attrs = rerank_gen_model_send_item_feas,
        send_common_attrs = rerank_gen_model_send_common_feas,
        request_type = "{{fountain_rerank_gen_model_ar_pinrec_request_type}}",
        partition_size = "{{fountain_rerank_gen_model_ar_pinrec_partition_size}}",
      )

  def fountain_rerank_es_playtime_adjust(self):
    self \
    .if_("enable_fountain_rerank_es_playtime_adjust == 1 and user_is_low_interact == 1") \
      .gen_common_attr_by_lua(
        attr_map={
          "rerank_es_fullrank_pfintr_score": "rerank_es_fullrank_pfintr_score * fountain_rerank_es_fullrank_pfintr_score_coeff",
          "rerank_es_fullrank_watchtime_score": "rerank_es_fullrank_watchtime_score * fountain_rerank_es_fullrank_watchtime_score_coeff",
          "rerank_es_fullrank_watchtime_ori_score": "rerank_es_fullrank_watchtime_ori_score * fountain_rerank_es_fullrank_watchtime_ori_score_coeff",
          "rerank_es_fullrank_evtr_v2_score": "rerank_es_fullrank_evtr_v2_score * fountain_rerank_es_fullrank_evtr_v2_score_coeff",
          "rerank_es_fullrank_pfintr_score_addAndMul": "rerank_es_fullrank_pfintr_score_addAndMul * fountain_rerank_es_fullrank_pfintr_score_addAndMul_coeff",
          "rerank_es_fullrank_watchtime_score_addAndMul": "rerank_es_fullrank_watchtime_score_addAndMul * fountain_rerank_es_fullrank_watchtime_score_addAndMul_coeff",
          "rerank_es_fullrank_watchtime_ori_score_addAndMul": "rerank_es_fullrank_watchtime_ori_score_addAndMul * fountain_rerank_es_fullrank_watchtime_ori_score_addAndMul_coeff",
          "rerank_es_fullrank_evtr_v2_score_addAndMul": "rerank_es_fullrank_evtr_v2_score_addAndMul * fountain_rerank_es_fullrank_evtr_v2_score_addAndMul_coeff",
          "rerank_es_fullrank_click_score": "rerank_es_fullrank_click_score * fountain_rerank_es_fullrank_click_score_coeff",
          "rerank_es_fullrank_click_score_addAndMul": "rerank_es_fullrank_click_score_addAndMul * fountain_rerank_es_fullrank_click_score_addAndMul_coeff",
          "rerank_es_fullrank_shortview_score": "rerank_es_fullrank_shortview_score * fountain_rerank_es_fullrank_shortview_score_coeff",
          "rerank_es_fullrank_shortview_score_addAndMul": "rerank_es_fullrank_shortview_score_addAndMul * fountain_rerank_es_fullrank_shortview_score_addAndMul_coeff",
        }
      ) \
    .end_()
    return self
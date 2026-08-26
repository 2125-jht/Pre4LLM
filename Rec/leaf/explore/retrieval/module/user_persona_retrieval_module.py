from retrieval import RetrievalModule

class UserPersonaRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self):
    self.flow \
      .copy_user_meta_info(
        save_user_id_to_attr = "uid"
      ) \
      .if_("open_user_person_subject_retrieval == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = ["uid"],
          export_common_attr = ["user_persona_subject_key"],
          function_name = "CalcUserPersonaSubjectKey",
          class_name = "ExploreLightFunctionSetV2"
        ) \
        .get_common_attr_from_redis(
          cluster_name = "userHeTuTag",
          redis_params = [
            {
              "redis_key" : "{{user_persona_subject_key}}",
              "output_attr_name" : "user_subject_persona"
            }
          ]
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "uid",
            "user_subject_persona"
          ],
          export_common_attr = [
            "user_persona_subject_lv3",
            "user_persona_subject_lv2",
            "user_persona_subject_lv1"
          ],
          function_name = "CalcUserPersonaSubjectIndexKey",
          class_name = "ExploreLightFunctionSetV2"
        ) \
      .end_() \
      .if_("open_user_persona_14009_retrieve_reason > 0") \
        .retrieve_by_remote_index(
          kess_service = "grpc_recoRelevanceTagOrderedIndexServer",
          timeout_ms = 100,
          reason = 14009,
          reset_item_type = 0,
          common_query = "",
          querys = [
            {
              "query": "hetu_tag_v2:{{user_persona_ip_id_list}}",
              "search_num": "{{user_persona_ip_search_num}}"
            },
            {
              "query": "hetu_tag_v2:{{user_persona_tag_id_list}}",
              "search_num": "{{user_persona_tag_search_num}}"
            },
            {
              "query": "hetu_tag_v2:{{user_persona_subject_lv3}}",
              "search_num": "{{user_persona_subject_lv1_search_num}}"
            },
            {
              "query": "hetu_tag_v2:{{user_persona_subject_lv2}}",
              "search_num": "{{user_persona_subject_lv2_search_num}}"
            },
            {
              "query": "hetu_tag_v2:{{user_persona_subject_lv1}}",
              "search_num": "{{user_persona_subject_lv3_search_num}}"
            }
          ],
          default_random_search = 1,
          browsed_item_count = 0  # 过滤 browset
        ) \
       .else_() \
        .if_("close_user_persona_random_retrivel == 1") \
          .retrieve_by_remote_index(
            kess_service = "grpc_recoRelevanceTagOrderedIndexServer",
            timeout_ms = 100,
            reason = 10049,
            reset_item_type = 0,
            common_query = "",
            querys = [
              {
                "query": "hetu_tag_v2:{{user_persona_ip_index_key}}",
                "search_num": "{{user_persona_ip_search_num}}"
              },
              {
                "query": "hetu_tag_v2:{{user_persona_tag_index_key}}",
                "search_num": "{{user_persona_tag_search_num}}"
              },
              {
                "query": "hetu_tag_v2:{{user_persona_subject_lv3}}",
                "search_num": "{{user_persona_subject_lv1_search_num}}"
              },
              {
                "query": "hetu_tag_v2:{{user_persona_subject_lv2}}",
                "search_num": "{{user_persona_subject_lv2_search_num}}"
              },
              {
                "query": "hetu_tag_v2:{{user_persona_subject_lv1}}",
                "search_num": "{{user_persona_subject_lv3_search_num}}"
              }
            ],
            default_random_search = 0,
            browsed_item_count = 0  # 过滤 browset
          ) \
        .else_() \
          .retrieve_by_remote_index(
            kess_service = "grpc_recoRelevanceTagOrderedIndexServer",
            timeout_ms = 100,
            reason = 10049,
            reset_item_type = 0,
            common_query = "",
            querys = [
              {
                "query": "hetu_tag_v2:{{user_persona_ip_index_key}}",
                "search_num": "{{user_persona_ip_search_num}}"
              },
              {
                "query": "hetu_tag_v2:{{user_persona_tag_index_key}}",
                "search_num": "{{user_persona_tag_search_num}}"
              },
              {
                "query": "hetu_tag_v2:{{user_persona_subject_lv3}}",
                "search_num": "{{user_persona_subject_lv1_search_num}}"
              },
              {
                "query": "hetu_tag_v2:{{user_persona_subject_lv2}}",
                "search_num": "{{user_persona_subject_lv2_search_num}}"
              },
              {
                "query": "hetu_tag_v2:{{user_persona_subject_lv1}}",
                "search_num": "{{user_persona_subject_lv3_search_num}}"
              }
            ],
            default_random_search = 1,
            browsed_item_count = 0  # 过滤 browset
          ) \
        .end_() \
      .end_() \
      .deduplicate() \
      .if_("open_user_persona_sort == 1") \
        .get_remote_embedding(
          kess_service = "{{user_persona_photo_embedding_service_name}}",
          shard_num = 8,
          timeout_ms = 100,
          id_converter = {
            "type_name": "kuibaEmbeddingIdConverter"
          },
          size = 128,
          output_attr_name = "user_persona_photo_emb",
          query_source_type = "item_id",
          save_to_common_attr = False,
          client_side_shard = True) \
        .explore_user_persona_sorter(
          user_emb_key = "mc_u2i_user_embedding_list",
          item_emb_key = "user_persona_photo_emb"
        ) \
      .end_() \
      .limit(
        size = "{{user_persona_return_size}}"
      )

  def post_process(self) -> None:
    self.flow \
      .log_debug_info(
        common_attrs = ["uid"],
        print_all_item_keys = True
      )


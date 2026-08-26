from retrieval.retrieval_module import RetrievalModule

class Ip2tag2ipRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .get_common_attr_from_redis(
        skip = "{{fountain_skip_ip2tag2ip_retr_splash}}",
        cluster_name = "mmuKgExtension",
        redis_params = [{
          "redis_key": "{{source_movie_related_ips_key}}",
          "redis_value_type": "string",
          "output_attr_name": "kg_tag",
          "output_attr_type": "string_list"
        }]
      ) \
      .enrich_attr_by_lua(
        skip = "{{fountain_skip_ip2tag2ip_retr_splash}}",
        import_common_attr = [
          "kg_tag",
          "fountain_enable_ip2tag2ip_retr_opt",
          "fountain_ip2tag2ip_retr_kg_tag_priority_level"
        ],
        export_common_attr = [
          "kg_tag_key",
        ],
        function_for_common = "gen_kg_tag_key",
        lua_script_file = "fountain/retrieval/lua/module/splash_ip2tag2ip_retr__kg.lua"
      ) \
      .get_common_attr_from_redis(
        skip = "{{fountain_skip_ip2tag2ip_retr_splash}}",
        cluster_name = "mmuKgExtension",
        redis_params = [{
          "redis_key": "{{kg_tag_key}}",
          "redis_value_type": "string",
          "output_attr_name": "kg_extend_ips_str",
          "output_attr_type": "string_list"
        }]
      ) \
      .get_common_attr_from_redis(
        skip = "{{fountain_skip_ip2tag2ip_retr_splash}}",
        cluster_name = "mmuKgExtension",
        redis_params = [{
          "redis_key": "{{source_movie_ip_extends_key}}",
          "redis_value_type": "string",
          "output_attr_name": "kg_ip2ip_extend_ip_str",
          "output_attr_type": "string_list"
        }]
      ) \
      .enrich_attr_by_lua(
        skip = "{{fountain_skip_ip2tag2ip_retr_splash}}",
        import_common_attr = [
          "kg_extend_ips_str",
          "kg_ip2ip_extend_ip_str",
          "fountain_enable_ip2tag2ip_retr_opt",
          "source_movie_ip_extends_key",
          "kg_tag_key"
        ],
        export_common_attr = [
          "kg_extend_ips",
          "kg_ip2ip_extend_ips",
        ],
        function_for_common = "parse_kg_ips",
        lua_script_file = "fountain/retrieval/lua/module/splash_ip2tag2ip_retr__kg.lua"
      ) \
      .log_debug_info(
          common_attrs = [
            "kg_extend_ips_str",
            "kg_ip2ip_extend_ip_str",
            "source_movie_ip_extends_key",
            "kg_tag_key",
            "kg_extend_ips",
            "kg_ip2ip_extend_ips",
            "kg_tag",
            "fountain_enable_ip2tag2ip_retr_opt",
            "fountain_ip2tag2ip_retr_kg_tag_priority_level"
          ],
          for_debug_request_only = True
      ) \
      .retrieve_by_remote_index(
        skip = "{{fountain_skip_ip2tag2ip_retr_splash}}",
        kess_service = "{{fountain_ip2tag2ip_retr_kess_name}}",
        timeout_ms = 100,
        reason = self.reason,
        reset_item_type = 1,
        common_query = "",
        querys = [{
          "query": "hetu_tag_v2:{{kg_extend_ips}}",
          "search_num": "{{fountain_ip2tag2ip_retr_search_num_splash}}",
          "max_attr_num": "{{fountain_ip2tag2ip_retr_search_seed_num_splash}}"
        }, {
          "query": "hetu_tag_v2:{{kg_ip2ip_extend_ips}}",
          "search_num": "{{fountain_ip2tag2ip_retr_search_num_splash}}",
          "max_attr_num": "{{fountain_ip2tag2ip_retr_search_seed_num_splash}}"
        }],
        default_search_num = 50,
        default_random_search = 1,
      )
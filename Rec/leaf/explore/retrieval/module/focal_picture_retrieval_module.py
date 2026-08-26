from retrieval.retrieval_module import RetrievalModule


class FocalPictureRetrievalModule(RetrievalModule):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def process(self) -> None:

        self.flow \
            .if_("retrieve_mode == 0") \
                .get_remote_embedding_lite(
                    kess_service="grpc_exploreLookAlikeUserEmbServer",
                    shard_num=1,
                    timeout_ms=20,
                    id_converter={
                        "type_name": "kuibaEmbeddingIdConverter"
                    },
                    size=128,
                    output_attr_name="la_user_embedding_list",
                    query_source_type="user_id",
                    client_side_shard=True
                ) \
                .retrieve_by_ann_embedding(
                    reason=self.reason,
                    kess_service="{{service_name}}",
                    space="cosine",
                    timeout_ms="{{service_timeout_ms}}",
                    items_from_attr=["_USER_ID_"],
                    embeddings_from_attr=["la_user_embedding_list"],
                    bound_type={
                        "top_k": "{{retrieve_num_per_trigger}}"
                    },
                    algo_type={
                        "scann": {},
                    },
                    src_bucket="photo",
                    dest_bucket="{{dest_bucket}}"
                ) \
            .end_() \
            .if_("retrieve_mode == 1") \
                .pack_common_attr(
                    input_common_attrs=["click_list", "profile_v1_interaction_trigger_list"],
                    output_common_attr="pic_cl_retr_trigger_list",
                    deduplicate=True
                ) \
                .filter_by_common_attr(
                    item_list_from_attr="pic_cl_retr_trigger_list",
                    common_attr=["hate_list"]
                ) \
                .truncate(
                    size_limit="{{mix_trigger_cnt}}",
                    item_list_from_attr="pic_cl_retr_trigger_list"
                ) \
                .retrieve_by_ann_embedding(
                    kess_service="{{service_name}}",
                    space="cosin",
                    timeout_ms=100,
                    reason=self.reason,
                    shard_num=1,
                    items_from_attr=["pic_cl_retr_trigger_list"],
                    bound_type={
                        "top_k": "{{retrieve_num_per_trigger}}"
                    },
                    algo_type={
                        "scann": {},
                    },
                    src_bucket="photo",
                    dest_bucket="{{dest_bucket}}",
                    dest_bucket_item_type=0,
                ) \
            .end_() \
            .if_("retrieve_mode == 2") \
                .delegate_retrieve(
                    kess_service="{{service_name}}",
                    timeout_ms=100,
                    reason=self.reason,
                    request_type="default",
                    request_num="{{retrieve_num_per_trigger}}",
                    send_common_attrs_in_request=False,
                    send_common_attrs=[
                        {"name": "userInfo", "as": "user"},
                    ],
                    reset_item_type=0
                ) \
            .end_() \
            .if_("retrieve_mode == 3") \
                .retrieve_by_ann_embedding(
                    reason=self.reason,
                    kess_service="{{service_name}}",
                    space="cosine",
                    timeout_ms=100,
                    items_from_attr=["_USER_ID_"],
                    bound_type={
                        "top_k": "{{retrieve_num_per_trigger}}"
                    },
                    algo_type={
                        "scann": {},
                    },
                    src_bucket="user",
                    dest_bucket="{{dest_bucket}}"
                ) \
            .end_() \
            .deduplicate() \
            .filter_by_common_attr(
                common_attr=["browse_screen__pid_list"]
            ) \
            .limit(size="{{retrieve_num}}")

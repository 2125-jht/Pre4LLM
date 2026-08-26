from retrieval import RetrievalModule


class PicTTHighInteractionRetrievalModule(RetrievalModule):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    @classmethod
    def is_retrieval(cls) -> bool:
        return True

    def process(self) -> None:
        self.flow \
            .retrieve_by_ann_embedding(
              reason=self.reason,
              kess_service="{{ann_service_name}}",
              space="{{ann_space}}",
              timeout_ms="{{ann_service_timeout_ms}}",
              items_from_attr=["_USER_ID_"],
              bound_type={
                  "total_limit": "{{retr_total_limit}}"
              },
              algo_type={
                  "scann": {}
              },
              src_data_type="user_src",
              src_bucket="user_src",
              dest_bucket="photo_dest"
            ) \
            .deduplicate(
            )

    def post_process(self) -> None:
        self.flow \
            .log_debug_info(
              common_attrs=[
                  "ann_service_name",
                  "ann_service_timeout_ms",
                  "retr_total_limit",
                  "_USER_ID_",
              ],
              item_attrs=[
              ],
              for_debug_request_only=True
            )

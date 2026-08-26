from retrieval.retrieval_module import RetrievalModule


class KgnnV2RetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
      super().__init__(name)
  
  def process(self) -> None:
    self.flow \
        .fetch_kgnn_neighbors(
          id_from_common_attr="featureUId",
          save_neighbors_to='retr_list',
          model_name="{{model_name}}",
          table_name="{{table_name}}",
          sample_num="{{sample_num}}",
          timeout_ms=50,
          sample_type="{{sample_type}}",
          padding_type="{{padding_type}}"
        ) \
        .retrieve_by_common_attr(
          attr="retr_list", 
          reason=self.reason
        ) \
        .deduplicate() \
        .filter_by_common_attr(
          common_attr=["browse_screen__pid_list"]
        ) \
        .filter_by_browse_set(
          skip="{{skip_browse_set}}"
        ) \
        .limit(
          size="{{retrieve_num}}"
        )
    if (self.custom_label_attrs):
      self.flow \
        .set_attr_value(
          item_attrs = [
            {
              "name": label_attr,
              "type": "int",
              "value": 1
            } for label_attr in self.custom_label_attrs
          ],
        )
  
  @property
  def custom_label_attrs(self) -> list:
    return self.config.get("custom_label_attrs", [])

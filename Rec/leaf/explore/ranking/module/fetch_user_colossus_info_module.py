from ranking import CommonModule

colossus_action_names = [
  "Play", 
  "Play0To7", 
  "Play7To20", 
  "Play20To58", 
  "Play58More", 
  "Ev", 
  "Lv", 
  "Interact"
]

colossus_feature_prefix = "pColossus"
colossus_score_name = "pColossusScore"

def get_photo_colossus_features():
  features = [colossus_score_name] # 老 leaf 中的 user_colos_score
  colos_feat_day_period_strs = ["D7", "D15", "D30", "All"]
  for action_name in colossus_action_names:
    for period in colos_feat_day_period_strs:
      for suffix in ["Hetu1", "Hetu2", "Hetu3"]:
        features.append(colossus_feature_prefix + action_name + period + suffix)
  return features

photo_colossus_features = get_photo_colossus_features()

class FetchUserColossusInfoModule(CommonModule):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def process(self) -> None:
      """leave empty function by AutoDelete"""

    def post_process(self) -> None:
      self.flow \
      .log_debug_info(
        common_attrs = [
          "colossus_feature_periods"
        ],
        item_attrs = photo_colossus_features,
        for_debug_request_only = True
      )
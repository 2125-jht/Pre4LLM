from cascading_v2 import CommonModule

class CascadingMainPredictModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .if_("enable_cascade_downgrade == 0") \
        .delegate_enrich(
          kess_service = "{{mc_new_arch_tower_service}}",
          timeout_ms = 100,
          request_type = "{{mc_new_arch_tower_request_type}}",
          send_common_attrs = [
            { "name": "userInfo", "as": "user_info_str" },
          ],
          recv_item_attrs = [
            # { "name": "ctr", "as": "cascade_pctr" }, 使用 fc model
            { "name": "ltr", "as": "cascade_pltr" },
            { "name": "wtr", "as": "cascade_pwtr" },
            { "name": "ftr", "as": "cascade_pftr" },
            # { "name": "lvr", "as": "cascade_plvtr" }, 使用 fc model
            { "name": "lvtr2", "as": "cascade_plvtr2" },
            # { "name": "svr", "as": "cascade_psvtr" }, 使用 fc model
            { "name": "ptr", "as": "cascade_ptr" },
            # { "name": "vtr", "as": "cascade_pwatch_time" }, 使用 fc model
            { "name": "eps", "as": "cascade_pepstr" },
            { "name": "ces", "as": "cascade_pcestr" },
            { "name": "cmtr", "as": "cascade_pcmtr" },
            { "name": "live", "as": "cascade_plivingtr" },
            { "name": "cltr", "as": "cascade_pcltr" },
            { "name": "down", "as": "cascade_pdtr" },
            { "name": "htr", "as": "cascade_phtr"},
            { "name": "eftr", "as": "cascade_peftr"},
            { "name": "efctr", "as": "cascade_pefctr"},
            # { "name": "cptr", "as": "cascade_pcptr"}, 使用 fc model
            { "name": "wtd", "as": "cascade_pwtd"},
            # picture
            { "name": "pic_wtdPlaytime", "as": "cascade_pic_wtd"},
            { "name": "pic_lvtr", "as": "cascade_pic_lvtr"},
            { "name": "pic_cpr", "as": "cascade_pic_cpr"},
          ],
          for_predict = True,
          use_packed_item_attr = True,
          infer_output_type = 2,
        ) \
      .end_()

#!/usr/bin/env python3
# coding=utf-8

from cascading import CommonModule

class CascadingDowngradeModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .if_("enable_cascade_downgrade == 1") \
      .enrich_attr_by_lua(  # 降级时使用经验 xtr 填充 pxtr
        import_item_attr = [
          "empirical_ctr",
          "empirical_ltr",
          "empirical_wtr",
          "empirical_ftr",
          "empirical_lvtr",
          "empirical_svtr",
          "empirical_ptr",
          "empirical_htr",
          "empirical_cmtr",
        ],
        export_item_attr = [
          "cascade_pctr",
          "cascade_pltr",
          "cascade_pwtr",
          "cascade_pftr",
          "cascade_plvtr",
          "cascade_psvtr",
          "cascade_ptr",
          "cascade_phtr",
          "cascade_pcmtr",
        ],
        function_for_item = "cascade_fill_pxtr_downgrade",
        lua_script_file = "explore/cascading/lua/module/cascading_downgrade__cascade_fill_pxtr_downgrade.lua") \
      .end_()

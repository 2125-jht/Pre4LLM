#!/usr/bin/env python3
# coding=utf-8

from cascading import CommonModule

class CascadingDowngradeModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .enrich_attr_by_lua(  # 降级时使用经验 xtr 填充 pxtr
        skip = "{{return enable_cascade_downgrade == 0}}",
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
        lua_script_file = "life/cascading/lua/module/cascading_downgrade__cascade_fill_pxtr_downgrade.lua")\
      .calc_by_formula1(
          kconf_key = "formula.scenarioKey07.NatureLongTermItemScore",
          import_item_attr = [
            "empirical_ctr",
            "empirical_ltr",
            "empirical_wtr",
            "empirical_ftr",
            "empirical_lvtr",
            "empirical_svtr",
            "empirical_ptr",
            "empirical_htr",
            "empirical_cmtr"
          ],
          export_formula_value = [
            {"name": "final_score", "as": "long_term_nature_score"}
          ],
          abtest_biz_name = "KUAISHOU_APPS"
        ) \
      


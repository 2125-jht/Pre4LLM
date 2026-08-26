from cascading import CommonModule

class CascadingCtrOptimizeModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow\
    .if_("enable_ctr_optimize_life == 1") \
      .if_("enable_empirical_ctr_filter == 1") \
        .filter_by_attr(
          attr_name = "empirical_ctr",
          remove_if = "<",
          compare_to = "{{life_empirical_ctr_filter_threshold}}",
        ) \
      .end_() \
      .if_("enable_click_count_filter == 1") \
        .filter_by_attr(
          attr_name = "explore_stat__click_count",
          remove_if = "<",
          compare_to = "{{life_click_count_filter_threshold}}",
        ) \
      .end_() \
      .if_("enable_long_play_count_filter == 1") \
        .filter_by_attr(
          attr_name = "explore_stat__long_play_count",
          remove_if = "<",
          compare_to = "{{life_long_play_count_filter_threshold}}",
        ) \
      .end_() \
      .if_("enable_real_show_count_filter == 1") \
        .filter_by_attr(
          attr_name = "explore_stat__real_show_count",
          remove_if = "<",
          compare_to = "{{life_real_show_count_filter_threshold}}",
        ) \
      .end_() \
      .item_attr_operation(
        item_attr_a = "empirical_svtr",
        common_attr_b = "{{life_emp_svtr_shift_coef}}",
        operator = "+",
        output_attr = "emp_shift_svtr"
      ) \
      .item_attr_operation(
        item_attr_a="empirical_ctr",
        item_attr_b="emp_shift_svtr",
        operator="/",
        output_attr="optimized_ctr"
      )\
      .log_debug_info(
        item_attrs=["explore_stat__click_count",
                    "explore_stat__long_play_count",
                    "empirical_ctr","empirical_svtr",
                    "optimized_ctr"
                      ],
        for_debug_request_only = True
      ) \
    .end_()
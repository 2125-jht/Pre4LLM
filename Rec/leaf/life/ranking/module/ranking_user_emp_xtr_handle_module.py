from ranking import CommonModule

class RankingUserEmpXtrHandleModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
    .enrich_attr_by_light_function(
      skip = "{{explore_fr_skip_user_emp_xtr_handle}}",
      import_common_attr = [
        "user_emp_ltr",
        "user_emp_wtr",
        "user_emp_ftr",
        "user_emp_cmtr",
        {"name": "explore_fr_user_emp_xtr_coeff", "as": "emp_xtr_coeff"}
      ],
      export_common_attr = [
        {"name": "user_emp_ltr_new", "as": "user_emp_ltr_fr_threshold"},
        {"name": "user_emp_wtr_new", "as": "user_emp_wtr_fr_threshold"},
        {"name": "user_emp_ftr_new", "as": "user_emp_ftr_fr_threshold"},
        {"name": "user_emp_cmtr_new", "as": "user_emp_cmtr_fr_threshold"}
      ],
      function_name = "EmpXtrThreshold",
      class_name = "ExploreLightFunctionSetV2",
    )

from cascading import CommonModule

class CascadingFinalSortDiversityFrModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    # 此 module 主要用来生成在粗排第二阶段及之后使用的 diversity fr
    """leave empty function by AutoDelete"""

from common import JsonConfigRecoModule

class CommonModule(JsonConfigRecoModule):
  def __init__(self, name: str) -> None:
    super().__init__(name, "fountain/filter/config/module")

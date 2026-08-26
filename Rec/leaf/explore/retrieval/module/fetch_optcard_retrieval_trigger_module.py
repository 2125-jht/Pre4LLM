from retrieval import CommonModule

# TODO:liucong03(2月份删除，后面到统一trigger里调整)

class FetchOptcardRetrievalTriggerModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self):
    """leave empty function by AutoDelete"""

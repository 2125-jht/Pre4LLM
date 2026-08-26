from ranking import CommonModule

class RankingEnsembleAbsXtrScoreModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    """leave empty function by AutoDelete"""

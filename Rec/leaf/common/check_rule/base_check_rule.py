from abc import ABC, abstractmethod

class BaseCheckRule(ABC):
  def __init__(self, name: str) -> None:
    self.__name = name

  @property
  def name(self) -> str:
    return self.__name

  @abstractmethod
  def check(self) -> None:
    pass

import os
from .base_check_rule import BaseCheckRule

class LuaNameCheckRule(BaseCheckRule):
  def __init__(self, module_config_dir: str, module_lua_dir: str) -> None:
    super().__init__("lua_name_check")
    self.__module_config_dir = module_config_dir
    self.__module_lua_dir = module_lua_dir

  def check(self) -> None:
    module_name_set = set()
    module_config_files = os.listdir(self.__module_config_dir)
    for file in module_config_files:
      module_name = os.path.splitext(file)[0]
      module_name_set.add(module_name)

    if os.path.isdir(self.__module_lua_dir):
      module_lua_files = os.listdir(self.__module_lua_dir)
      for file in module_lua_files:
        items = os.path.splitext(file)
        assert len(items) == 2 and items[1] == ".lua", "{} 命名不合法，请以 lua 为后缀名".format(file)
        items = items[0].split("__", 1)
        assert len(items) == 2, "{} 命名不合法，请以 {{module_name}}__{{xxx}}.lua 格式命名 lua 文件".format(file)
        assert items[0] in module_name_set, "{} 命名不合法，不存在 name 为 {} 的 module".format(file, items[0])

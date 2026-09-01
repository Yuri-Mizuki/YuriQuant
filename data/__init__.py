"""数据层统一入口。"""
from data.cache import DataCache
from data.datasource import DataSource, create_datasource
from data.universe import Universe


def get_cache() -> DataCache:
    """创建带缓存的数据访问对象（推荐入口）。"""
    ds = create_datasource()
    return DataCache(ds)


def get_universe() -> Universe:
    """创建 Universe 管理器。"""
    return Universe(get_cache())


__all__ = ["DataSource", "DataCache", "Universe", "create_datasource", "get_cache", "get_universe"]

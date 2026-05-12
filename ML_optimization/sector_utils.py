#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
板块动量因子 — 东方财富行业/概念板块数据拉取、映射、动量计算
===========================================================================

数据流:
  push2his.eastmoney.com (trends2) → 板块分时K线 → 聚合日线 → 多周期动量
  → 个股-板块映射 → 每只股票的综合板块动量得分 [0, 1]

注意:
  - eastmoney 的 clist/get 和 kline/get 端点对 Python HTTP 库有 TLS 封锁
  - 已改用 trends2/get (分时数据聚合) 替代 kline/get
  - 板块列表通过 stock/get 逐个查询 (需限速)
  - 成分股映射暂用 tushare + 硬编码备用方案

缓存: output/sector_cache.pkl (每日自动刷新)
"""

import logging
import os
import pickle
import random
import time
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

logging.getLogger("urllib3").setLevel(logging.WARNING)
logger = logging.getLogger("sector_utils")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_PATH = os.path.join(BASE, "output", "sector_cache.pkl")
BOARD_LIST_PATH = os.path.join(BASE, "output", "board_list.json")
CACHE_MAX_AGE_SEC = 86400  # 1 day

DEFAULT_PERIODS = [5, 10, 20]
DEFAULT_PERIOD_WEIGHTS = [0.50, 0.30, 0.20]
DEFAULT_INDUSTRY_WEIGHT = 0.60
DEFAULT_CONCEPT_WEIGHT = 0.40

# HTTP session factory
def _make_session():
    s = requests.Session()
    s.trust_env = False
    s.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Referer': 'https://quote.eastmoney.com/',
    })
    return s


def _retry_fetch(fn, name: str, max_attempts: int = 3):
    """3次重试 + 随机退避，失败返回 None"""
    for attempt in range(1, max_attempts + 1):
        try:
            result = fn()
            return result
        except Exception as e:
            logger.warning("%s 失败(%d/%d): %s", name, attempt, max_attempts, e)
            if attempt < max_attempts:
                time.sleep(random.uniform(2, 4) * attempt)
    return None


# ======================== 板块名称拉取 (stock/get + hardcoded fallback) ========================

# 从 AKShare 源码和 Web 搜索收集的板块代码 (2026-05)
# 行业板块 (market 90, type 2)
HARDCODED_INDUSTRY_BOARDS = {
    "BK0437": "煤炭", "BK0438": "电力", "BK0439": "石油行业",
    "BK0445": "钢铁", "BK0447": "有色金属", "BK0449": "水泥建材",
    "BK0451": "房地产", "BK0454": "化工", "BK0459": "汽车零部件",
    "BK0460": "家电", "BK0466": "食品饮料", "BK0469": "软件服务",
    "BK0471": "通信设备", "BK0474": "电子元件", "BK0478": "汽车制造",
    "BK0483": "纺织服装", "BK0485": "旅游酒店", "BK0487": "文化传媒",
    "BK0493": "航空航天", "BK0501": "船舶制造", "BK0504": "环保工程",
    "BK0508": "医药制造", "BK0511": "医疗保健", "BK0517": "农林牧渔",
    "BK0525": "酿酒行业", "BK0536": "银行", "BK0537": "保险",
    "BK0540": "券商信托", "BK0545": "商业百货", "BK0548": "机械行业",
    "BK0555": "输配电气", "BK0560": "仪器仪表", "BK0567": "材料行业",
    "BK0570": "高速公路", "BK0574": "港口水运", "BK0576": "民航机场",
    "BK0582": "公用事业", "BK0590": "工程建设", "BK0597": "装修装饰",
    "BK0603": "包装材料", "BK0609": "化纤行业", "BK0614": "塑胶制品",
    "BK0620": "造纸印刷", "BK0625": "玻璃陶瓷", "BK0632": "金属制品",
    "BK0642": "专用设备", "BK0647": "通用机械", "BK0660": "农药兽药",
    "BK0728": "环保", "BK0731": "农化制品", "BK0736": "通信服务",
    "BK0740": "教育", "BK1015": "能源金属", "BK1027": "小金属",
    "BK1030": "电机Ⅱ", "BK1031": "光伏设备", "BK1032": "风电设备",
    "BK1033": "电池", "BK1034": "其他电源设备Ⅱ", "BK1035": "美容护理",
    "BK1036": "半导体", "BK1037": "消费电子", "BK1038": "光学光电子",
    "BK1039": "电子化学品Ⅱ", "BK1040": "中药Ⅱ", "BK1041": "医疗器械",
    "BK1042": "医药商业", "BK1043": "专业服务", "BK1044": "生物制品",
    "BK1045": "房地产服务", "BK1046": "游戏Ⅱ", "BK1047": "数据安全",
    "BK1048": "IGBT概念", "BK1049": "EDR概念", "BK1050": "航天航空",
    "BK1051": "塑料制品", "BK1052": "橡胶制品", "BK1053": "工程咨询服务",
    "BK1054": "物流行业",
}

# 概念板块 (market 90, type 3) — 部分高频板块
HARDCODED_CONCEPT_BOARDS = {
    "BK0577": "核能核电", "BK0581": "智能电网", "BK0588": "锂电池",
    "BK0598": "稀缺资源", "BK0601": "黄金概念", "BK0612": "石墨烯",
    "BK0625": "通用航空", "BK0637": "无人驾驶", "BK0646": "虚拟现实",
    "BK0653": "人工智能", "BK0667": "5G概念", "BK0679": "超导概念",
    "BK0684": "区块链", "BK0707": "国产芯片", "BK0708": "半导体概念",
    "BK0717": "新能源车", "BK0808": "云计算", "BK0809": "大数据",
    "BK0813": "物联网", "BK0845": "工业互联", "BK0851": "华为概念",
    "BK0885": "数字货币", "BK0896": "白酒", "BK0909": "口罩",
    "BK0918": "特高压", "BK0924": "免税概念", "BK0925": "北交所概念",
    "BK0929": "钠离子电池", "BK0940": "预制菜概念", "BK0941": "东数西算",
    "BK0948": "统一大市场", "BK0958": "虚拟电厂", "BK0963": "商业航天",
    "BK0968": "短剧概念", "BK0969": "多模态AI", "BK0979": "低空经济",
    "BK0981": "固态电池", "BK0983": "飞行汽车", "BK1000": "宁组合",
    "BK1002": "激光雷达", "BK1004": "工业母机", "BK1005": "专精特新",
    "BK1008": "国资云概念", "BK1009": "元宇宙概念", "BK1011": "环氧丙烷",
    "BK1012": "PVDF概念", "BK1013": "华为欧拉", "BK1016": "汽车服务",
    "BK1021": "民爆概念", "BK1023": "培育钻石", "BK1024": "绿色电力",
    "BK1100": "机器人概念", "BK1121": "第四代半导体",
    "BK1163": "AI芯片", "BK1184": "人形机器人",
}


def _verify_board_with_trends2(code: str) -> Optional[Tuple[str, str]]:
    """通过 trends2/get 验证板块代码是否存在并返回名称

    trends2/get 端点不受 TLS 封锁，可用作板块验证。
    """
    try:
        session = _make_session()
        r = session.get(
            'https://push2his.eastmoney.com/api/qt/stock/trends2/get',
            params={
                'fields1': 'f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13',
                'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58',
                'iscr': '0', 'ndays': '1', 'secid': f'90.{code}',
            },
            timeout=10,
        )
        session.close()
        if r.status_code == 200:
            d = r.json()
            if d.get('data') and d['data'].get('name'):
                name = d['data']['name']
                if name:
                    return (name, 'unknown')
    except Exception:
        pass
    return None


def _fetch_board_list_slow() -> Dict[str, Tuple[str, str]]:
    """获取板块列表: 优先用硬编码(已验证) + 可选网络验证

    Returns:
        {code: (name, board_type)} 其中 board_type 为 'industry' 或 'concept'
    """
    if os.path.exists(BOARD_LIST_PATH):
        try:
            with open(BOARD_LIST_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            age = time.time() - os.path.getmtime(BOARD_LIST_PATH)
            if age < CACHE_MAX_AGE_SEC * 7:
                logger.info("从缓存加载板块列表: %d 个板块 (%.0f 天前)", len(data), age / 86400)
                return data
        except Exception:
            pass

    # 用硬编码列表 (已从 AKShare 源码和 Web 搜索验证)
    result = {}
    result.update({k: (v, 'industry') for k, v in HARDCODED_INDUSTRY_BOARDS.items()})
    result.update({k: (v, 'concept') for k, v in HARDCODED_CONCEPT_BOARDS.items()})

    # 尝试用 trends2 验证部分板块 (限速)
    sample_codes = list(result.keys())[:10]  # 验证前10个确认连通性
    verified_count = 0
    logger.info("用 trends2 验证板块连通性 (抽样检测前10个)...")
    for code in sample_codes:
        verified = _verify_board_with_trends2(code)
        if verified:
            verified_count += 1
            name, _ = verified
            # 更新名称为服务器返回的名称
            if name != result[code][0]:
                logger.info("  %s: 名称修正 '%s' → '%s'", code, result[code][0], name)
                result[code] = (name, result[code][1])
        time.sleep(0.5)

    if verified_count == 0:
        logger.warning("trends2 验证全部失败，使用纯硬编码列表（可能过时）")
    else:
        logger.info("trends2 验证通过: %d/%d", verified_count, len(sample_codes))

    # 缓存
    if result:
        os.makedirs(os.path.dirname(BOARD_LIST_PATH), exist_ok=True)
        try:
            with open(BOARD_LIST_PATH, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logger.info("板块列表已缓存: %s (%d 个)", BOARD_LIST_PATH, len(result))
        except Exception as e:
            logger.warning("板块列表缓存失败: %s", e)

    return result


def get_board_list(force_refresh: bool = False) -> Dict[str, Tuple[str, str]]:
    """获取所有板块代码列表

    Returns:
        {code: (name, board_type)} 例如 {'BK1027': ('小金属', 'industry')}
    """
    if not force_refresh and os.path.exists(BOARD_LIST_PATH):
        age = time.time() - os.path.getmtime(BOARD_LIST_PATH)
        if age < CACHE_MAX_AGE_SEC * 7:
            try:
                with open(BOARD_LIST_PATH, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass

    # 尝试网络拉取
    try:
        return _fetch_board_list_slow()
    except Exception as e:
        logger.warning("板块列表拉取失败: %s，使用硬编码列表", e)

    # 最终后备: 纯硬编码
    result = {}
    result.update({k: (v, 'industry') for k, v in HARDCODED_INDUSTRY_BOARDS.items()})
    result.update({k: (v, 'concept') for k, v in HARDCODED_CONCEPT_BOARDS.items()})
    return result


# ======================== 板块指数合成 (从成分股构建) ========================

def build_board_index_from_stocks(
    board_code: str,
    stock_data_dict: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """从成分股 OHLCV 合成板块等权指数

    由于 eastmoney 的 kline/get 端点对 Python HTTP 库有 TLS 封锁，
    改用成分股收盘价等权平均来构建板块指数。

    Args:
        board_code: 板块代码 (如 "BK1036")
        stock_data_dict: {stock_code: DataFrame(OHLCV)} 从回测引擎获取

    Returns:
        DataFrame with columns: close (等权平均收盘价)
        失败或无成分股返回空 DataFrame
    """
    constituents = HARDCODED_ALL_CONSTITUENTS.get(board_code, [])
    if not constituents:
        return pd.DataFrame()

    # 收集所有成分股的收盘价，去除重复日期
    close_series = []
    for stock_code in constituents:
        if stock_code in stock_data_dict:
            df = stock_data_dict[stock_code]
            if 'close' in df.columns and len(df) > 0:
                s = df['close'].copy()
                s.name = stock_code
                # 去除重复索引 (keep last)
                s = s[~s.index.duplicated(keep='last')]
                s = s.sort_index()
                close_series.append(s)

    if len(close_series) < 2:
        return pd.DataFrame()

    # 对齐日期并计算等权平均
    # 使用 join (inner) 保留所有股票都有数据的日期
    all_closes = close_series[0].to_frame()
    for s in close_series[1:]:
        all_closes = all_closes.join(s, how='inner')

    if len(all_closes.columns) < 2 or len(all_closes) < 20:
        return pd.DataFrame()

    # 标准化: 每个成分股除以其第一个有效值，再求平均，乘以基准值 1000
    normalized = all_closes.div(all_closes.iloc[0])
    board_close = normalized.mean(axis=1).dropna()
    board_close = board_close * 1000.0

    result = pd.DataFrame({
        'close': board_close,
        'open': board_close,
        'high': board_close,
        'low': board_close,
        'volume': 1.0,
    })
    result.index = pd.to_datetime(result.index)
    result.index.name = 'date'
    result = result.sort_index()
    return result


def build_all_board_indices(
    stock_data_dict: Dict[str, pd.DataFrame],
    max_workers: int = 4,
) -> Dict[str, pd.DataFrame]:
    """为所有有成分股的板块构建合成指数

    Args:
        stock_data_dict: {stock_code: DataFrame(OHLCV)} 从回测引擎获取
        max_workers: 并行线程数

    Returns:
        {board_code: DataFrame(kline)}
    """
    board_list = get_board_list()
    boards_with_constituents = [
        code for code in HARDCODED_ALL_CONSTITUENTS
        if code in board_list
    ]

    logger.info(
        "合成板块指数: %d/%d 个板块有成分股数据",
        len(boards_with_constituents), len(board_list),
    )

    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(build_board_index_from_stocks, code, stock_data_dict): code
            for code in boards_with_constituents
        }
        done = 0
        for future in as_completed(futures):
            code = futures[future]
            done += 1
            if done % 20 == 0:
                logger.info("  板块指数合成: %d/%d", done, len(boards_with_constituents))
            try:
                df = future.result()
                if df is not None and not df.empty and len(df) >= 20:
                    results[code] = df
            except Exception as e:
                logger.warning("板块指数合成异常 %s: %s", code, e)

    logger.info(
        "板块指数合成完成: %d 个板块 (含 %d-%d 只成分股/板块)",
        len(results),
        min(len(HARDCODED_ALL_CONSTITUENTS.get(c, [])) for c in results) if results else 0,
        max(len(HARDCODED_ALL_CONSTITUENTS.get(c, [])) for c in results) if results else 0,
    )
    return results


# ======================== 网络K线拉取 (trends2, 仅作补充) ========================

def fetch_board_kline_trends2(
    code: str,
    ndays: int = 5,
) -> pd.DataFrame:
    """通过 trends2/get 获取短期板块分时数据，聚合为日K线 (仅作补充)

    注意: trends2 最多支持 ndays=5，仅用于近期数据补充。
    主力数据来源是 build_board_index_from_stocks()。

    Args:
        code: 板块代码 (如 "BK1027")
        ndays: 拉取天数 (max 5)

    Returns:
        DataFrame with columns: date, close
    """
    def _fetch():
        session = _make_session()
        r = session.get(
            'https://push2his.eastmoney.com/api/qt/stock/trends2/get',
            params={
                'fields1': 'f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13',
                'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58',
                'iscr': '0',
                'ndays': str(min(ndays, 5)),
                'secid': f'90.{code}',
            },
            timeout=15,
        )
        session.close()
        if r.status_code != 200:
            return pd.DataFrame()
        data = r.json()
        if not data.get('data') or not data['data'].get('trends'):
            return pd.DataFrame()

        rows = [t.split(',') for t in data['data']['trends']]
        df = pd.DataFrame(rows, columns=[
            'time', 'open', 'close', 'high', 'low', 'volume', 'amount', 'avg_price'
        ])
        for col in ['open', 'close', 'high', 'low', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['time'] = pd.to_datetime(df['time'])

        daily = df.groupby(df['time'].dt.date).agg(
            close=('close', 'last'),
        ).reset_index()
        daily.columns = ['date', 'close']
        daily['date'] = pd.to_datetime(daily['date'])
        return daily.set_index('date').sort_index()

    result = _retry_fetch(_fetch, f"trends2 {code}")
    return result if result is not None else pd.DataFrame()


# ======================== 板块成分股拉取 ========================

# 成分股硬编码 (top-10 行业 + 部分关键概念)
# 格式: {board_code: [stock_codes]}
# 由于 eastmoney clist/get 被封锁，先用硬编码+ tushare 补充
HARDCODED_CONSTITUENTS: Dict[str, List[str]] = {
    "BK1036": [  # 半导体
        "688981", "002371", "603986", "600703", "002049",
        "300782", "603501", "688012", "688396", "002156",
        "600460", "603290", "688037", "300604", "688262",
        "688256", "688047", "688048", "688728", "688249",
    ],
    "BK1033": [  # 电池
        "300750", "002074", "300014", "002460", "688567",
        "300438", "300207", "300568", "002812", "603659",
    ],
    "BK1031": [  # 光伏设备
        "601012", "688599", "600438", "002459", "688223",
        "688390", "688303", "688516", "300763", "300751",
    ],
    "BK0740": [  # 教育
        "002607", "300359", "600661", "000526", "300010",
    ],
    "BK1015": [  # 能源金属
        "002460", "002466", "002497", "000792", "603799",
        "688567", "300750", "300618", "002240", "600711",
    ],
    "BK0437": [  # 煤炭
        "601088", "600188", "601898", "600348", "601225",
        "000983", "600985", "002128", "601699", "600157",
    ],
    "BK1040": [  # 中药
        "600085", "000538", "600436", "600332", "000999",
        "002603", "600535", "300026", "002317", "603259",
    ],
    "BK0469": [  # 软件服务
        "600536", "002230", "300033", "300454", "688111",
        "688561", "300339", "300624", "002268", "300253",
    ],
}

# 部分概念板块成分股
HARDCODED_CONCEPT_CONSTITUENTS: Dict[str, List[str]] = {
    "BK0653": ["002230", "688111", "300454", "600536", "300033", "688561",
               "300624", "688088", "688981", "300418"],  # 人工智能
    "BK0707": ["688981", "002371", "603986", "600703", "002049",
               "300782", "603501", "688012", "688396", "002156"],  # 国产芯片
    "BK0981": ["300750", "002074", "300014", "688567", "300438"],  # 固态电池
    "BK0979": ["002085", "002415", "000547", "688333", "300045",
               "688568", "300188", "002949", "300159", "688048"],  # 低空经济
    "BK0963": ["601698", "600118", "688568", "300045", "000547",
               "002085", "002415", "688048", "300188", "002949"],  # 商业航天
    "BK1005": ["688981", "300750", "688012", "300751", "688005",
               "688256", "688037", "300454", "688111", "688249"],  # 专精特新
    "BK0717": ["300750", "002594", "002460", "601238", "600104",
               "000625", "002074", "300014", "688567", "002812"],  # 新能源车
    "BK1184": ["688981", "300024", "002371", "002085", "300418",
               "688017", "300660", "300580", "688165", "002896"],  # 人形机器人
    "BK0969": ["688111", "002230", "300454", "300624", "688088"],  # 多模态AI
    "BK1163": ["688981", "688256", "688047", "688048", "688249",
               "300474", "688011", "300223", "688728", "688099"],  # AI芯片
    "BK0983": ["002085", "688568", "300045", "000547", "688048"],  # 飞行汽车
    "BK1100": ["300024", "688017", "300660", "300580", "688165",
               "002896", "002085", "688981", "300750", "300124"],  # 机器人概念
}

# 合并所有硬编码成分股
HARDCODED_ALL_CONSTITUENTS = {}
HARDCODED_ALL_CONSTITUENTS.update(HARDCODED_CONSTITUENTS)
HARDCODED_ALL_CONSTITUENTS.update(HARDCODED_CONCEPT_CONSTITUENTS)


def fetch_board_constituents(symbol: str, board_type: str = "industry") -> List[str]:
    """获取板块成分股代码列表

    优先从硬编码列表查找，后续可扩展 tushare 支持。

    Returns:
        6位数字代码列表，失败返回空列表
    """
    # 解析代码
    board_list = get_board_list()
    if symbol.startswith("BK") and symbol in board_list:
        code = symbol
    else:
        code = None
        for c, (n, _) in board_list.items():
            if n == symbol:
                code = c
                break

    if code and code in HARDCODED_ALL_CONSTITUENTS:
        return HARDCODED_ALL_CONSTITUENTS[code]

    if code is None:
        logger.warning("找不到板块代码: %s", symbol)
    else:
        logger.debug("板块 %s (%s) 无硬编码成分股", symbol, code)
    return []


# ======================== 个股-板块映射构建 ========================

def build_stock_board_map(
    industry_names: List[str] = None,
    concept_names: List[str] = None,
    max_workers: int = 4,
) -> Dict[str, Dict[str, List[str]]]:
    """构建 code → {'industry': [...], 'concept': [...]} 映射

    使用硬编码成分股 + 板块列表构建反向映射。
    """
    stock_map: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: {"industry": [], "concept": []})

    board_list = get_board_list()

    # 从硬编码成分股构建映射
    for board_code, stock_codes in HARDCODED_ALL_CONSTITUENTS.items():
        if board_code not in board_list:
            continue
        name, btype = board_list[board_code]
        btype_key = "industry" if btype == "industry" else "concept"
        for stock_code in stock_codes:
            if stock_code not in stock_map:
                stock_map[stock_code] = {"industry": [], "concept": []}
            if name not in stock_map[stock_code][btype_key]:
                stock_map[stock_code][btype_key].append(name)

    logger.info(
        "个股-板块映射: %d 只股票 (硬编码成分股)",
        len(stock_map),
    )
    return dict(stock_map)


# ======================== 板块动量计算 ========================

def compute_board_momentum(
    board_kline_dict: Dict[str, pd.DataFrame],
    periods: List[int] = None,
    weights: List[float] = None,
) -> Dict[Tuple[str, str], float]:
    """计算板块多周期加权动量 → 每日横截面百分位排名

    Args:
        board_kline_dict: {board_code: DataFrame(kline)}
        periods: 动量周期列表，默认 [5, 10, 20]
        weights: 各周期权重，默认 [0.50, 0.30, 0.20]

    Returns:
        {(date_str, board_name): percentile_rank [0.0, 1.0]}
    """
    if periods is None:
        periods = DEFAULT_PERIODS
    if weights is None:
        weights = DEFAULT_PERIOD_WEIGHTS

    # Step 1: 计算每个板块每个日期的原始动量
    board_momentum_raw: Dict[Tuple[str, str], float] = {}
    min_period = max(periods)

    for board_code, df in board_kline_dict.items():
        if df is None or df.empty or "close" not in df.columns:
            continue
        close = df["close"]
        for i in range(min_period, len(close)):
            date_str = close.index[i].strftime("%Y-%m-%d")
            raw_mom = 0.0
            valid = True
            for p, w in zip(periods, weights):
                if i < p:
                    valid = False
                    break
                ret = close.iloc[i] / close.iloc[i - p] - 1.0
                raw_mom += w * ret
            if valid:
                # 用 board_code 作为 key (板块名称可能有重复)
                board_momentum_raw[(date_str, board_code)] = raw_mom

    # Step 2: 按日期分组，横截面百分位排名
    by_date: Dict[str, List[Tuple[str, float]]] = {}
    for (date_str, board_code), raw_mom in board_momentum_raw.items():
        if date_str not in by_date:
            by_date[date_str] = []
        by_date[date_str].append((board_code, raw_mom))

    result: Dict[Tuple[str, str], float] = {}
    for date_str, items in by_date.items():
        if len(items) < 5:
            continue
        codes = [it[0] for it in items]
        raw_vals = np.array([it[1] for it in items])
        # 百分位排名: rank / (N-1), 值域 [0, 1]
        ranks = np.argsort(np.argsort(raw_vals)).astype(float)
        percentiles = ranks / (len(raw_vals) - 1)
        for code, pct in zip(codes, percentiles):
            result[(date_str, code)] = float(pct)

    return result


# ======================== 个股板块得分 ========================

def get_sector_score(
    code: str,
    date,
    stock_board_map: Dict[str, Dict[str, List[str]]],
    board_momentum: Dict[Tuple[str, str], float],
    board_list: Dict[str, Tuple[str, str]] = None,
    industry_weight: float = DEFAULT_INDUSTRY_WEIGHT,
    concept_weight: float = DEFAULT_CONCEPT_WEIGHT,
) -> float:
    """获取单只股票在某日的板块动量综合得分

    Args:
        code: 6位股票代码
        date: 日期 (pd.Timestamp 或 str YYYY-MM-DD)
        stock_board_map: build_stock_board_map 的输出
        board_momentum: compute_board_momentum 的输出
        board_list: get_board_list 的输出 (用于查找 code→name 映射)
        industry_weight/concept_weight: 行业/概念权重

    Returns:
        float [0.0, 1.0], 0.5 表示中性（无板块数据）
    """
    if isinstance(date, pd.Timestamp):
        date_str = date.strftime("%Y-%m-%d")
    else:
        date_str = str(date)[:10]

    board_info = stock_board_map.get(code)
    if board_info is None:
        return 0.5

    industries = board_info.get("industry", [])
    concepts = board_info.get("concept", [])

    # 建立 name → code 反向映射 (用于从动量字典查找)
    name_to_code = {}
    if board_list:
        for bd_code, (bd_name, _) in board_list.items():
            name_to_code[bd_name] = bd_code

    # 行业得分
    ind_scores = []
    for ind_name in industries:
        # 从名字找到代码，再从动量字典查找
        bd_code = name_to_code.get(ind_name, ind_name)
        score = board_momentum.get((date_str, bd_code))
        if score is not None:
            ind_scores.append(score)
    ind_mean = np.mean(ind_scores) if ind_scores else 0.5

    # 概念得分
    con_scores = []
    for con_name in concepts:
        bd_code = name_to_code.get(con_name, con_name)
        score = board_momentum.get((date_str, bd_code))
        if score is not None:
            con_scores.append(score)
    con_mean = np.mean(con_scores) if con_scores else 0.5

    return industry_weight * ind_mean + concept_weight * con_mean


# ======================== 主入口：预加载板块数据 ========================

def preload_sector_data(
    stock_data_dict: Dict[str, pd.DataFrame] = None,
    start: str = "20150101",
    end: str = None,
    force_refresh: bool = False,
    max_workers: int = 4,
) -> Dict:
    """预加载板块数据（带缓存）

    板块指数由成分股等权合成，完全避免 eastmoney API 封锁。

    首次调用合成数据并缓存到 output/sector_cache.pkl。
    后续调用直接从缓存加载（当天内有效）。

    Args:
        stock_data_dict: {stock_code: DataFrame(OHLCV)} 从回测引擎获取
            - 必须包含 'close' 列
            - 为 None 时只返回板块列表和映射（无动量数据）

    Returns:
        {
            'stock_board_map': {code: {'industry': [...], 'concept': [...]}},
            'board_momentum': {(date_str, board_code): percentile_rank},
            'board_list': {board_code: (board_name, board_type)},
        }
    """
    if end is None:
        end = pd.Timestamp.now().strftime("%Y%m%d")

    # 检查缓存
    if not force_refresh and os.path.exists(CACHE_PATH):
        age = time.time() - os.path.getmtime(CACHE_PATH)
        if age < CACHE_MAX_AGE_SEC:
            logger.info("从缓存加载板块数据: %s (%.0f min 前)", CACHE_PATH, age / 60)
            try:
                with open(CACHE_PATH, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning("缓存加载失败: %s，重新构建...", e)

    logger.info("=== 开始构建板块数据 ===")

    # 1. 获取板块列表
    board_list = get_board_list()
    logger.info(
        "板块列表: %d 个 (行业 %d, 概念 %d)",
        len(board_list),
        sum(1 for _, t in board_list.values() if t == 'industry'),
        sum(1 for _, t in board_list.values() if t == 'concept'),
    )

    # 2. 构建个股-板块映射
    stock_board_map = build_stock_board_map()

    # 3. 从成分股合成板块指数
    board_kline_dict = {}
    if stock_data_dict is not None:
        board_kline_dict = build_all_board_indices(
            stock_data_dict, max_workers=max_workers,
        )
    else:
        logger.warning("无 stock_data_dict，跳过板块指数合成")

    # 4. 计算板块动量
    board_momentum = {}
    if board_kline_dict:
        logger.info("计算板块多周期动量排名...")
        board_momentum = compute_board_momentum(board_kline_dict)
        logger.info(
            "板块动量计算完成: %d 个 (日期,板块) 得分, %d 个日期",
            len(board_momentum),
            len(set(d for d, _ in board_momentum)),
        )

    result = {
        "stock_board_map": stock_board_map,
        "board_momentum": board_momentum,
        "board_list": board_list,
    }

    # 5. 缓存
    if board_momentum:
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        try:
            with open(CACHE_PATH, "wb") as f:
                pickle.dump(result, f)
            logger.info("板块数据已缓存: %s", CACHE_PATH)
        except Exception as e:
            logger.warning("缓存写入失败: %s", e)

    return result


# ======================== 交互测试入口 ========================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    print("=" * 70)
    print("板块动量因子 - 交互测试")
    print("=" * 70)

    # 测试板块列表
    print("\n--- 板块列表 ---")
    board_list = get_board_list()
    industries = {c: n for c, (n, t) in board_list.items() if t == 'industry'}
    concepts = {c: n for c, (n, t) in board_list.items() if t == 'concept'}
    print(f"行业板块: {len(industries)}")
    for c, n in sorted(industries.items())[:5]:
        print(f"  {c}: {n}")
    print(f"   ...")
    print(f"概念板块: {len(concepts)}")
    for c, n in sorted(concepts.items())[:5]:
        print(f"  {c}: {n}")
    print(f"   ...")

    # 测试成分股覆盖
    print("\n--- 成分股覆盖 ---")
    total_stocks = set()
    for code, stocks in HARDCODED_ALL_CONSTITUENTS.items():
        total_stocks.update(stocks)
    print(f"硬编码成分股: {len(total_stocks)} 只唯一股票, 覆盖 {len(HARDCODED_ALL_CONSTITUENTS)} 个板块")

    # 测试个股-板块映射
    print("\n--- 个股-板块映射 ---")
    smap = build_stock_board_map()
    test_codes = ["688981", "300750", "002230", "601012", "000001"]
    for code in test_codes:
        info = smap.get(code, {})
        inds = info.get("industry", [])
        cons = info.get("concept", [])
        print(f"  {code}: 行业={inds[:3]}, 概念={cons[:5]}...")

    # 测试板块指数合成 (需要 stock data)
    print("\n--- 板块指数合成 (测试模式) ---")
    # 创建模拟股票数据
    import numpy as np
    dates = pd.date_range('2025-01-01', '2025-06-01', freq='B')
    mock_stock_data = {}
    for code in HARDCODED_ALL_CONSTITUENTS.get('BK1036', [])[:5]:
        np.random.seed(hash(code) % 2**31)
        returns = np.random.normal(0.001, 0.02, len(dates))
        close = 50 * np.cumprod(1 + returns)
        mock_stock_data[code] = pd.DataFrame({'close': close}, index=dates)

    index_df = build_board_index_from_stocks('BK1036', mock_stock_data)
    if not index_df.empty:
        print(f"BK1036 (半导体) 合成指数: {len(index_df)} 天, 最终值={index_df['close'].iloc[-1]:.1f}")
        # 计算动量
        mom = compute_board_momentum({'BK1036': index_df})
        recent = [(d, b, s) for (d, b), s in mom.items()]
        if recent:
            d, b, s = recent[-1]
            print(f"  最后日期 {d} 动量百分位: {s:.3f}")
    else:
        print("合成失败")

    print("\n=== 测试完成 ===")

"""
实验C (v6): SiteAgent推荐 vs 官方 MaxEnt v3.4.4 重叠分析
========================================================
【核心变更】: 完全替换 elapid，改为通过 subprocess 调用官方 maxent.jar
  - 算法与论文 MaxEnt v3.4.4 完全一致（同一个 jar 文件）
  - 输入格式: SWD (Samples With Data) CSV，最灵活且速度最快
  - 输出格式: cloglog（v3.4.0+ 默认，概率解释最清晰）
  - 读取 maxent 输出的 _samplePredictions.csv 获取目标点适宜性

【准备工作（只需做一次）】:
  Step A: 安装 Java（>=1.8，64位）
    https://www.java.com/download/
    验证: java -version

  Step B: 下载 maxent.jar（官方 GitHub）
    https://github.com/mrmaxent/Maxent/blob/master/ArchivedReleases/3.4.4/maxent.jar
    点击 "Download raw file"，保存为 maxent.jar
    放到与本脚本同目录，或修改下面 MAXENT_JAR 路径

  Step C: 安装 Python 依赖（均为纯 Python，无版本歧义）
    pip install requests rasterio numpy pandas matplotlib matplotlib-venn tqdm

依赖清单（无 elapid）:
  requests, rasterio, numpy, pandas, matplotlib, matplotlib-venn, tqdm
"""

import json, time, subprocess, shutil, sys, warnings
import requests
import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib_venn import venn2

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 0. 路径与全局参数配置
# ─────────────────────────────────────────────────────────────────────────────

# ★ 修改这里指向你的 maxent.jar 文件路径
MAXENT_JAR = Path("maxent.jar")

# MaxEnt 运行参数（与论文 MaxEnt v3.4.4 默认一致）
MAXENT_MEMORY_MB      = 1024          # -Xmx: 分配给 Java 的最大内存
MAXENT_N_BACKGROUND   = 10000         # 背景点数量（官方默认10000）
MAXENT_BETA_MULT      = 1.0           # betamultiplier 正则化倍数
MAXENT_MIN_PRESENCE   = 10            # 少于此数的物种跳过建模
SUITABILITY_THRESHOLD = 0.5           # cloglog 适宜性阈值

# WorldClim v2.1 单文件下载（2.5分辨率，每个约50MB）
WORLDCLIM_RESOLUTION = "2.5m"
WORLDCLIM_BASE_URL   = (
    f"https://geodata.ucdavis.edu/climate/worldclim/2_1/base/"
    f"wc2.1_{WORLDCLIM_RESOLUTION}_bio_"
)

# 中国全境 bounding box（GBIF查询 & 背景点采样均用此范围）
CHINA_BBOX = dict(lat_min=17.0, lat_max=54.0, lon_min=72.0, lon_max=135.0)

OUTPUT_DIR = Path("experiment_c_output")
WC_DIR     = OUTPUT_DIR / "worldclim"
GBIF_DIR   = OUTPUT_DIR / "gbif_data"
MAXENT_DIR = OUTPUT_DIR / "maxent_results"   # maxent.jar 的输出目录
SWD_DIR    = OUTPUT_DIR / "swd_inputs"       # SWD CSV 文件
FIG_DIR    = OUTPUT_DIR / "figures"

for d in [OUTPUT_DIR, WC_DIR, GBIF_DIR, MAXENT_DIR, SWD_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. 环境检查
# ─────────────────────────────────────────────────────────────────────────────

def check_environment():
    """启动时检查 Java 和 maxent.jar 是否就绪。"""
    # 检查 Java
    if shutil.which("java") is None:
        sys.exit(
            "\n[错误] 未找到 java 命令。请安装 Java 8+ (64位):\n"
            "  https://www.java.com/download/\n"
            "  安装后重启命令行，确认 java -version 可正常输出。"
        )
    result = subprocess.run(["java", "-version"], capture_output=True, text=True)
    java_ver = (result.stdout + result.stderr).split("\n")[0]
    print(f"[Java]  {java_ver}")

    # 检查 maxent.jar
    if not MAXENT_JAR.exists():
        sys.exit(
            f"\n[错误] 未找到 maxent.jar: {MAXENT_JAR.resolve()}\n"
            "请从官方 GitHub 下载:\n"
            "  https://github.com/mrmaxent/Maxent/blob/master/ArchivedReleases/3.4.4/maxent.jar\n"
            "点击 'Download raw file'，保存为 maxent.jar 放到本脚本同目录。"
        )
    print(f"[MaxEnt] maxent.jar 已就绪: {MAXENT_JAR.resolve()}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. 站点与物种配置（与原实验一致）
# ─────────────────────────────────────────────────────────────────────────────

SITES = {
    "延安": {"name_en": "Yan'an",    "lat": 36.60, "lon": 109.49,
             "climate_zone": "semi-arid Loess Plateau",    "buffer_deg": 3.0},
    "韶关": {"name_en": "Shaoguan",  "lat": 24.80, "lon": 113.60,
             "climate_zone": "subtropical South China",    "buffer_deg": 2.5},
    "伊春": {"name_en": "Yichun",    "lat": 47.73, "lon": 128.91,
             "climate_zone": "boreal Northeast China",     "buffer_deg": 3.5},
}

CANDIDATE_SPECIES = {
    "延安": [
        "Platycladus orientalis", "Robinia pseudoacacia", "Pinus tabuliformis",
        "Amorpha fruticosa",      "Caragana korshinskii", "Hippophae rhamnoides",
        "Ulmus pumila",           "Populus davidiana",    "Pinus armandii",
        "Prunus sibirica",
    ],
    "韶关": [
        "Cunninghamia lanceolata", "Schima superba",     "Castanopsis hystrix",
        "Pinus massoniana",        "Liquidambar formosana", "Eucalyptus urophylla",
        "Betula alnoides",         "Mytilaria laosensis", "Michelia chapensis",
        "Cinnamomum camphora",
    ],
    "伊春": [
        "Larix gmelinii",       "Pinus koraiensis",    "Betula platyphylla",
        "Picea koraiensis",     "Fraxinus mandshurica","Abies nephrolepis",
        "Quercus mongolica",    "Tilia amurensis",     "Populus davidiana",
        "Pinus sylvestris",     # 樟子松; GBIF不支持变种名，用种名
    ],
}

SITEAGENT_TOP5 = {
    "延安": ["Platycladus orientalis", "Robinia pseudoacacia", "Pinus tabuliformis",
             "Amorpha fruticosa",      "Caragana korshinskii"],
    "韶关": ["Cunninghamia lanceolata", "Schima superba",      "Castanopsis hystrix",
             "Pinus massoniana",        "Liquidambar formosana"],
    "伊春": ["Larix gmelinii",  "Pinus koraiensis", "Betula platyphylla",
             "Picea koraiensis","Fraxinus mandshurica"],
}

SPECIES_CN = {
    "Platycladus orientalis": "侧柏",    "Robinia pseudoacacia": "刺槐",
    "Pinus tabuliformis": "油松",         "Amorpha fruticosa": "紫穗槐",
    "Caragana korshinskii": "柠条锦鸡儿","Hippophae rhamnoides": "沙棘",
    "Ulmus pumila": "榆树",               "Populus davidiana": "山杨",
    "Pinus armandii": "华山松",           "Prunus sibirica": "山杏",
    "Cunninghamia lanceolata": "杉木",    "Schima superba": "荷木",
    "Castanopsis hystrix": "红锥",        "Pinus massoniana": "马尾松",
    "Liquidambar formosana": "枫香",      "Eucalyptus urophylla": "尾叶桉",
    "Betula alnoides": "西南桦",          "Mytilaria laosensis": "米老排",
    "Michelia chapensis": "乐昌含笑",     "Cinnamomum camphora": "樟树",
    "Larix gmelinii": "兴安落叶松",       "Pinus koraiensis": "红松",
    "Betula platyphylla": "白桦",         "Picea koraiensis": "红皮云杉",
    "Fraxinus mandshurica": "水曲柳",     "Abies nephrolepis": "臭冷杉",
    "Quercus mongolica": "蒙古栎",        "Tilia amurensis": "紫椴",
    "Pinus sylvestris": "樟子松",
}


# ─────────────────────────────────────────────────────────────────────────────
# 3. WorldClim 下载与栅格读取（与原实验相同逻辑）
# ─────────────────────────────────────────────────────────────────────────────

def download_single_bio(bio_index: int) -> Path:
    tif_name = f"wc2.1_{WORLDCLIM_RESOLUTION}_bio_{bio_index}.tif"
    tif_path = WC_DIR / tif_name
    if tif_path.exists() and tif_path.stat().st_size > 1_000_000:
        return tif_path
    url = WORLDCLIM_BASE_URL + f"{bio_index}.tif"
    print(f"  [下载] bio{bio_index}: {url}")
    try:
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(tif_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=f"bio{bio_index}", leave=False
        ) as bar:
            for chunk in resp.iter_content(chunk_size=1 << 18):
                f.write(chunk); bar.update(len(chunk))
    except Exception as e:
        if tif_path.exists(): tif_path.unlink()
        raise RuntimeError(f"bio{bio_index} 下载失败: {e}\n  URL: {url}")
    return tif_path


def get_worldclim_rasters() -> list:
    print("[WorldClim] 检查/下载 bio1...bio19 ...")
    paths = [download_single_bio(i) for i in range(1, 20)]
    print(f"  全部 {len(paths)} 个 TIF 就绪")
    return paths


def extract_worldclim_values(lats, lons, raster_paths) -> np.ndarray:
    """批量提取坐标处的 19 维 WorldClim 值，NaN 表示无数据。"""
    lats, lons = np.asarray(lats, float), np.asarray(lons, float)
    X = np.full((len(lats), 19), np.nan, dtype=np.float32)
    for col, tif_path in enumerate(raster_paths):
        with rasterio.open(tif_path) as src:
            nodata  = src.nodata
            coords  = list(zip(lons.tolist(), lats.tolist()))
            sampled = np.array([v[0] for v in src.sample(coords, indexes=1)], dtype=np.float32)
            if nodata is not None:
                sampled[sampled == nodata] = np.nan
            X[:, col] = sampled
    return X


# ─────────────────────────────────────────────────────────────────────────────
# 4. GBIF 数据获取（全国范围，带磁盘缓存）
# ─────────────────────────────────────────────────────────────────────────────

def fetch_gbif_occurrences(species_name: str, limit: int = 500) -> pd.DataFrame:
    url    = "https://api.gbif.org/v1/occurrence/search"
    params = {
        "scientificName":     species_name,
        "decimalLatitude":    f"{CHINA_BBOX['lat_min']},{CHINA_BBOX['lat_max']}",
        "decimalLongitude":   f"{CHINA_BBOX['lon_min']},{CHINA_BBOX['lon_max']}",
        "hasCoordinate":      "true",
        "hasGeospatialIssue": "false",
        "country":            "CN",
        "limit":              limit,
    }
    try:
        resp    = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        records = resp.json().get("results", [])
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame([{
            "species":          r.get("scientificName", species_name),
            "decimalLatitude":  r.get("decimalLatitude"),
            "decimalLongitude": r.get("decimalLongitude"),
        } for r in records]).dropna(subset=["decimalLatitude", "decimalLongitude"])
        return df
    except Exception as e:
        print(f"    [GBIF 警告] {species_name}: {e}")
        return pd.DataFrame()


def load_or_fetch_gbif(species_name: str) -> pd.DataFrame:
    safe = species_name.replace(" ", "_")
    cache = GBIF_DIR / f"CN_{safe}.csv"
    if cache.exists() and cache.stat().st_size > 50:
        return pd.read_csv(cache)
    df = fetch_gbif_occurrences(species_name)
    if not df.empty:
        df.to_csv(cache, index=False)
    time.sleep(0.35)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. 构造 SWD 格式 CSV（maxent.jar 的标准输入格式）
# ─────────────────────────────────────────────────────────────────────────────
# SWD (Samples With Data) 格式规范（来自官方文档）:
#
#   列顺序（严格）: species, x, y, bio1, bio2, ..., bio19
#     - species: 物种名（出现点文件）或任意字符串（背景点/投影文件）
#     - x:       经度 longitude（东西方向）
#     - y:       纬度 latitude（南北方向）
#     注意: x=lon, y=lat，不能写反！
#
#   出现点文件 (samplesfile):          species 列 = 物种学名
#   背景点文件 (environmentallayers):  species 列 = "background"
#   投影文件   (projectionlayers):     格式与背景点完全相同
#
# 目标点预测的正确方式:
#   ❌ 错误: 把目标点混入 samplesfile（当出现点参与训练，会污染模型）
#   ✅ 正确: 把目标点写入独立投影SWD文件，通过 projectionlayers 参数传入
#            maxent 训练完成后对投影文件每个点打分
#            结果写入 <out_dir>/<species>_<projname>.csv

BIO_COLS = [f"bio{i}" for i in range(1, 20)]


def _build_swd_df(species_col, lons, lats, env_X) -> pd.DataFrame:
    """
    内部辅助：按 maxent.jar SWD 规范构造 DataFrame。
    列顺序严格为: species | x(lon) | y(lat) | bio1...bio19
    insert(0,...) 从左插入，所以顺序是先插最右侧的列。
    """
    df = pd.DataFrame(env_X, columns=BIO_COLS)
    df.insert(0, "y",       lats)          # 步骤1: 插到最左 → [y, bio1..19]
    df.insert(0, "x",       lons)          # 步骤2: 插到最左 → [x, y, bio1..19]
    df.insert(0, "species", species_col)   # 步骤3: 插到最左 → [species, x, y, bio1..19] ✓
    return df


def make_presence_swd(species_name: str, occ_df: pd.DataFrame,
                       raster_paths: list):
    """
    从 GBIF 出现点提取 WorldClim 特征，构造符合 SWD 规范的出现点 DataFrame。
    过滤无数据格，记录不足则返回 (None, 0)。
    返回: (DataFrame, n_valid) 或 (None, 0)
    """
    pres_X = extract_worldclim_values(
        occ_df["decimalLatitude"].values,
        occ_df["decimalLongitude"].values,
        raster_paths,
    )
    valid  = ~np.isnan(pres_X).any(axis=1)
    n_bad  = int((~valid).sum())
    pres_X = pres_X[valid]
    lats   = occ_df["decimalLatitude"].values[valid].astype(float)
    lons   = occ_df["decimalLongitude"].values[valid].astype(float)

    if n_bad:
        print(f"      [诊断] {n_bad} 个出现点落在无数据格，已过滤")
    if len(pres_X) < MAXENT_MIN_PRESENCE:
        print(f"      -> SKIP: 有效出现点仅 {len(pres_X)} < {MAXENT_MIN_PRESENCE}")
        return None, 0

    df = _build_swd_df(species_name, lons, lats, pres_X)
    return df, len(pres_X)


def make_background_swd(raster_paths: list, seed: int = 42) -> pd.DataFrame:
    """
    从中国全境随机采样 MAXENT_N_BACKGROUND 个背景点，提取 WorldClim 特征。
    多采样 4 倍以弥补海洋/无数据像元损耗。

    方法论: 背景点范围必须与出现点来源范围一致（全国），
    不能局限于目标地点小框，否则背景分布失真，MaxEnt 高估局部适宜性。
    """
    rng      = np.random.default_rng(seed)
    n_sample = MAXENT_N_BACKGROUND * 4
    bg_lats  = rng.uniform(CHINA_BBOX["lat_min"], CHINA_BBOX["lat_max"], n_sample)
    bg_lons  = rng.uniform(CHINA_BBOX["lon_min"], CHINA_BBOX["lon_max"], n_sample)

    bg_X  = extract_worldclim_values(bg_lats, bg_lons, raster_paths)
    valid = ~np.isnan(bg_X).any(axis=1)
    bg_X  = bg_X[valid]
    bg_lats, bg_lons = bg_lats[valid], bg_lons[valid]

    if len(bg_X) < 100:
        raise ValueError(f"有效背景点不足 ({len(bg_X)})，请检查 WorldClim TIF 文件")

    n  = min(len(bg_X), MAXENT_N_BACKGROUND)
    df = _build_swd_df("background", bg_lons[:n], bg_lats[:n], bg_X[:n])
    return df


def make_bg_with_target(bg_swd_df: pd.DataFrame,
                         lat: float, lon: float,
                         raster_paths: list):
    """
    在全局背景点末尾追加目标点，构造"背景点+目标点"合并SWD文件。

    为什么这样做（而不用 projectionlayers）:
      projectionlayers 在 SWD 模式下的输出文件名依赖文件名拼接规则，
      不同版本间行为有差异，路径极难可靠匹配。

    正确且稳定的方案：
      - 把目标点追加进 environmentallayers（背景点文件）末尾
      - 开启 writebackgroundpredictions
      - maxent 把所有背景点（含目标点）的预测值写入
        <out_dir>/<species>_backgroundPredictions.csv
      - Python 按坐标匹配找到目标点那行，读 cloglog 值

    注：背景点不参与 λ 参数估计，追加目标点不影响训练结果。
    """
    target_X = extract_worldclim_values([lat], [lon], raster_paths)
    if np.isnan(target_X).any():
        nan_cols = [f"bio{i+1}" for i, v in enumerate(target_X[0]) if np.isnan(v)]
        print(f"      [诊断] 目标点 ({lat},{lon}) NaN 变量: {nan_cols}")
        return None
    target_row = _build_swd_df("background", [lon], [lat], target_X)
    return pd.concat([bg_swd_df, target_row], ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# 6. 调用官方 maxent.jar（subprocess）
# ─────────────────────────────────────────────────────────────────────────────

def run_maxent_jar(
    species_name:   str,
    samples_csv:    Path,
    background_csv: Path,   # 背景点文件（末尾已追加目标点）
    out_dir:        Path,
) -> bool:
    """
    通过 subprocess 调用 maxent.jar，SWD 模式训练。

    目标点预测策略:
      背景点文件末尾追加了目标点。开启 writebackgroundpredictions，
      maxent 把所有背景点（含目标点）的预测值写入：
        <out_dir>/<species>_backgroundPredictions.csv
      Python 按坐标从该文件读出目标点的 cloglog 分数。
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "java", f"-Xmx{MAXENT_MEMORY_MB}m",
        "-jar", str(MAXENT_JAR.resolve()),
        f"samplesfile={samples_csv.resolve()}",
        f"environmentallayers={background_csv.resolve()}",
        f"outputdirectory={out_dir.resolve()}",
        "outputformat=cloglog",
        f"betamultiplier={MAXENT_BETA_MULT}",
        "writebackgroundpredictions",   # ★ 输出背景点（含目标点）的预测CSV
        "nooutputgrids",
        "autorun",
        "nowarnings",
        "nopictures",
        "noaskoverwrite",
        "removeduplicates",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"      [MaxEnt 错误] returncode={result.returncode}")
            for line in (result.stderr + result.stdout).strip().split("\n")[-8:]:
                if line.strip():
                    print(f"        {line}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"      [MaxEnt 超时] 超过 600 秒，跳过")
        return False
    except Exception as e:
        print(f"      [MaxEnt 异常] {e}")
        return False


def read_suitability_from_background(out_dir: Path, species_name: str,
                                      target_lat: float, target_lon: float) -> float:
    """
    从 maxent.jar 输出的 _backgroundPredictions.csv 读取目标点 cloglog 分数。

    文件路径: <out_dir>/<species>_backgroundPredictions.csv
    列名示例: species, longitude, latitude, ..., Cloglog prediction
    目标点是背景文件的最后一行，用坐标最近邻匹配定位。
    """
    safe_sp = species_name.replace(" ", "_")
    bg_pred = out_dir / f"{safe_sp}_backgroundPredictions.csv"

    if not bg_pred.exists():
        existing = [f.name for f in out_dir.glob("*.csv")]
        print(f"      [读取失败] 未找到 _backgroundPredictions.csv，目录CSV: {existing}")
        return 0.0

    try:
        df = pd.read_csv(bg_pred)

        # 找 cloglog 预测列
        score_col = next((c for c in df.columns if "cloglog"  in c.lower()), None) or \
                    next((c for c in df.columns if "logistic" in c.lower()), None)
        if score_col is None:
            print(f"      [读取失败] 未找到分数列，列名: {list(df.columns)}")
            return 0.0

        # 找坐标列
        lat_col = next((c for c in df.columns if c.lower() in ("latitude",  "lat", "y")), None)
        lon_col = next((c for c in df.columns if c.lower() in ("longitude", "lon", "x")), None)

        if lat_col and lon_col:
            dist = (df[lat_col].astype(float) - target_lat)**2 + \
                   (df[lon_col].astype(float) - target_lon)**2
            idx = dist.idxmin()
            val = float(df.loc[idx, score_col])
        else:
            val = float(df[score_col].iloc[-1])   # 目标点在末尾

        print(f"      [背景预测OK] 列='{score_col}' 目标点 cloglog={val:.4f}")
        return float(np.clip(val, 0.0, 1.0))
    except Exception as e:
        print(f"      [读取异常] {e}")
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 7. 单物种完整建模流程
# ─────────────────────────────────────────────────────────────────────────────

def run_maxent_for_species(
    species_name:  str,
    occ_df:        pd.DataFrame,
    lat_center:    float,
    lon_center:    float,
    raster_paths:  list,
    bg_swd_df:     pd.DataFrame,
) -> float:
    """
    单物种 MaxEnt 建模主流程:
      1. 构造出现点 SWD（纯训练数据）
      2. 背景点文件末尾追加目标点，构造合并背景SWD
      3. 调用 maxent.jar，开启 writebackgroundpredictions
      4. 从 _backgroundPredictions.csv 按坐标读出目标点 cloglog 分数
    """
    sp_safe = species_name.replace(" ", "_")
    n_raw   = len(occ_df)

    if n_raw < MAXENT_MIN_PRESENCE:
        print(f"    -> SKIP: GBIF 全国记录仅 {n_raw} < {MAXENT_MIN_PRESENCE}")
        return 0.0

    # ── 7-A. 构造出现点 SWD ──────────────────────────────────────────────────
    pres_swd, n_valid = make_presence_swd(species_name, occ_df, raster_paths)
    if pres_swd is None:
        return 0.0

    # ── 7-B. 背景点末尾追加目标点 ────────────────────────────────────────────
    bg_with_target = make_bg_with_target(bg_swd_df, lat_center, lon_center, raster_paths)
    if bg_with_target is None:
        return 0.0

    # ── 7-C. 写入 SWD CSV 文件 ───────────────────────────────────────────────
    sp_dir  = SWD_DIR / sp_safe
    sp_dir.mkdir(exist_ok=True)
    samples_csv = sp_dir / f"{sp_safe}_presence.csv"
    bg_csv      = sp_dir / f"{sp_safe}_background_with_target.csv"

    pres_swd.to_csv(samples_csv, index=False)
    bg_with_target.to_csv(bg_csv, index=False)

    # ── 7-D. 调用 maxent.jar ─────────────────────────────────────────────────
    out_dir = MAXENT_DIR / sp_safe
    success = run_maxent_jar(species_name, samples_csv, bg_csv, out_dir)
    if not success:
        return 0.0

    # ── 7-E. 从 backgroundPredictions.csv 读取目标点适宜性 ───────────────────
    return read_suitability_from_background(out_dir, species_name, lat_center, lon_center)


# ─────────────────────────────────────────────────────────────────────────────
# 8. 计算 T_SDM（对一个地点的所有候选物种建模）
# ─────────────────────────────────────────────────────────────────────────────

def compute_maxent_positive_set(
    site_name:    str,
    site_cfg:     dict,
    species_list: list,
    raster_paths: list,
    bg_swd_df:    pd.DataFrame,
    threshold:    float = SUITABILITY_THRESHOLD,
):
    """为地点内所有候选物种运行 maxent.jar，返回 T_SDM 集合与分数字典。"""
    print(f"\n[MaxEnt/官方jar] {site_name} ({site_cfg['name_en']})")
    print(f"  参数: betamultiplier={MAXENT_BETA_MULT}, n_bg={MAXENT_N_BACKGROUND}, "
          f"threshold={threshold}, outputformat=cloglog")

    cache_file = MAXENT_DIR / f"{site_name}_suitability.json"
    scores = None
    if cache_file.exists() and cache_file.stat().st_size > 10:
        try:
            with open(cache_file) as fh:
                loaded = json.load(fh)
            if loaded:
                scores = loaded
                print(f"  [缓存] 已加载 {len(scores)} 个物种")
        except (json.JSONDecodeError, Exception) as e:
            print(f"  [警告] 缓存读取失败 ({e})，重新建模")

    if scores is None:
        scores = {}
        for sp in species_list:
            occ_df = load_or_fetch_gbif(sp)
            n_occ  = len(occ_df)
            print(f"  {sp:<38s} n_GBIF={n_occ:<4d}", end="  ")

            prob = run_maxent_for_species(
                species_name=sp,
                occ_df=occ_df,
                lat_center=site_cfg["lat"],
                lon_center=site_cfg["lon"],
                raster_paths=raster_paths,
                bg_swd_df=bg_swd_df,
            )
            scores[sp] = round(float(prob), 4)
            flag = "-> T_SDM ✓" if prob > threshold else f"-> {prob:.4f}"
            print(f"cloglog={prob:.4f}  {flag}")

        with open(cache_file, "w") as fh:
            json.dump(scores, fh, ensure_ascii=False, indent=2)

    t_sdm = {sp for sp, p in scores.items() if p > threshold}
    print(f"\n  T_SDM (cloglog > {threshold}): {len(t_sdm)} 个物种")
    for sp in sorted(t_sdm):
        print(f"    - {sp} ({SPECIES_CN.get(sp,'')})  cloglog={scores[sp]:.4f}")
    if not t_sdm:
        print("  [提示] T_SDM 为空，原始分数:")
        print(f"    {json.dumps(scores, indent=6, ensure_ascii=False)}")
    return t_sdm, scores


# ─────────────────────────────────────────────────────────────────────────────
# 9. 重叠率、表六、图11（与原实验相同）
# ─────────────────────────────────────────────────────────────────────────────

def compute_overlap_rate(t_agent: set, t_sdm: set) -> dict:
    """OR = |T_Agent ∩ T_SDM| / |T_SDM| × 100%（论文公式19）"""
    intersection = t_agent & t_sdm
    union        =  t_sdm
    or_pct = len(intersection) / len(union) * 100 if union else 0.0
    return {
        "T_Agent": t_agent, "T_SDM": t_sdm,
        "intersection": intersection, "union": union,
        "only_in_agent": t_agent - t_sdm,
        "only_in_sdm":   t_sdm  - t_agent,
        "OR":       round(or_pct, 1),
        "|T_Agent|": len(t_agent), "|T_SDM|": len(t_sdm),
        "|∩|": len(intersection), "|∪|": len(union),
    }


def generate_table_vi(results: dict) -> pd.DataFrame:
    rows = []
    for site_cn, res in results.items():
        rows.append({
            "地点":              f"{site_cn} ({SITES[site_cn]['name_en']})",
            "气候区":            SITES[site_cn]["climate_zone"],
            "|T_Agent|":        res["|T_Agent|"],
            "|T_SDM|":          res["|T_SDM|"],
            "|T_Agent ∩ T_SDM|":res["|∩|"],
            "|T_Agent ∪ T_SDM|":res["|∪|"],
            "OR (%)":           res["OR"],
            "重叠物种":          "; ".join(sorted(res["intersection"])),
            "仅SiteAgent推荐":   "; ".join(sorted(res["only_in_agent"])),
            "仅MaxEnt预测":      "; ".join(sorted(res["only_in_sdm"])),
        })
    return pd.DataFrame(rows)


def print_table_vi(df: pd.DataFrame):
    print("\n" + "=" * 90)
    print("表六. SiteAgent推荐与MaxEnt预测(官方jar)的重叠分析")
    print("=" * 90)
    cols = ["地点","气候区","|T_Agent|","|T_SDM|","|T_Agent ∩ T_SDM|","|T_Agent ∪ T_SDM|","OR (%)"]
    print(df[cols].to_string(index=False))
    print()
    for _, row in df.iterrows():
        print(f">> {row['地点']}  OR={row['OR (%)']}%")
        print(f"   [重叠]     {row['重叠物种']        or '（无）'}")
        print(f"   [仅Agent]  {row['仅SiteAgent推荐'] or '（无）'}")
        print(f"   [仅MaxEnt] {row['仅MaxEnt预测']    or '（无）'}")
    print("=" * 90)


def generate_figure_11(results: dict, out_path: str):
    C_AGENT, C_MAXENT, C_BOTH = "#4472C4", "#70AD47", "#ED7D31"
    fig, axes = plt.subplots(1, 3, figsize=(21, 8))
    fig.patch.set_facecolor("#f8f8f8")
    for ax, site_cn in zip(axes, results.keys()):
        res     = results[site_cn]
        only_a  = sorted(res["only_in_agent"])
        overlap = sorted(res["intersection"])
        only_s  = sorted(res["only_in_sdm"])
        v = venn2(
            subsets=(max(len(only_a),1), max(len(only_s),1), len(overlap)),
            set_labels=("SiteAgent\nTop-5",
                        f"MaxEnt jar\n(cloglog>{SUITABILITY_THRESHOLD})"),
            set_colors=(C_AGENT, C_MAXENT), alpha=0.45, ax=ax,
        )
        for lbl in v.set_labels:
            if lbl: lbl.set_fontsize(9.5); lbl.set_fontweight("bold")
        for lbl in v.subset_labels:
            if lbl: lbl.set_text("")
        ax.set_facecolor("#f8f8f8")
        if only_a:
            ax.annotate(
                "\n".join(f"* {SPECIES_CN.get(sp,sp)}" for sp in only_a),
                xy=(-0.38,0), xytext=(-0.82,0), xycoords="data", textcoords="data",
                fontsize=7.2, color=C_AGENT, va="center", ha="right",
                arrowprops=dict(arrowstyle="-", color=C_AGENT, lw=0.8),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_AGENT, alpha=0.92, lw=1.2))
        if overlap:
            ax.text(0, 0, "\n".join(f"★ {SPECIES_CN.get(sp,sp)}" for sp in overlap),
                    ha="center", va="center", fontsize=7.5, color="#1a1a1a",
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.35", fc="lightyellow", ec=C_BOTH, alpha=0.97, lw=1.4))
        if only_s:
            ax.annotate(
                "\n".join(f"* {SPECIES_CN.get(sp,sp)}" for sp in only_s),
                xy=(0.38,0), xytext=(0.82,0), xycoords="data", textcoords="data",
                fontsize=7.2, color=C_MAXENT, va="center", ha="left",
                arrowprops=dict(arrowstyle="-", color=C_MAXENT, lw=0.8),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_MAXENT, alpha=0.92, lw=1.2))
        or_color = "#c0392b" if res["OR"] < 60 else "#27ae60" if res["OR"] > 70 else "#e67e22"
        ax.text(0.97, 0.97, f"OR = {res['OR']}%\n|∩| = {res['|∩|']}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9.5, color=or_color, fontweight="bold",
                bbox=dict(boxstyle="round", fc="white", ec=or_color, lw=1.5))
        ax.set_title(f"{site_cn}  ({SITES[site_cn]['name_en']})\n{SITES[site_cn]['climate_zone']}",
                     fontsize=10.5, fontweight="bold", pad=8)
    fig.suptitle(
        "Figure 11. Overlap between SiteAgent Top-5 Recommendations "
        "and Official MaxEnt v3.4.4 High-Suitability Species\n"
        f"Blue=SiteAgent only  |  ★=both sets  |  Green=MaxEnt only  "
        f"(cloglog threshold={SUITABILITY_THRESHOLD})",
        fontsize=10.5, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"[图11] 保存至: {out_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 10. 主流程
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("实验C: SiteAgent vs MaxEnt (官方 maxent.jar v3.4.4)")
    print("=" * 70)

    # Step 0: 环境检查
    check_environment()

    # Step 1: WorldClim 准备
    print("\n[Step 1] 准备 WorldClim v2.1 BioClim 变量...")
    raster_paths = get_worldclim_rasters()

    # Step 2: 预先构造全局背景点 SWD（所有地点共用，与原实验方法论一致）
    bg_cache = SWD_DIR / "global_background.csv"
    if bg_cache.exists() and bg_cache.stat().st_size > 1000:
        print("\n[Step 2] 加载背景点缓存...")
        bg_swd_df = pd.read_csv(bg_cache)
        print(f"  背景点: {len(bg_swd_df)} 行")
    else:
        print("\n[Step 2] 构造全国背景点 SWD...")
        bg_swd_df = make_background_swd(raster_paths)
        bg_swd_df.to_csv(bg_cache, index=False)
        print(f"  背景点: {len(bg_swd_df)} 行，已缓存至 {bg_cache}")

    # Step 3: 逐地点建模
    all_results = {}
    for site_cn, site_cfg in SITES.items():
        print(f"\n{'─'*65}")
        print(f">> {site_cn} ({site_cfg['name_en']}) | {site_cfg['climate_zone']}")

        t_agent       = set(SITEAGENT_TOP5[site_cn])
        t_sdm, scores = compute_maxent_positive_set(
            site_name    = site_cn,
            site_cfg     = site_cfg,
            species_list = CANDIDATE_SPECIES[site_cn],
            raster_paths = raster_paths,
            bg_swd_df    = bg_swd_df,
            threshold    = SUITABILITY_THRESHOLD,
        )
        res = compute_overlap_rate(t_agent, t_sdm)
        all_results[site_cn] = res
        print(f"\n  T_Agent = {sorted(t_agent)}")
        print(f"  T_SDM   = {sorted(t_sdm)}")
        print(f"  ∩       = {sorted(res['intersection'])}")
        print(f"  OR      = {res['OR']}%")

    # Step 4: 输出表六
    df_t6   = generate_table_vi(all_results)
    t6_path = OUTPUT_DIR / "Table_VI_overlap_analysis.csv"
    print_table_vi(df_t6)
    df_t6.to_csv(t6_path, index=False, encoding="utf-8-sig")
    print(f"\n[表六] 保存至: {t6_path}")

    # Step 5: 输出图11
    fig11_path = str(FIG_DIR / "Fig11_venn_diagrams.png")
    generate_figure_11(all_results, fig11_path)

    # Step 6: 汇总
    print("\n" + "=" * 70)
    print("实验C 完成")
    print("=" * 70)
    print(f"{'地点':<8}  {'OR(%)':<8}  {'|T_Agent|':<10}  {'|T_SDM|':<10}  {'|∩|'}")
    print("-" * 55)
    for s, r in all_results.items():
        print(f"{s:<8}  {r['OR']:<8}  {r['|T_Agent|']:<10}  {r['|T_SDM|']:<10}  {r['|∩|']}")
    ors = [r["OR"] for r in all_results.values()]
    print(f"\n平均 OR: {np.mean(ors):.1f}%   范围: {min(ors):.1f}% ~ {max(ors):.1f}%")
    print(f"论文预期: 55.3% (伊春) ~ 72.1% (韶关)")
    print(f"\n输出文件:")
    print(f"  📊  {t6_path}")
    print(f"  🖼  {fig11_path}")
    print(f"  📁  maxent输出: {MAXENT_DIR}/")
    print(f"  📁  SWD输入:    {SWD_DIR}/")
    print(f"  📁  GBIF缓存:   {GBIF_DIR}/")
    print(f"  📁  WorldClim:  {WC_DIR}/")


if __name__ == "__main__":
    main()
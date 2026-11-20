"""Fungsi data preparation proyek: prediksi kemunculan hotspot (titik api) H+1.

Dipakai oleh notebook 01_learning_problem_eda dan 02_data_preparation agar
definisi target & fitur identik di kedua notebook (reproducible).
"""
from pathlib import Path
import numpy as np
import pandas as pd

# ----------------------------------------------------------------- konfigurasi
SEED = 42
RADIUS_KM = 25            # radius pencarian hotspot di sekitar titik cuaca
HORIZON_DAYS = 1          # prediksi H+1 (besok)
DROP_CONFIDENCE = {"l"}   # buang deteksi VIIRS berkepercayaan rendah
KEEP_TYPE = 0             # 0 = vegetation fire (buang volcano/static/offshore)
WEATHER_WARMUP_DAYS = 30  # jendela rolling terpanjang (30 hari)

# batas split KRONOLOGIS berdasarkan tanggal target (hari yang diprediksi)
TRAIN_END = pd.Timestamp("2022-12-31")
VAL_START, VAL_END = pd.Timestamp("2023-01-01"), pd.Timestamp("2023-12-31")
TEST_START = pd.Timestamp("2024-01-01")

RAW_WEATHER_COLS = [
    "precipitation_sum", "temperature_2m_max", "et0_fao_evapotranspiration",
    "relative_humidity_2m_min", "wind_speed_10m_max",
    "vapour_pressure_deficit_max", "soil_moisture_0_to_7cm_mean",
    "soil_moisture_7_to_28cm_mean",
]


def find_raw_dir() -> Path:
    for cand in [Path("../data/raw"), Path("data/raw")]:
        if cand.exists():
            return cand
    raise FileNotFoundError("Folder data/raw tidak ditemukan")


# ----------------------------------------------------------------- loading
def haversine_km(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = (np.sin((lat2 - lat1) * p / 2) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def load_weather(path) -> pd.DataFrame:
    w = pd.read_csv(path)
    w["date"] = pd.to_datetime(w["date"], utc=True).dt.tz_localize(None).dt.normalize()
    pts = (w[["latitude", "longitude"]].drop_duplicates()
           .sort_values(["longitude", "latitude"]).reset_index(drop=True))
    pts["loc_id"] = [f"L{i+1}" for i in range(len(pts))]
    w = w.merge(pts, on=["latitude", "longitude"], how="left")
    return w.sort_values(["loc_id", "date"]).reset_index(drop=True)


def location_table(weather: pd.DataFrame) -> pd.DataFrame:
    return (weather[["loc_id", "latitude", "longitude"]].drop_duplicates()
            .reset_index(drop=True))


def load_viirs(path) -> pd.DataFrame:
    v = pd.read_csv(path)
    v["acq_date"] = pd.to_datetime(v["acq_date"])
    return v


def filter_viirs(v: pd.DataFrame) -> pd.DataFrame:
    """Hanya kebakaran vegetasi (type==0) dan buang confidence 'low'."""
    return v[(v["type"] == KEEP_TYPE) & (~v["confidence"].isin(DROP_CONFIDENCE))]


# ----------------------------------------------------------------- hotspot harian
def gap_days(v_all: pd.DataFrame, start, end) -> pd.DatetimeIndex:
    """Hari dengan 0 hotspot di SELURUH Indonesia = indikasi data satelit hilang."""
    per_day = v_all.groupby("acq_date").size()
    idx = pd.date_range(start, end)
    per_day = per_day.reindex(idx).fillna(0)
    return per_day[per_day == 0].index


def daily_hotspots(v_filt: pd.DataFrame, locs: pd.DataFrame, start, end,
                   radius_km=RADIUS_KM) -> pd.DataFrame:
    """Jumlah hotspot & total FRP per (lokasi, hari) dalam radius `radius_km`."""
    idx = pd.date_range(start, end)
    out = []
    lat, lon = v_filt["latitude"].values, v_filt["longitude"].values
    for _, r in locs.iterrows():
        d = haversine_km(r.latitude, r.longitude, lat, lon)
        s = v_filt.loc[d <= radius_km]
        g = s.groupby("acq_date").agg(hs_count=("frp", "size"), hs_frp_sum=("frp", "sum"))
        g = g.reindex(idx).fillna(0.0)
        g.index.name = "date"
        g["loc_id"] = r.loc_id
        out.append(g.reset_index())
    return pd.concat(out, ignore_index=True)


# ----------------------------------------------------------------- fitur
def _consecutive_run(mask: pd.Series) -> pd.Series:
    """Panjang run berturut-turut True sampai hari ini (0 jika False)."""
    grp = (~mask).cumsum()
    return mask.groupby(grp).cumsum()


def _days_since(event: pd.Series, cap: int) -> pd.Series:
    """Hari sejak event terakhir (0 = event hari ini); dibatasi `cap`."""
    last = pd.Series(np.where(event, np.arange(len(event)), np.nan), index=event.index).ffill()
    d = np.arange(len(event)) - last
    return d.fillna(cap).clip(upper=cap)


def add_hotspot_history(hs: pd.DataFrame, gaps) -> pd.DataFrame:
    """Fitur riwayat hotspot (hanya memakai informasi hari <= t)."""
    hs = hs.sort_values(["loc_id", "date"]).copy()
    gap_mask = hs["date"].isin(gaps)
    hs.loc[gap_mask, ["hs_count", "hs_frp_sum"]] = np.nan   # data hilang != 0 hotspot
    parts = []
    for loc, g in hs.groupby("loc_id", sort=False):
        g = g.copy()
        c, f = g["hs_count"], g["hs_frp_sum"]
        g["hs_count_7d"] = c.rolling(7, min_periods=1).sum()
        g["hs_count_30d"] = c.rolling(30, min_periods=1).sum()
        g["hs_frp_sum_7d"] = f.rolling(7, min_periods=1).sum()
        g["hs_days_since_last"] = _days_since((c.fillna(0) > 0), cap=60)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling / lag / seasonal feature dari cuaca (hanya hari <= t)."""
    parts = []
    for loc, g in df.sort_values(["loc_id", "date"]).groupby("loc_id", sort=False):
        g = g.copy()
        p = g["precipitation_sum"]
        for k in (3, 7, 14, 30):
            g[f"precip_sum_{k}d"] = p.rolling(k).sum()
        g["dry_days_consec"] = _consecutive_run(p < 1.0).clip(upper=90)
        g["temp_max_mean_7d"] = g["temperature_2m_max"].rolling(7).mean()
        g["rh_min_mean_7d"] = g["relative_humidity_2m_min"].rolling(7).mean()
        g["vpd_max_mean_7d"] = g["vapour_pressure_deficit_max"].rolling(7).mean()
        g["wind_max_mean_7d"] = g["wind_speed_10m_max"].rolling(7).mean()
        g["soil_0_7_mean_7d"] = g["soil_moisture_0_to_7cm_mean"].rolling(7).mean()
        g["soil_7_28_mean_14d"] = g["soil_moisture_7_to_28cm_mean"].rolling(14).mean()
        g["soil_0_7_delta_7d"] = (g["soil_moisture_0_to_7cm_mean"]
                                  - g["soil_moisture_0_to_7cm_mean"].shift(7))
        g["water_deficit_14d"] = (g["et0_fao_evapotranspiration"].rolling(14).sum()
                                  - g["precip_sum_14d"])
        parts.append(g)
    out = pd.concat(parts, ignore_index=True)
    doy = out["date"].dt.dayofyear
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return out


def add_target(df: pd.DataFrame, gaps, last_label_date) -> pd.DataFrame:
    """Target: ada >=1 hotspot di radius R pada hari t+H (H=1)."""
    df = df.sort_values(["loc_id", "date"]).copy()
    df["target_date"] = df["date"] + pd.Timedelta(days=HORIZON_DAYS)
    df["target_count"] = df.groupby("loc_id")["hs_count"].shift(-HORIZON_DAYS)
    df["fire_next_day"] = (df["target_count"] > 0).astype(float)
    invalid = (df["target_count"].isna() | df["target_date"].isin(gaps)
               | (df["target_date"] > last_label_date))
    df.loc[invalid, "fire_next_day"] = np.nan
    return df


def assign_split(target_date: pd.Series) -> pd.Series:
    s = pd.Series("test", index=target_date.index)
    s[target_date <= TRAIN_END] = "train"
    s[(target_date >= VAL_START) & (target_date <= VAL_END)] = "val"
    return s


# fitur akhir yang dipakai model
WEATHER_ENGINEERED = [
    "precip_sum_3d", "precip_sum_7d", "precip_sum_14d", "precip_sum_30d",
    "dry_days_consec", "temp_max_mean_7d", "rh_min_mean_7d", "vpd_max_mean_7d",
    "wind_max_mean_7d", "soil_0_7_mean_7d", "soil_7_28_mean_14d",
    "soil_0_7_delta_7d", "water_deficit_14d",
]
HOTSPOT_HISTORY = ["hs_count", "hs_count_7d", "hs_count_30d",
                   "hs_frp_sum_7d", "hs_days_since_last"]
SEASONAL = ["doy_sin", "doy_cos"]
LOG1P_COLS = ["precipitation_sum", "precip_sum_3d", "precip_sum_7d",
              "precip_sum_14d", "precip_sum_30d", "hs_count", "hs_count_7d",
              "hs_count_30d", "hs_frp_sum_7d"]


def build_dataset(raw_dir=None):
    """Jalankan seluruh pipeline -> (frame lengkap berlabel, info)."""
    raw_dir = Path(raw_dir) if raw_dir else find_raw_dir()
    w = load_weather(raw_dir / "data_cuaca_semua_lokasi.csv")
    v = load_viirs(raw_dir / "viirs-jpss1_2018_2025_Indonesia.csv")
    locs = location_table(w)
    v_start, v_end = v["acq_date"].min(), v["acq_date"].max()
    gaps = gap_days(v, v_start, v_end)
    vf = filter_viirs(v)
    hs = daily_hotspots(vf, locs, v_start, v_end)
    hs = add_hotspot_history(hs, gaps)
    df = w.merge(hs, on=["loc_id", "date"], how="left")
    df = add_weather_features(df)
    df = add_target(df, gaps, v_end)
    df["split"] = assign_split(df["target_date"])
    info = dict(v_start=v_start, v_end=v_end, gaps=gaps, locs=locs, n_viirs=len(v),
                n_viirs_filtered=len(vf))
    return df, info

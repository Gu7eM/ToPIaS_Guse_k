import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

DATA = Path("data")
NUMERIC = [
    "Usage_kWh", "Lagging_Current_Reactive_Power_kVarh",
    "Leading_Current_Reactive_Power_kVarh", "CO2_tCO2_",
    "Lagging_Current_Power_Factor", "Leading_Current_Power_Factor",
    "Temperature_C", "Humidity_Pct",
]


def _fail(task, message):
    print(f"❌ Задание {task}: {message}")


def check(task, value):
    # Самопроверка без раскрытия полного решения.
    try:
        if value is Ellipsis or value is None:
            return _fail(task, "ответ не заполнен")
        if task == 1:
            got = tuple(pd.Timestamp(x) for x in value)
            expected = (pd.Timestamp("2018-01-01 00:00"), pd.Timestamp("2018-12-31 23:45"))
            ok = got == expected
        elif task == 2:
            ok = (isinstance(value, pd.DataFrame) and value.shape == (8, 8)
                  and np.isclose(value.loc["Temperature_C", "max"], 37.4)
                  and np.isclose(value.loc["Usage_kWh", "mean"], 27.386892))
        elif task == 3:
            expected = {"Весна": (26.9542, 32.8478, 8.5258),
                        "Зима": (32.0801, 37.1293, -12.1306),
                        "Лето": (24.4156, 30.7429, 16.7532),
                        "Осень": (26.1866, 32.3212, -4.2537)}
            got = value.set_index("Season")[["avg_usage", "std_usage", "avg_temp"]]
            ok = set(got.index) == set(expected) and all(
                np.allclose(got.loc[k].to_numpy(float), v, atol=1e-4) for k, v in expected.items())
        elif task == 4:
            got = value.to_dict() if isinstance(value, pd.Series) else dict(value)
            ok = got == {"Низкая": 11634, "Средняя": 11851, "Высокая": 11555}
        elif task == 5:
            a = np.asarray(value, dtype=float)
            variances = np.sort(np.var(a, axis=0))[::-1] if a.ndim == 2 else np.array([])
            ok = a.shape == (35040, 2) and np.allclose(variances, [3.40061, 2.33782], atol=2e-3)
        elif task == 6:
            labels = np.asarray(value)
            ok = labels.shape == (35040,) and len(np.unique(labels)) == 3
        elif task == 7:
            a = np.asarray(value)
            ok = a.shape == (2049,) and np.isclose(a[0], 0, atol=1e-8) and round(a.mean()) == 1200
        else:
            return _fail(task, "неизвестный номер")
    except Exception as exc:
        return _fail(task, f"не удалось проверить ({type(exc).__name__}: {exc})")
    print(f"{'✅' if ok else '❌'} Задание {task}" + (" выполнено" if ok else ": проверьте результат"))


print("Папка с данными найдена:", DATA.exists())
print("Файлы:", [p.name for p in DATA.iterdir()] if DATA.exists() else [])


df = pd.read_csv(DATA / "Сталелитейные_данные.csv")

print("Размерность:", df.shape)
print("Пропусков:", int(df.isna().sum().sum()))
print("Полных дубликатов:", int(df.duplicated().sum()))
df.head(3)

df["DateTime"] = pd.to_datetime(df["date"], format="%d/%m/%Y %H:%M")
# df = df.sort_values("DateTime").reset_index(drop=True)
print("Диапазон:", df["DateTime"].min(), "—", df["DateTime"].max())
print("Время изначально упорядочено:", df["DateTime"].is_monotonic_increasing)
print(df)

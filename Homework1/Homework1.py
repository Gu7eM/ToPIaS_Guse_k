import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA = Path("data")
plt.rcParams["figure.figsize"] = (11, 5)

EXPECTED = {
    1: (35040, 11),
    2: [3.17, 4.0, 3.24, 3.31, 3.82, 3.28, 3.6, 3.6, 3.28, 3.78],
    3: 35040,
    4: (-28.9, 37.4),
    5: ["date", "Usage_kWh", "Lagging_Current_Reactive_Power_kVarh",
        "Leading_Current_Reactive_Power_kVarh", "CO2_tCO2_",
        "Lagging_Current_Power_Factor", "Leading_Current_Power_Factor"],
    6: [-12.3, -9.8, -9.7],
}


def _brief(v):
    """Компактное представление ответа для печати."""
    if isinstance(v, (pd.Series, np.ndarray)):
        return list(np.round(np.asarray(v, dtype=object), 4)) if v.dtype == object else v.tolist()
    return v


def check(task, value):
    """Сравнивает ответ с эталоном и печатает вердикт."""
    expected = EXPECTED[task]
    try:
        if value is None or (isinstance(value, type(Ellipsis))):
            print(f"❌ Задание {task}: ответ не заполнен")
            return
        if isinstance(expected, tuple) and not isinstance(value, (list, np.ndarray, pd.Series)):
            ok = tuple(np.round(np.array(value, dtype=float), 6)) == tuple(
                np.round(np.array(expected, dtype=float), 6))
        elif isinstance(expected, list) and isinstance(expected[0], str):
            ok = list(value) == expected
        elif isinstance(expected, list):
            got = np.asarray(pd.Series(value).tolist(), dtype=float)
            ok = got.shape == np.array(expected).shape and np.allclose(got, expected)
        else:
            ok = np.isclose(float(value), float(expected))
    except Exception as e:
        print(f"❌ Задание {task}: не удалось сравнить ответ ({type(e).__name__}: {e})")
        return
    print(f"{'✅' if ok else '❌'} Задание {task}: ваш ответ {_brief(value)}")
    if not ok:
        print(f"   ожидалось: {expected}")


df_energy = pd.read_csv(DATA / "Steel_Industry.csv")
ans = df_energy.shape
check(1, ans)


ans = list(df_energy["Usage_kWh"].head(10))
check(2, ans)


temperature = np.loadtxt(DATA / "температура.txt")
ans = temperature.size
check(3, ans)


ans = (temperature.min(), temperature.max())
check(4, ans)


df_energy = df_energy.drop(columns=["NSM", "WeekStatus", "Day_of_week", "Load_Type"])
ans = list(df_energy.columns)
check(5, ans)


df_energy["Temperature_C"] = list(temperature)
ans = df_energy["Temperature_C"].head(3)
check(6, ans)


var_plot1 = "Temperature_C"
var_plot2 = "CO2_tCO2_"

df_energy["DateTime"] = pd.to_datetime(df_energy["date"], format="%d/%m/%Y %H:%M")
df_plot = df_energy.sort_values("DateTime")

fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
for ax, name in zip(axes, [var_plot1, var_plot2]):
    ax.plot(df_plot["DateTime"], df_plot[name], linewidth=0.4)
    ax.set_ylabel(name, fontsize=9)
    ax.grid(True)
axes[-1].set_xlabel("Время")
axes[0].set_title("Итоговый набор данных")
plt.tight_layout()
plt.show()
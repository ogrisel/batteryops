# %%
# https://ember-climate.org/data-catalogue/european-wholesale-electricity-price-data/
# %%
from zipfile import ZipFile
from pathlib import Path


hourly_zip_filename = "european_wholesale_electricity_price_data_hourly.zip"
all_countries_csv_filename = (
    "european_wholesale_electricity_price_data_hourly/all_countries.csv"
)


if not Path(all_countries_csv_filename).exists():
    with ZipFile(hourly_zip_filename) as zf:
        zf.extract(
            "european_wholesale_electricity_price_data_hourly/all_countries.csv", "."
        )
# %%
import pandas as pd


electricity_price = pd.read_csv(
    "european_wholesale_electricity_price_data_hourly/all_countries.csv",
    parse_dates=["Datetime (UTC)", "Datetime (Local)"],
).rename(
    columns={
        "Country": "country",
        "ISO3 Code": "country_code",
        "Datetime (UTC)": "datetime_utc",
        "Datetime (Local)": "datetime_local",
        "Price (EUR/MWhe)": "price_euro_MWhe",
    }
)
# %%
electricity_price.info()
# %%
electricity_price["country_code"].unique()
# %%
electricity_price_fra = electricity_price.query("country_code == 'FRA'")
electricity_price_fra
# %%
electricity_price_fra.plot(x="datetime_utc", y="price_euro_MWhe")
# %%
electricity_price_fra["datetime_utc"].max()

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(18, 6))

start_date = pd.to_datetime("2024-06-12")
end_date = start_date + pd.to_timedelta("7 day")
_ = electricity_price_fra.query(
    "datetime_utc > @start_date & datetime_utc <= @end_date"
).plot(x="datetime_utc", y="price_euro_MWhe", ax=ax)

# %%

fig, ax = plt.subplots(figsize=(7 * 5, 5))
electricity_price_fra.groupby(
    by=[
        electricity_price_fra["datetime_utc"].dt.day_of_week.rename("day_of_week"),
        electricity_price_fra["datetime_utc"].dt.hour.rename("hour"),
    ]
).agg(price_euro_MWhe=("price_euro_MWhe", "mean")).plot(
    y="price_euro_MWhe",
    kind="line",
    ax=ax,
)
weekday_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
ticks = []
tick_labels = []
for day_idx, day in enumerate(weekday_names):
    for hours_bin in range(24 // 3):
        ticks.append(day_idx * 24 + hours_bin * 3)
        if hours_bin == 0:
            tick_labels.append(day)
        else:
            tick_labels.append(f"{hours_bin * 3}:00")
print(tick_labels)

_ = ax.set(
    xticks=ticks,
    xticklabels=tick_labels,
    title="Average electricity price in France",
)

# %%

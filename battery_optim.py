# %% [markdown] Battery optimization
#
# Use scipy linear programming to find the optimal battery battery hourly buy
# and sell operations assuming the following battery model:
#
# - 4 MWh usable storage
# - 1 MW charge power
# - 1 MW discharge power
# - 82% round-trip efficiency
#
# We also set the following initial and boundary conditions:
#
# - we optimize for a 7 days time horizon;
# - the battery load state is a variable to optimize and is constrained to be
#   the same at the start and the end of the time period;
# - at each hour, the battery can either be charged, discharged or stay the
#   same.
# - the total number of variables are therefore 2 * (7 * 24) + 1 (for the
#   initial/final state) = 337.
# %%
# Battery parameters
BATTERY_PARAMS = dict(
    charge_efficiency=0.91,
    discharge_efficiency=0.91,
    max_storage=4,
    max_charge_power=1,
    max_discharge_power=1,
)

# Country code for which to optimize the battery operation
country_code = "DEU"

# %% [markdown]
# ## Data
#
# We first create a synthetic dataset for the electricity price: we assume a
# sinusoidal electricity price profile with 2 peaks and 2 valleys per day over
# 7 days.

# %%
import pandas as pd
import numpy as np

# hourly_electricity_price = pd.DataFrame(
#     {
#         "datetime_utc": pd.date_range(
#             start="2024-06-12",
#             end="2024-06-19",
#             freq="h",
#         )[:-1],
#     }
# )
# horizon_duration_h = hourly_electricity_price.shape[0]
# hourly_electricity_price["price_euro_MWhe"] = (
#     1 + np.sin(np.arange(horizon_duration_h) * np.pi / 6)
# ) * 30 + 40
# # %%
# hourly_electricity_price.plot(x="datetime_utc", y="price_euro_MWhe")

# %% [markdown]
# ## Load electricity price data
#
# %%
electricity_price_all = pd.read_csv(
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


def average_weekly_price(electricity_price_all, country_code):
    electricity_price_country = electricity_price_all.query(
        "country_code == @country_code"
    )
    hourly_electricity_price = electricity_price_country.groupby(
        by=[
            electricity_price_country["datetime_utc"].dt.day_of_week.rename(
                "day_of_week"
            ),
            electricity_price_country["datetime_utc"].dt.hour.rename("hour"),
        ]
    ).agg(price_euro_MWhe=("price_euro_MWhe", "mean"))
    return hourly_electricity_price


# %%
average_weekly_price(electricity_price_all, "DEU").plot(
    y="price_euro_MWhe",
    kind="line",
)

# %%
average_weekly_price(electricity_price_all, "FRA").plot(
    y="price_euro_MWhe",
    kind="line",
)

# %%
average_weekly_price(electricity_price_all, "ESP").plot(
    y="price_euro_MWhe",
    kind="line",
)

# %% [markdown]
# ## Linear programming
#
# We now set up the linear programming problem.
#
# ### Variables and costs/gains
#
# The variables are the charge power, the discharge power and the load state at
# each hour. The costs are the electricity price for charging and the gains are
# the electricity price for discharging (while taking the efficiency into
# account). The load state is the state of charge of the battery at each hour,
# it does not have a direct cost or gain but interacts with the charge and
# discharge in the constraints.
#
# ### Constraints
#
# The load state at each hour is the load state at the previous hour plus the
# charge multiplied by the charge efficiency minus the discharge.
#
# We now set up the constraints.
#
# - The current load state is equal to the previous load state plus the
#   current charge multiplied by the charging efficiency minus the current
#   discharge.
# - The final load state must be the same as the initial load state.
# - At each hour, the load state cannot be higher than the maximum storage.
# - At each hour, the load state cannot be lower than 0.
# - At each hour, the charge power cannot be higher than the maximum charge
#   power.
# - At each hour, the discharge power cannot be higher than the maximum
#   discharge power.
# %%
from scipy.optimize import linprog
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class BatterySchedule:
    battery_params: dict
    price: np.ndarray
    charge: np.ndarray
    discharge: np.ndarray
    load_state: np.ndarray
    revenue: float

    def plot(self, figsize=(18, 8), title=None):
        fig, ax = plt.subplots(nrows=4, figsize=figsize, sharex=True)
        if title is None:
            title = f"Battery operations over {self.price.shape[0]/24:.1f} days"
        fig.suptitle(title)

        ax[0].plot(self.price, label="price")
        ax[0].set(ylabel="Price (Euro/MWh)")

        ax[1].plot(self.charge)
        ax[1].set(ylabel="Charge (MW)")

        ax[2].plot(self.discharge)
        ax[2].set(ylabel="Discharge (MW)")

        ax[3].plot(self.load_state)
        ax[3].set(ylabel="Load state (MWh)")

    def print_summary(self):
        print(f"Expected weekly revenue: {self.revenue:.1f} euro")
        print(f"Maximimum load: {self.load_state.max():.1f} MWh")
        print(
            f"Effective weekly cycles: {self.discharge.sum() / self.battery_params["max_storage"]:.1f}"
        )
        print(
            f"Battery duration: {self.battery_params['max_storage'] / self.battery_params['max_discharge_power']:.1f} hours"
        )


def optimize_schedule(
    hourly_electricity_price, initial_load_state=None, **custom_battery_params
):
    bp = BATTERY_PARAMS.copy()
    bp.update(custom_battery_params)
    n_time_steps = hourly_electricity_price.shape[0]
    n_variables = 3 * n_time_steps + 1

    charge_range = slice(0, n_time_steps)
    discharge_range = slice(n_time_steps, 2 * n_time_steps)
    load_state_range = slice(2 * n_time_steps, 3 * n_time_steps + 1)

    charge_costs = hourly_electricity_price
    discharge_gains = hourly_electricity_price * bp["discharge_efficiency"]
    c = np.concatenate([charge_costs, -discharge_gains, [1e-12] * (n_time_steps + 1)])
    assert c.shape == (n_variables,)

    # A_eq @ x = b_eq
    extra_equations = 2 if initial_load_state is not None else 1
    A_eq = np.zeros((n_time_steps + extra_equations, n_variables))
    b_eq = np.zeros(n_time_steps + extra_equations)

    # At each time step, the load state is the load state at the previous time step
    # plus the charge minus the discharge (taking into account the charging
    # efficiency). Discharging efficiency is already taken into account in the
    # computation of the gains.
    for t in range(n_time_steps):
        A_eq[t, charge_range][t] = -bp["charge_efficiency"]
        A_eq[t, discharge_range][t] = 1
        A_eq[t, load_state_range][t] = -1
        A_eq[t, load_state_range][t + 1] = 1

    # Initial and final load state should match
    A_eq[-1, load_state_range.start] = -1
    A_eq[-1, load_state_range.stop - 1] = 1
    if initial_load_state is not None:
        A_eq[-2, load_state_range.start] = 1
        b_eq[-2] = initial_load_state

    # A_ub @ x <= b_ub
    A_ub = np.zeros((6 * n_time_steps, n_variables))
    b_ub = np.zeros(6 * n_time_steps)

    for t in range(n_time_steps):
        # Load state should be between 0 and max_storage
        A_ub[6 * t, load_state_range][t] = 1
        b_ub[6 * t] = bp["max_storage"]

        A_ub[6 * t + 1, load_state_range][t] = -1
        b_ub[6 * t + 1] = 0

        # Charge power should be between 0 and max_charge_power
        A_ub[6 * t + 2, charge_range][t] = 1
        b_ub[6 * t + 2] = bp["max_charge_power"]

        A_ub[6 * t + 3, charge_range][t] = -1
        b_ub[6 * t + 3] = 0

        # Discharge power should be between 0 and max_discharge_power
        A_ub[6 * t + 4, discharge_range][t] = 1
        b_ub[6 * t + 4] = bp["max_discharge_power"]

        A_ub[6 * t + 5, discharge_range][t] = -1
        b_ub[6 * t + 5] = 0

    res = linprog(
        c,
        A_eq=A_eq,
        b_eq=b_eq,
        A_ub=A_ub,
        b_ub=b_ub,
        method="highs",
    )
    return BatterySchedule(
        battery_params=bp,
        price=hourly_electricity_price,
        charge=res.x[charge_range],
        discharge=res.x[discharge_range],
        load_state=res.x[load_state_range][:-1],
        revenue=-res.fun,
    )


# %%
res = optimize_schedule(
    average_weekly_price(electricity_price_all, "ESP")["price_euro_MWhe"].to_numpy(),
)
res.plot()
# %%
res.print_summary()

# %%
res = optimize_schedule(
    average_weekly_price(electricity_price_all, "FRA")["price_euro_MWhe"].to_numpy(),
)
res.plot()

# %%
res.print_summary()
# %%
res = optimize_schedule(
    average_weekly_price(electricity_price_all, "DEU")["price_euro_MWhe"].to_numpy(),
)
res.plot()

# %%
res.print_summary()

# %%
res = optimize_schedule(
    average_weekly_price(electricity_price_all, "DEU")["price_euro_MWhe"].to_numpy(),
    max_storage=12,
)
res.plot()

# %%
res.print_summary()
# %%
res = optimize_schedule(
    average_weekly_price(electricity_price_all, "DEU")["price_euro_MWhe"].to_numpy(),
    max_storage=12,
    charge_efficiency=0.95,
    discharge_efficiency=0.95,
)
res.plot()
# %%
res.print_summary()

# %%
import random

electricity_price_deu = electricity_price_all.query("country_code == 'DEU'")
deu_datetimes = electricity_price_deu["datetime_utc"]


mondays = electricity_price_deu.query(
    "datetime_utc.dt.weekday == 0 & datetime_utc.dt.hour == 0"
)["datetime_utc"][:-1].to_list()
start_datetime = random.choices(mondays)[0]
end_datetime = start_datetime + pd.to_timedelta("7 day")
electricity_price_deu_random_week = electricity_price_deu.query(
    "datetime_utc >= @start_datetime & datetime_utc < @end_datetime"
)
electricity_price_deu_random_week.plot(x="datetime_utc", y="price_euro_MWhe")

res = optimize_schedule(
    electricity_price_deu_random_week["price_euro_MWhe"].to_numpy(),
    max_storage=12,
)
res.print_summary()
# %%
res.plot()


# %%
electricity_price_deu = electricity_price_all.query(
    "country_code == 'DEU' & datetime_utc.dt.year >= 2023"
)


# %%
def optimize_all_weeks(electricity_price, **kwargs):
    all_results = []
    mondays = electricity_price.query(
        "datetime_utc.dt.weekday == 0 & datetime_utc.dt.hour == 0"
    )["datetime_utc"][:-1].to_list()
    for monday in mondays:
        start_datetime = monday
        end_datetime = start_datetime + pd.to_timedelta("7 day")
        electricty_price_for_week = electricity_price.query(
            "datetime_utc >= @start_datetime & datetime_utc < @end_datetime"
        )
        res = optimize_schedule(
            electricty_price_for_week["price_euro_MWhe"].to_numpy(),
            **kwargs,
        )
        all_results.append(res)
    return all_results


max_storage = 4
all_results = optimize_all_weeks(
    electricity_price_deu,
    max_storage=max_storage,
)
len(all_results)
# %%
revenues = pd.Series([res.revenue for res in all_results])
# %%
revenues.plot(kind="hist", bins=30)

# %%
# all_results[-1].plot()

# %%
init_load_states = pd.Series([res.load_state[0] for res in all_results])
# %%
# Re-run with the average initial load state as extra constraint:
all_results = optimize_all_weeks(
    electricity_price_deu,
    max_storage=max_storage,
    initial_load_state=init_load_states.mean(),
)
# %%
# 10 years break even Euro/kWh CAPEX
duration = round(max_storage / BATTERY_PARAMS["max_discharge_power"], 1)
print(
    f"Maximum CAPEX cost for a {duration} h battery to "
    f"break-even at 10 years: "
    f"{revenues.mean() * 52 * 10 / (max_storage * 1000):.1f} euro/kWh"
)

# %%

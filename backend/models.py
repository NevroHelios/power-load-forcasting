from typing import Literal, List
from pydantic import BaseModel
import numpy as np
from datetime import datetime
import holidays



class SetModel(BaseModel):
    name: Literal['lgbm', 'xgboost']


# Index(['Usage_kWh', 'Lagging_Current_Reactive.Power_kVarh',
#        'Leading_Current_Reactive_Power_kVarh', 'CO2(tCO2)',
#        'Lagging_Current_Power_Factor', 'Leading_Current_Power_Factor', 'train',
#        'weekday', 'dayofmonth', 'month_sin', 'month_cos', 'hour',
#        'hour_sin t/2', 'hour_cos t/2', 'hour_sin', 'hour_cos', 'hour_sin 2t',
#        'hour_cos 2t', 'hour_sin 3t', 'hour_cos 3t', 'hour_sin 4t',
#        'hour_cos 4t', 'dayofweek_sin', 'dayofweek_cos', 'total_reactive_power',
#        'power_factor_diff', 'reactive_usage_ratio', 'isHoliday', 'season',
#        'usage_hour_sin', 'usage_hour_cos'],
#       dtype='object')
class LoadFeatures(BaseModel):
    Date_Time: str  # "01-01-2018 00:15"
    Usage_kWh: float
    Lagging_Current_Reactive_Power_kVarh: float
    Leading_Current_Reactive_Power_kVarh: float
    CO2_tCO2: float
    Lagging_Current_Power_Factor: float
    Leading_Current_Power_Factor: float
    NSM: float  # dropped later but keep for schema compatibility

    @property
    def get_features(self) -> List[float]:
        # --- Parse datetime ---
        dt = datetime.strptime(self.Date_Time, "%m-%d-%Y %H:%M")
        month = dt.month
        dayofmonth = dt.day
        weekday = dt.weekday()
        hour = dt.hour

        # --- Cyclical encodings ---
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)

        hour_sin_t_2 = np.sin(2 * np.pi * hour / 24)
        hour_cos_t_2 = np.cos(2 * np.pi * hour / 24)

        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)

        hour_sin_2t = np.sin(4 * np.pi * hour / 24)
        hour_cos_2t = np.cos(4 * np.pi * hour / 24)

        hour_sin_3t = np.sin(6 * np.pi * hour / 24)
        hour_cos_3t = np.cos(6 * np.pi * hour / 24)

        hour_sin_4t = np.sin(8 * np.pi * hour / 24)
        hour_cos_4t = np.cos(8 * np.pi * hour / 24)

        dayofweek_sin = np.sin(2 * np.pi * weekday / 7)
        dayofweek_cos = np.cos(2 * np.pi * weekday / 7)

        # --- derived features ---
        total_reactive_power = (
            self.Lagging_Current_Reactive_Power_kVarh +
            self.Leading_Current_Reactive_Power_kVarh
        )
        power_factor_diff = (
            self.Lagging_Current_Power_Factor -
            self.Leading_Current_Power_Factor
        )
        reactive_usage_ratio = total_reactive_power / (self.Usage_kWh + 1e-5)

        # --- holidays ---
        isHoliday = 0
        ind_holidays = holidays.CountryHoliday("IND", years=[2018])
        if dt.date() in ind_holidays:
            isHoliday = 1

        # --- season mapping ---
        season_map = {
            12: 1, 1: 1, 2: 1,  # Winter
            3: 2, 4: 2, 5: 2,  # Spring
            6: 3, 7: 3, 8: 3,  # Summer
            9: 4, 10: 4, 11: 4 # Fall
        }
        season = season_map[month]

        usage_hour_sin = self.Usage_kWh * hour_sin
        usage_hour_cos = self.Usage_kWh * hour_cos

        # odered
        return [
            self.Usage_kWh,
            self.Lagging_Current_Reactive_Power_kVarh,
            self.Leading_Current_Reactive_Power_kVarh,
            self.CO2_tCO2,
            self.Lagging_Current_Power_Factor,
            self.Leading_Current_Power_Factor,
            weekday,
            dayofmonth,
            month_sin,
            month_cos,
            hour,
            hour_sin_t_2,
            hour_cos_t_2,
            hour_sin,
            hour_cos,
            hour_sin_2t,
            hour_cos_2t,
            hour_sin_3t,
            hour_cos_3t,
            hour_sin_4t,
            hour_cos_4t,
            dayofweek_sin,
            dayofweek_cos,
            total_reactive_power,
            power_factor_diff,
            reactive_usage_ratio,
            isHoliday,
            season,
            usage_hour_sin,
            usage_hour_cos
        ]
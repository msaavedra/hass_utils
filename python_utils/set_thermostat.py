import json
from dataclasses import dataclass
from datetime import time, datetime, timedelta
from enum import Enum, IntEnum
from functools import cached_property
from time import mktime
from typing import Protocol, Any, Optional
from zoneinfo import ZoneInfo
import os
import sys
import traceback
import logging
import logging.handlers

syslog_handler = logging.handlers.SysLogHandler("/dev/log")
syslog_handler.level = logging.INFO
logging.basicConfig(
    level=logging.DEBUG,
    handlers=[
        logging.StreamHandler(),
        syslog_handler,
    ]
)
logger = logging.getLogger(__file__)


class ThermostatMode(Enum):
    HEATING = "heating"
    COOLING = "cooling"


class TemperatureLevel(IntEnum):
    HIGHEST = 4
    HIGH = 3
    MEDIUM = 2
    LOW = 1
    LOWEST = 0


@dataclass
class Config:
    hass_base_url: str
    hvac_entity: str
    timezone: str
    cool_weather_max: int = 69
    cool_weather_min: int = 62
    warm_weather_max: int = 78
    warm_weather_min: int = 71
    peak_usage_start_time: str = "16:00"
    peak_usage_end_time: str = "21:00"
    default_bed_time: str = "23:00"
    default_wake_time: str = "8:00"


@dataclass
class Forecast:
    high: int
    low: int


@dataclass
class ThermostatSettings:
    mode: ThermostatMode
    temperature: int


class TemperatureRange:

    def __init__(self, low: int, high: int):
        self.low = low
        self.high = high

    def choose_temperature_by_level(self, temperature_level: TemperatureLevel) -> int:
        value = round(
            self.low + (float(temperature_level) * ((self.high - self.low) / len(TemperatureLevel)))
        )
        logger.debug(f"{self.low} -> {value} -> {self.high}")
        return value


class TimeRange:

    def __init__(self, start: datetime, end: datetime):
        self.start = start
        self.end = end

    def __contains__(self, item: datetime):
        return self.start <= item <= self.end


class DataTransformer:
    def __init__(self, config: Config, sensor_data: list[dict[str, Any]]):
        self.config = config
        self.local_tzinfo = ZoneInfo(config.timezone)
        self.sensor_data = sensor_data

    @cached_property
    def current_timestamp(self) -> datetime:
        if "TIME_OVERRIDE" in os.environ:
            if int(os.environ["TIME_OVERRIDE"].split(":")[0]) < 9:
                now = datetime.now(self.local_tzinfo) + timedelta(hours=24)
            else:
                now = datetime.now(self.local_tzinfo)
            current_timestamp = datetime.combine(
                now.date(),
                time.fromisoformat(os.environ["TIME_OVERRIDE"]),
                tzinfo=self.local_tzinfo,
            )
        else:
            current_timestamp = datetime.now(self.local_tzinfo)
        logger.debug(f"Current timestamp: {current_timestamp.isoformat()}")
        return current_timestamp

    @cached_property
    def thermostat_attibutes(self) -> dict[str, Any]:
        return self.sensor_map[self.config.hvac_entity]["attributes"]

    @cached_property
    def hard_max_temp(self) -> int:
        return int(self.thermostat_attibutes["max_temp"])

    @cached_property
    def hard_min_temp(self) -> int:
        return int(self.thermostat_attibutes["max_temp"])

    @cached_property
    def last_midnight(self) -> datetime:
        return datetime.combine(self.current_timestamp.date(), time.min, tzinfo=self.local_tzinfo)

    @cached_property
    def next_midnight(self) -> datetime:
        return self.last_midnight + timedelta(hours=24)

    @cached_property
    def sensor_map(self):
        return {s["entity_id"]: s for s in self.sensor_data}

    @cached_property
    def current_indoor_temperature(self) -> int:
        return int(self.thermostat_attibutes["current_temperature"])

    @cached_property
    def daylight_length(self) -> timedelta:
        sun_data = self.sensor_map["sun.sun"]
        next_rising = datetime.fromisoformat(sun_data["attributes"]["next_rising"]).astimezone(self.local_tzinfo)
        next_setting = datetime.fromisoformat(sun_data["attributes"]["next_setting"]).astimezone(self.local_tzinfo)
        if next_rising < next_setting:
            # the sun will rise before it sets again, meaning it's night. Finding daylight is simply the delta
            # of the two values.
            return next_setting - next_rising
        else:
            # the sun will set before it rises again, so if we get the delta of the values, we'll be measuring
            # nighttime hours instead. We can correct this by subtracting the nighttime value from a full 24 hours.
            return timedelta(hours=24) - (next_rising - next_setting)

    @cached_property
    def forecast(self) -> Forecast:
        for entity_id in self.sensor_map:
            if entity_id.startswith("weather") and entity_id.endswith("daynight"):
                weather_forecast = self.sensor_map[entity_id]["attributes"]["forecast"]
                break
        else:
            raise Exception("Weather forecast not found!")

        closest_forecasts: list[int] = sorted([wf["temperature"] for wf in weather_forecast[:2]], reverse=True)
        return Forecast(high=closest_forecasts[0], low=closest_forecasts[1])

    @cached_property
    def number_of_people_at_home(self) -> int:
        if "PEOPLE_OVERRIDE" in os.environ:
            return int(os.environ["PEOPLE_OVERRIDE"])

        return int(self.sensor_map["sensor.number_of_people_at_home"]["state"])

    @cached_property
    def alarms(self) -> Optional[list[datetime]]:
        alarms = []
        for entity_id, entity in self.sensor_map.items():
            if not entity_id.endswith("next_alarm"):
                continue
            alarm = datetime.fromisoformat(entity["state"]).astimezone(self.local_tzinfo)
            if alarm > self.current_timestamp and (alarm - self.current_timestamp) > timedelta(hours=24):
                continue
            alarms.append(alarm)

        alarms.sort()
        logger.debug(f"Alarm times: {[a.isoformat() for a in alarms]}")
        return alarms

    @cached_property
    def peak_usage_range(self) -> TimeRange:
        peak_start = self._time_str_to_datetime(self.config.peak_usage_start_time)
        peak_end = self._time_str_to_datetime(self.config.peak_usage_end_time)
        return TimeRange(start=peak_start, end=peak_end)

    @cached_property
    def wake_times(self) -> list[datetime]:
        if not self.alarms:
            return [self._time_str_to_datetime(self.config.default_wake_time)]

        return self.alarms

    @cached_property
    def bed_times(self) -> list[datetime]:
        if not self.alarms:
            return [self._time_str_to_datetime(self.config.default_bed_time)]
        bed_times = [a - timedelta(hours=8) for a in self.alarms]
        bed_times.sort()
        logger.debug(f"Bed times: {[bt.isoformat() for bt in bed_times]}")
        return bed_times

    @cached_property
    def mode(self) -> ThermostatMode:
        if "MODE_OVERRIDE" in os.environ:
            return ThermostatMode(os.environ["MODE_OVERRIDE"])

        if self.current_indoor_temperature >= self.hard_max_temp:
            return ThermostatMode.COOLING

        if self.current_indoor_temperature <= self.hard_min_temp:
            return ThermostatMode.HEATING

        if self.forecast.high > self.config.cool_weather_max:
            return ThermostatMode.COOLING

        if self.forecast.low < self.config.warm_weather_min:
            return ThermostatMode.HEATING

        if self.daylight_length < timedelta(hours=10, minutes=30):
            return ThermostatMode.HEATING

        return ThermostatMode.COOLING

    @cached_property
    def thermostat_range(self):
        if self.mode == ThermostatMode.COOLING:
            return TemperatureRange(self.config.warm_weather_min, self.config.warm_weather_max)
        else:
            return TemperatureRange(self.config.cool_weather_min, self.config.cool_weather_max)

    def _time_str_to_datetime(self, time_str):
        hour, _, minute = time_str.partition(":")
        if len(hour) != 2:
            hour = hour[:2].rjust(2, "0")
        if len(minute) != 2:
            minute = minute[:2].rjust(2, "0")

        time_str = f"{hour}:{minute}"

        return datetime.combine(
            self.last_midnight,
            time.fromisoformat(time_str),
            tzinfo=self.local_tzinfo,
        )


class TemperatureStrategy(Protocol):
    cooling_level: TemperatureLevel
    heating_level: TemperatureLevel
    mode: ThermostatMode
    thermostat_range: TemperatureRange

    def get_temperature(self) -> int:
        ...

    def explain_strategy(self) -> str:
        ...

    @classmethod
    def meets_criteria(cls, transformer: DataTransformer) -> bool:
        ...


class TemperatureStrategyBaseImplementation(TemperatureStrategy):

    def __init__(self, mode: ThermostatMode, thermostat_range: TemperatureRange):
        self.mode = mode
        self.thermostat_range = thermostat_range

    def get_temperature(self) -> int:
        if self.mode == ThermostatMode.COOLING:
            return self.thermostat_range.choose_temperature_by_level(self.cooling_level)

        return self.thermostat_range.choose_temperature_by_level(self.heating_level)

    def explain_strategy(self) -> str:
        return self.__doc__


class NobodyHomeStrategyBaseImplementation(TemperatureStrategyBaseImplementation):
    """When nobody is home, we want to use the least amount of energy."""
    cooling_level = TemperatureLevel.HIGHEST
    heating_level = TemperatureLevel.LOWEST

    @classmethod
    def meets_criteria(cls, transformer: DataTransformer) -> bool:
        return transformer.number_of_people_at_home < 1


class StandardStrategyBaseImplementation(TemperatureStrategyBaseImplementation):
    """Typical strategy that can be used when none of the others are suitable."""
    cooling_level = TemperatureLevel.MEDIUM
    heating_level = TemperatureLevel.MEDIUM

    @classmethod
    def meets_criteria(cls, transformer: DataTransformer) -> bool:
        return True


class ReadyForBedStrategyBaseImplementation(TemperatureStrategyBaseImplementation):
    """When settling into bed, most people like the temperature to be cool."""
    cooling_level = TemperatureLevel.LOWEST
    heating_level = TemperatureLevel.LOW

    @classmethod
    def meets_criteria(cls, transformer: DataTransformer) -> bool:
        ready_for_bed_start = transformer.bed_times[0] - timedelta(minutes=15)
        ready_for_bed_end = transformer.bed_times[-1]
        logger.debug(
            f"Ready for bed: {ready_for_bed_start} - {ready_for_bed_end}."
            f" Current: {transformer.current_timestamp}"
        )
        if transformer.current_timestamp in TimeRange(ready_for_bed_start, ready_for_bed_end):
            return True

        return False


class WakingUpStrategyBaseImplementation(TemperatureStrategyBaseImplementation):
    """It's nice to have the house warmer when getting out of bed, especially on cold days."""
    cooling_level = TemperatureLevel.MEDIUM
    heating_level = TemperatureLevel.HIGH

    @classmethod
    def meets_criteria(cls, transformer: DataTransformer) -> bool:
        for alarm in transformer.alarms:
            waking_up_start = alarm - timedelta(minutes=20)
            waking_up_end = alarm + timedelta(minutes=30)
            logger.debug(
                f"Waking up times: {waking_up_start} - {waking_up_end}."
                f" Current: {transformer.current_timestamp}"
            )
            if transformer.current_timestamp in TimeRange(waking_up_start, waking_up_end):
                return True

        return False


class SleepingStrategyBaseImplementation(TemperatureStrategyBaseImplementation):
    """While sleeping, most people like to balance between coolness and saving energy."""
    cooling_level = TemperatureLevel.MEDIUM
    heating_level = TemperatureLevel.LOWEST

    @classmethod
    def meets_criteria(cls, transformer: DataTransformer) -> bool:
        sleep_start = min(transformer.bed_times)
        sleep_end = max(transformer.wake_times)
        return transformer.current_timestamp in TimeRange(sleep_start, sleep_end)


class NearlyPeakStrategyBaseImplementation(TemperatureStrategyBaseImplementation):
    """When it's almost peak usage time, we should use a bit more energy while it's still cheap."""
    cooling_level = TemperatureLevel.LOW
    heating_level = TemperatureLevel.MEDIUM  # set to high if electric heat

    @classmethod
    def meets_criteria(cls, transformer: DataTransformer) -> bool:
        nearly_peak_start = transformer.peak_usage_range.start - timedelta(minutes=30)
        nearly_peak_end = transformer.peak_usage_range.start - timedelta(seconds=1)
        return transformer.current_timestamp in TimeRange(nearly_peak_start, nearly_peak_end)


class PeakUsageStrategyBaseImplementation(TemperatureStrategyBaseImplementation):
    """Prices are high during peak usage times, so we want to reduce energy usage a bit."""
    # This only reduces power use for cooling, as most heaters are natural-gas-powered, and there
    # usually aren't time of use plans for gas. There are electric heaters, though, so this can
    # be subclassed to handle those if necessary.
    cooling_level = TemperatureLevel.HIGHEST
    heating_level = TemperatureLevel.MEDIUM

    @classmethod
    def meets_criteria(cls, transformer: DataTransformer) -> bool:
        return transformer.current_timestamp in transformer.peak_usage_range


class ThermostatRules:
    strategies_by_priority = (
        NobodyHomeStrategyBaseImplementation,
        ReadyForBedStrategyBaseImplementation,
        WakingUpStrategyBaseImplementation,
        PeakUsageStrategyBaseImplementation,
        NearlyPeakStrategyBaseImplementation,
        SleepingStrategyBaseImplementation,
    )

    def __init__(self, config: Config, sensor_data: list[dict[str, Any]]):
        self.transformer = DataTransformer(config, sensor_data)

    def get_strategy(self) -> TemperatureStrategy:
        for strategy_class in self.strategies_by_priority:
            if strategy_class.meets_criteria(self.transformer):
                break
        else:
            strategy_class = StandardStrategyBaseImplementation

        return strategy_class(self.transformer.mode, self.transformer.thermostat_range)


def mean_timestamp(timestamps: list[datetime], tz):
    total = len(timestamps)
    if total == 1:
        return timestamps[0]

    seconds = sum(mktime(t.timetuple()) for t in timestamps)
    mean_seconds = seconds / total
    return datetime.fromtimestamp(mean_seconds, tz=tz)


def set_environment():
    dirname = os.path.dirname(__file__)
    path = os.path.join(dirname,".env.json")
    logger.info(f"Loading environment from {path}")
    try:
        with open(path, "r") as f:
            env_data = json.load(f)

        for key, value in env_data.items():
            if not all([isinstance(key, str), isinstance(value, (str, int, float))]):
                continue
            logger.debug(f"{key}={value}")
            os.environ[key] = value

    except Exception as e:
        logger.warning("Environment loading failed. Skipping.")


def get_config():
    return Config(
        hass_base_url="http://citrine.home:8123",
        hvac_entity="climate.t6_pro_z_wave_programmable_thermostat",
        timezone="America/Los_Angeles",
    )


def get_sensor_data(hass_base_url):
    from urllib import request
    import json
    url = f"{hass_base_url}/api/states"
    token = os.environ["HASS_TOKEN"]
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    response = request.urlopen(request.Request(url, headers=headers, method="GET"))
    assert response.status < 300
    return json.loads(response.read())


def set_thermostat_values(config: Config, strategy: TemperatureStrategy):
    from urllib import request
    import json
    url = f"{config.hass_base_url}/api/services/climate/set_temperature"
    token = os.environ["HASS_TOKEN"]
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    post_data = {
        "entity_id": config.hvac_entity,
        "hvac_mode": "heat" if strategy.mode == ThermostatMode.HEATING else "cool",
        "temperature": float(strategy.get_temperature()),
    }
    logger.debug(f"POST DATA: {post_data}")
    response = request.urlopen(
        request.Request(url, headers=headers, data=json.dumps(post_data).encode("utf-8"), method="POST")
    )
    assert response.status < 300
    logger.debug(f"Thermostat set response: {response.read()}")


def main():
    try:
        set_environment()
        config = get_config()
        sensor_data = get_sensor_data(config.hass_base_url)
        rules = ThermostatRules(config, sensor_data)
        strategy = rules.get_strategy()
        logger.info(
            f"{strategy.mode}: {strategy.get_temperature()} - {strategy.__class__.__name__}:"
            f" {strategy.explain_strategy()}"
        )
        set_thermostat_values(config, strategy)
    except Exception as e:
        traceback.print_exception(e)
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

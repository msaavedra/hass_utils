import json
from dataclasses import dataclass
from datetime import time, datetime, timedelta
from enum import Enum, IntEnum
from functools import cached_property
from time import mktime
from typing import Protocol, Any, Optional, runtime_checkable
from zoneinfo import ZoneInfo
import os
import sys
import traceback
import logging
import logging.handlers


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
    cool_weather_max: int = 68
    cool_weather_min: int = 61
    warm_weather_max: int = 78
    warm_weather_min: int = 71
    peak_usage_start_time: str = "16:00"
    peak_usage_end_time: str = "21:00"
    expected_sleep_interval: str = "8:00"
    default_wake_time: str = "8:00"
    syslog_path: str = "/dev/log"
    syslog_level: str = "INFO"


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
        return int(self.thermostat_attibutes["min_temp"])

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
    def current_thermostat_mode(self) -> ThermostatMode:
        if self.sensor_map[self.config.hvac_entity]["state"] == "heat":
            return ThermostatMode.HEATING
        else:
            return ThermostatMode.COOLING

    @cached_property
    def thermostat_set_temperature(self) -> int:
        temp = int(self.thermostat_attibutes["temperature"])
        logger.debug(f"Set temperature: {temp}")
        return temp

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
            return [self._time_str_to_datetime(self.config.default_wake_time, future_only=True)]

        return self.alarms

    @cached_property
    def bed_times(self) -> list[datetime]:
        bed_times = [a - self._time_str_to_timedelta(self.config.expected_sleep_interval) for a in self.wake_times]
        bed_times.sort()
        logger.debug(f"Bed times: {[bt.isoformat() for bt in bed_times]}")
        return bed_times

    @cached_property
    def mode(self) -> ThermostatMode:
        if "MODE_OVERRIDE" in os.environ:
            logger.debug(f"Using override mode {os.environ['MODE_OVERRIDE']}.")
            return ThermostatMode(os.environ["MODE_OVERRIDE"])

        if self.current_indoor_temperature >= self.hard_max_temp:
            logger.debug(f"Using mode {ThermostatMode.COOLING} because we're over hard max temp.")
            return ThermostatMode.COOLING

        if self.current_indoor_temperature <= self.hard_min_temp:
            logger.debug(f"Using mode {ThermostatMode.HEATING} because we're under hard min temp.")
            return ThermostatMode.HEATING

        if self.daylight_length < timedelta(hours=10, minutes=30):
            logger.debug(f"Using mode {ThermostatMode.HEATING} because daylight length is {self.daylight_length}.")
            return ThermostatMode.HEATING

        if self.forecast.high > self.config.cool_weather_max:
            logger.debug(f"Using mode {ThermostatMode.COOLING} because the expected high is {self.forecast.high}.")
            return ThermostatMode.COOLING

        if self.forecast.low < self.config.warm_weather_min:
            logger.debug(f"Using mode {ThermostatMode.HEATING} because the expected low is {self.forecast.low}.")
            return ThermostatMode.HEATING

        logger.debug(f"Using mode {ThermostatMode.COOLING} because no other checks matched.")
        return ThermostatMode.COOLING

    @cached_property
    def thermostat_range(self):
        if self.mode == ThermostatMode.COOLING:
            return TemperatureRange(self.config.warm_weather_min, self.config.warm_weather_max)
        else:
            return TemperatureRange(self.config.cool_weather_min, self.config.cool_weather_max)

    def _time_str_to_datetime(self, time_str: str, future_only: bool = False) -> datetime:
        hour, _, minute = time_str.partition(":")
        if len(hour) != 2:
            hour = hour[:2].rjust(2, "0")
        if len(minute) != 2:
            minute = minute[:2].rjust(2, "0")

        time_str = f"{hour}:{minute}"

        timestamp = datetime.combine(
            self.last_midnight,
            time.fromisoformat(time_str),
            tzinfo=self.local_tzinfo,
        )
        if future_only and timestamp < self.current_timestamp:
            timestamp += timedelta(hours=24)

        return timestamp

    def _time_str_to_timedelta(self, time_str: str) -> timedelta:
        hour, _, minute = time_str.partition(":")
        if len(hour) != 2:
            hour = hour[:2]
        if len(minute) != 2:
            minute = minute[:2]

        return timedelta(hours=int(hour), minutes=int(minute))

@runtime_checkable
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

    def __str__(self) -> str:
        ...

    def matches_current_data(self, transformer: DataTransformer) -> bool:
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

    def __str__(self) -> str:
        return f"{self.__class__.__name__} - {self.mode.name}: {self.get_temperature()}"

    def matches_current_data(self, data_transformer: DataTransformer) -> bool:
        logger.debug(
            f"Transformer({data_transformer.current_thermostat_mode},"
            f"{data_transformer.thermostat_set_temperature})")
        logger.debug(str(self))
        matches = (
            data_transformer.thermostat_set_temperature == self.get_temperature()
            and data_transformer.current_thermostat_mode == self.mode
        )
        logger.debug(f"MATCHES: {matches}")
        return matches


class NobodyHomeStrategy(TemperatureStrategyBaseImplementation):
    """When nobody is home, we want to use the least amount of energy."""
    cooling_level = TemperatureLevel.HIGHEST
    heating_level = TemperatureLevel.LOWEST

    @classmethod
    def meets_criteria(cls, transformer: DataTransformer) -> bool:
        return transformer.number_of_people_at_home < 1


class StandardStrategy(TemperatureStrategyBaseImplementation):
    """Typical strategy that can be used when none of the others are suitable."""
    cooling_level = TemperatureLevel.MEDIUM
    heating_level = TemperatureLevel.MEDIUM

    @classmethod
    def meets_criteria(cls, transformer: DataTransformer) -> bool:
        return True


class ReadyForBedStrategy(TemperatureStrategyBaseImplementation):
    """When settling into bed, most people like the temperature to be cool."""
    cooling_level = TemperatureLevel.LOWEST
    heating_level = TemperatureLevel.LOW

    @classmethod
    def meets_criteria(cls, transformer: DataTransformer) -> bool:
        approaching_bed_times = [
            bt for bt in transformer.bed_times if (bt - transformer.current_timestamp) < timedelta(hours=8)
        ]
        if not approaching_bed_times:
            return False

        ready_for_bed_start = approaching_bed_times[0] - timedelta(minutes=15)
        ready_for_bed_end = min([transformer.bed_times[-1], (ready_for_bed_start + timedelta(hours=3))])
        logger.debug(
            f"Ready for bed: {ready_for_bed_start} - {ready_for_bed_end}."
            f" Current: {transformer.current_timestamp}"
        )
        if transformer.current_timestamp in TimeRange(ready_for_bed_start, ready_for_bed_end):
            return True

        return False


class WakingUpStrategy(TemperatureStrategyBaseImplementation):
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


class SleepingStrategy(TemperatureStrategyBaseImplementation):
    """While sleeping, most people like to balance between coolness and saving energy."""
    cooling_level = TemperatureLevel.MEDIUM
    heating_level = TemperatureLevel.LOWEST

    @classmethod
    def meets_criteria(cls, transformer: DataTransformer) -> bool:
        sleep_start = min(transformer.bed_times)
        sleep_end = max(transformer.wake_times)
        if sleep_start > sleep_end:
            sleep_start -= timedelta(hours=24)
        logger.debug(
            f"Sleep times: {sleep_start} - {sleep_end}."
            f" Current: {transformer.current_timestamp}"
        )
        return transformer.current_timestamp in TimeRange(sleep_start, sleep_end)


class NearlyPeakStrategy(TemperatureStrategyBaseImplementation):
    """When it's almost peak usage time, we should use a bit more energy while it's still cheap."""
    cooling_level = TemperatureLevel.LOW
    heating_level = TemperatureLevel.MEDIUM  # set to high if electric heat

    @classmethod
    def meets_criteria(cls, transformer: DataTransformer) -> bool:
        nearly_peak_start = transformer.peak_usage_range.start - timedelta(minutes=30)
        nearly_peak_end = transformer.peak_usage_range.start - timedelta(seconds=1)
        return transformer.current_timestamp in TimeRange(nearly_peak_start, nearly_peak_end)


class PeakUsageStrategy(TemperatureStrategyBaseImplementation):
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
        NobodyHomeStrategy,
        WakingUpStrategy,
        ReadyForBedStrategy,
        PeakUsageStrategy,
        NearlyPeakStrategy,
        SleepingStrategy,
    )

    def __init__(self, config: Config, sensor_data: list[dict[str, Any]]):
        self.transformer = DataTransformer(config, sensor_data)

    def get_strategy(self) -> TemperatureStrategy:
        for strategy_class in self.strategies_by_priority:
            logger.debug(f"Checking strategy {strategy_class.__name__}")
            if strategy_class.meets_criteria(self.transformer):
                break
        else:
            strategy_class = StandardStrategy

        return strategy_class(self.transformer.mode, self.transformer.thermostat_range)


def mean_timestamp(timestamps: list[datetime], tz):
    total = len(timestamps)
    if total == 1:
        return timestamps[0]

    seconds = sum(mktime(t.timetuple()) for t in timestamps)
    mean_seconds = seconds / total
    return datetime.fromtimestamp(mean_seconds, tz=tz)


class ConfigFileError(Exception):
    pass


def load_json(file_name):
    dir_name = os.path.dirname(__file__)
    path = os.path.join(dir_name, file_name)
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as exc:
        raise ConfigFileError(f"Could not load json data from '{path}'.") from exc


def set_environment():
    env_data = load_json(".env.json")
    if "HASS_TOKEN" not in env_data:
        raise ConfigFileError('The .env.json file must contain a "HASS_TOKEN" object')
    for key, value in env_data.items():
        if not all([isinstance(key, str), isinstance(value, (str, int, float))]):
            continue
        os.environ[key] = value


def get_config():
    config_data = load_json("config.json")
    return Config(**config_data)


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
        config = get_config()
        syslog_handler = logging.handlers.SysLogHandler(config.syslog_path)
        syslog_handler.level = getattr(logging, config.syslog_level)
        logging.basicConfig(
            level=logging.DEBUG,
            handlers=[
                logging.StreamHandler(),
                syslog_handler,
            ]
        )
    except ConfigFileError:
        # set up logging with generally sane defaults.
        syslog_handler = logging.handlers.SysLogHandler("/dev/log")
        syslog_handler.level = logging.DEBUG
        logging.basicConfig(
            level=logging.DEBUG,
            handlers=[
                logging.StreamHandler(),
                syslog_handler,
            ]
        )
        logger.error("Could not read config file.")
        return 1

    try:
        set_environment()
        sensor_data = get_sensor_data(config.hass_base_url)
        rules = ThermostatRules(config, sensor_data)
        strategy = rules.get_strategy()
        if strategy.matches_current_data(rules.transformer):
            logger.info(f"No change to strategy {strategy}")
        else:
            logger.info(f"Setting thermostat strategy {strategy} {strategy.explain_strategy()}")
            set_thermostat_values(config, strategy)
    except Exception as exc:
        logger.error(f"Setting thermostat failed:\n {traceback.format_exc()}")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

import jax_datetime as jdt
import jcm.forcing as _jcm_forcing
from jcm.forcing import _solar_from_date
from jcm.date import DateData


def freeze_solar_at(date_str: str, calendar: str = "365_day") -> None:
    """Freeze jcm solar geometry at the given date for the lifetime of the process.

    Monkeypatches jcm.forcing._solar_from_date so every call to
    ForcingData.select() returns a fixed SolarGeometry regardless of the
    running model clock.  Must be called before the model step function is
    JIT-compiled.

    Args:
        date_str: ISO-8601 date string, e.g. "2000-03-20" or "2000-06-21T12:00:00".
                  Include a time component to also fix the synodic (time-of-day) phase.
        calendar:  Calendar used to compute the fraction-of-year.  Must match
                   the calendar passed to Model / ForcingData.select().
                   Defaults to "365_day" (no-leap), which is the calendar used
                   in the coupled JCM-Veros experiment.
    """
    fixed_date = DateData.set_date(jdt.to_datetime(date_str))
    fixed_solar = _solar_from_date(fixed_date, calendar)
    _jcm_forcing._solar_from_date = lambda date, calendar: fixed_solar

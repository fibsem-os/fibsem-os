import numpy as np
MICRON_TO_METRE = 1e-6
METRE_TO_MICRON = 1.0 / MICRON_TO_METRE

MILLIMETRE_TO_METRE = 1e-3
METRE_TO_MILLIMETRE = 1.0 / MILLIMETRE_TO_METRE

SI_TO_KILO = 1e-3
KILO_TO_SI =  1/ SI_TO_KILO

SI_TO_MILLI = 1e3
MILLI_TO_SI = 1 / SI_TO_MILLI

SI_TO_MICRO = 1e6
MICRO_TO_SI = 1 / SI_TO_MICRO

SI_TO_NANO = 1e9
NANO_TO_SI = 1 / SI_TO_NANO

SI_TO_PICO = 1e12
PICO_TO_SI = 1 / SI_TO_PICO

RADIANS_TO_DEGREES = 180.0 / np.pi

DEGREES_TO_RADIANS = np.pi / 180.0

TO_PERCENTAGES = 100.0

FROM_PERCENTAGES = 1.0 / TO_PERCENTAGES

DEGREE_SYMBOL = "°"
MU_SYMBOL = "µ"
MICRON_SYMBOL = f"{MU_SYMBOL}m"
MILLIMETRE_SYMBOL = "mm"
MICROSECOND_SYMBOL = f"{MU_SYMBOL}s"
MILLISECOND_SYMBOL = "ms"
NANOSECOND_SYMBOL = "ns"
PICOSECOND_SYMBOL = "ps"

# Date / time format strings
DATETIME_FILE           = "%Y-%m-%d_%H-%M-%S"    # filename-safe datetime
DATETIME_COMPACT        = "%Y%m%d_%H%M%S"         # compact datetime (no separators)
DATETIME_DISPLAY        = "%Y-%m-%d %H:%M:%S"     # human-readable datetime
DATETIME_DISPLAY_SHORT  = "%Y-%m-%d  %H:%M"       # short display datetime
DATETIME_DISPLAY_FULL   = "%Y-%m-%d %H:%M:%S %p"  # datetime with AM/PM suffix
DATETIME_EXPERIMENT     = "%Y-%m-%d-%H-%M"         # experiment folder name
DATE_COMPACT            = "%Y%m%d"                 # compact date only
DATE_LONG               = "%B %d, %Y"              # e.g. "March 19, 2026"
TIME_FILE               = "%H-%M-%S"               # filename-safe time
TIME_DISPLAY            = "%H:%M:%S"               # colon-separated time
TIME_DISPLAY_AMPM_SHORT = "%I:%M%p"                # 12-hour time AM/PM (no space)
DATETIME_LOG            = "%Y-%m-%d-%I-%M-%S%p"   # legacy log datetime (note: %p unavailable on Windows)

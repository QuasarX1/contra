# SPDX-FileCopyrightText: 2024-present Christopher Rowe <chris.rowe19@outlook.com>
#
# SPDX-License-Identifier: None
#BAD_TESTING_SNIPSHOTS = ("snip_354_z000p225.0.hdf5", "snip_288_z000p770.0.hdf5", "snip_224_z001p605.0.hdf5", "snip_159_z002p794.0.hdf5")

import numpy as np
from typing import Any, Union, List, Dict, Tuple, Iterable, Iterator
from QuasarCode import Console
import datetime
import time



class Stopwatch(object):

    def __print(self, *values: str, at: float|None = None, **kwargs):
        t = time.time()
        if at is not None:
            t = at
        print("".join(["--|| TIME ||--  ", *[v.replace('\n', '\n                ') for v in values], ("" if len(values) == 0 else "\n                ") +  "(Stopwatch " + (f"\"{self.__name}\" " if self.__name is not None else "") + f"{t:.1f})"]), flush = True, **kwargs)

    _time_format = "%H:%M:%S[%f\u03BCs]"
    @staticmethod
    def srftime(dt: datetime.timedelta|float):
        if isinstance(dt, float):
            dt = datetime.timedelta(seconds = dt)
        return (datetime.datetime(year = 1, month = 1, day = 1) + dt).strftime(Stopwatch._time_format)

    def __init__(self, name: str|None = None, show_time_since_lap: bool = False, start_message: str|None = None, print_on_start: bool = True, message_on_new_line: bool = False, synchronise: "Stopwatch|None" = None):
        self.__name = name
        self.__use_lap_time_output = show_time_since_lap

        self.__running: bool
        self.__start_time: float
        self.__lap_times: List[float]
        self.__last_lap_time: float|None
        self.__stop_time: float|None

        self.start(start_message, print = print_on_start, synchronise = synchronise, message_on_new_line = message_on_new_line)

    def start(self, message: str|None = None, /, print: bool = True, synchronise: "Stopwatch|None" = None, message_on_new_line: bool = False) -> None:

        self.__lap_times = []
        self.__last_lap_time = None
        self.__stop_time = None

        self.__start_time = time.time()
        if synchronise is not None:
            self.__start_time = synchronise.start_time
        self.__lap_times.append(self.__start_time)
        self.__running = True

        if print:
            if synchronise is None:
                self.__print("START" + ("" if message is None else f"\n{message}" if message_on_new_line else f" {message}"), at = self.__start_time)
            else:
                self.__print(f"SYNCHRONISED START at {self.__start_time}" + ("" if message is None else f"\n{message}" if message_on_new_line else f" {message}"))

    def lap(self, message: str|None = None, /, print: bool = True, suppress_errors: bool = False, message_on_new_line: bool = False) -> None:
        t = time.time()

        if self.__running:
            self.__last_lap_time = t
            self.__lap_times.append(t)
        else:
            if not suppress_errors:
                raise RuntimeError("Timer not running. Unable to lap timer.")

        if print:
            self.__print(f"LAP {Stopwatch.srftime(self.__lap_times[-1] - self.__lap_times[-2])}" + ("" if message is None else f"\n{message}" if message_on_new_line else f" {message}"), at = self.__last_lap_time)

    def stop(self, message: str|None = None, /, print: bool = True, suppress_errors: bool = False, message_on_new_line: bool = False) -> None:
        t = time.time()

        if self.__running:
            self.__stop_time = t
            self.__lap_times.append(self.__stop_time)
            self.__running = False
        else:
            if not suppress_errors:
                raise RuntimeError("Timer not running. Unable to stop timer.")

        if print:
            self.__print(f"STOPPED {Stopwatch.srftime(self.__stop_time - self.__start_time)}" + ("" if message is None else f"\n{message}" if message_on_new_line else f" {message}"), at = self.__stop_time)

    @property
    def start_time(self) -> float:
        return self.__start_time

    @property
    def lap_time(self) -> float|None:
        return self.__last_lap_time

    @property
    def stop_time(self) -> float|None:
        return self.__stop_time

    @property
    def lap_times(self) -> Tuple[float, ...]:
        return tuple(self.__lap_times[1:])
    
    def get_elapsed_time(self, at: float|None = None, string: bool = False) -> float|str:
        t = time.time()
        if at is not None:
            t = at
        elif not self.__running:
            t = self.__stop_time

        if string:
            return Stopwatch.srftime(t - self.__start_time)
        else:
            return t - self.__start_time
    
    def get_elapsed_time_since_last_lap(self, at: float|None = None, string: bool = False) -> float:
        t = time.time()
        if at is not None:
            t = at
        elif not self.__running:
            t = self.__stop_time

        if string:
            return Stopwatch.srftime(t - self.__lap_times[-1])
        else:
            return t - self.__lap_times[-1]

    def get_lap_delta_times(self, string: bool = False) -> Tuple[float, ...]|Tuple[str, ...]:
        return tuple([(float if not string else Stopwatch.srftime)(v) for v in (np.array(self.__lap_times[1:]) - np.array(self.__lap_times[:-1]))])

    def get_elapsed_lap_delta_times(self, string: bool = False) -> Tuple[float, ...]|Tuple[str, ...]:
        return tuple([(float if not string else Stopwatch.srftime)(v) for v in (np.array(self.__lap_times[1:]) - self.__start_time)])
    
    def _get_total_string(self, at: float) -> str:
        return "(Stopwatch" + (f" \"{self.__name}\"" if self.__name is not None else "") + f") {self.get_elapsed_time(at = at, string = True)}"
    
    def _get_lap_string(self, at: float) -> str:
        return "(Stopwatch" + (f" \"{self.__name}\"" if self.__name is not None else "") + f") {self.get_elapsed_time_since_last_lap(at = at, string = True)}"
    
    def _get_string(self, at: float) -> str:
        if self.__use_lap_time_output:
            return self._get_lap_string(at)
        else:
            return self._get_total_string(at)

    def __str__(self):
        t = time.time()
        return self._get_string(at = t if self.__running else self.__stop_time)

    def __repr__(self) -> str:
        t = time.time()
        return self._get_string(at = t if self.__running else self.__stop_time)

    def print(self, /, *args, at: float|None = None, **kwargs) -> None:
        t = time.time()
        if at is not None:
            t = at
        elif not self.__running:
            t = self.__stop_time
        self.__print(self._get_string(at = t), *args, at = t, **kwargs)

    def print_info(self, /, *args, at: float|None = None, **kwargs) -> None:
        t = time.time()
        if at is not None:
            t = at
        elif not self.__running:
            t = self.__stop_time
        Console.print_info(self._get_string(at = t), *args, **kwargs)

    def print_verbose_info(self, /, *args, at: float|None = None, **kwargs) -> None:
        t = time.time()
        if at is not None:
            t = at
        elif not self.__running:
            t = self.__stop_time
        Console.print_verbose_info(self._get_string(at = t), *args, **kwargs)

    def print_warning(self, /, *args, at: float|None = None, **kwargs) -> None:
        t = time.time()
        if at is not None:
            t = at
        elif not self.__running:
            t = self.__stop_time
        Console.print_warning(self._get_string(at = t), *args, **kwargs)

    def print_verbose_warning(self, /, *args, at: float|None = None, **kwargs) -> None:
        t = time.time()
        if at is not None:
            t = at
        elif not self.__running:
            t = self.__stop_time
        Console.print_warning(self._get_string(at = t), *args, **kwargs)

    def print_debug(self, /, *args, at: float|None = None, **kwargs) -> None:
        t = time.time()
        if at is not None:
            t = at
        elif not self.__running:
            t = self.__stop_time
        Console.print_debug(self._get_string(at = t), *args, **kwargs)

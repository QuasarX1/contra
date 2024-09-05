# SPDX-FileCopyrightText: 2024-present Christopher Rowe <chris.rowe19@outlook.com>
#
# SPDX-License-Identifier: None
from .._ParticleType import ParticleType

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Sized
from typing import Generic, TypeVar
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from scipy.constants import gravitational_constant
import os
from functools import singledispatchmethod

import numpy as np
from unyt import unyt_array, unyt_quantity
from QuasarCode.Tools import Struct



class Interface(ABC):
    """
    Base class for interface types.
    """
    def __new__(cls, *args, **kwargss):
        ensure_not_interface(cls, ISimulation)
        return super().__new__(cls)

T_Interface = TypeVar("T_Interface", bound = Interface)
U_Interface = TypeVar("U_Interface", bound = Interface)

def check_interface(cls: type[U_Interface], interface_type: type[T_Interface]) -> bool:
    return cls is interface_type
def ensure_not_interface(cls: type[U_Interface], interface_type: type[T_Interface]):
    if check_interface(cls, interface_type):
        raise TypeError(f"Abstract interface type {interface_type.__name__} cannot be instantiated. To use this type, create an instance of a subclass.")



class ISimulation(Interface):
    def __new__(cls, *args, **kwargss):
        ensure_not_interface(cls, ISimulation)
        return super().__new__(cls)
T_ISimulation = TypeVar("T_ISimulation", bound = ISimulation)



class ISimulationData(Interface):
    """
    Interface indicating types used for reading simulation data.
    """
    def __new__(cls, *args, **kwargs):
        ensure_not_interface(cls, ISimulationData)
        return super().__new__(cls, *args, **kwargs)
T_ISimulationData = TypeVar("T_ISimulationData", bound = ISimulationData)
class SimulationDataBase(Generic[T_ISimulation], ISimulationData):
    pass



class ISimulationFileTreeLeaf(Interface, Sized):
    def __new__(cls, *args, **kwargs):
        ensure_not_interface(cls, ISimulationFileTreeLeaf)
        return super().__new__(cls, *args, **kwargs)
    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError()
    @abstractmethod
    def load(self) -> ISimulationData:
        raise NotImplementedError()
    @property
    @abstractmethod
    def number(self) -> str:
        raise NotImplementedError()
    @property
    @abstractmethod
    def number_numerical(self) -> int:
        raise NotImplementedError()
    @property
    @abstractmethod
    def filepaths(self) -> tuple[str, ...]:
        raise NotImplementedError()
class SimulationFileTreeLeafBase(Generic[T_ISimulationData], ISimulationFileTreeLeaf):
    @abstractmethod
    def load(self) -> T_ISimulationData:
        raise NotImplementedError()



class ISimulationFileTree(Interface, Iterable, Sized):
    def __new__(cls, *args, **kwargs):
        ensure_not_interface(cls, ISimulationFileTree)
        return super().__new__(cls, *args, **kwargs)
    @abstractmethod
    def __iter__(self) -> Iterator[ISimulationFileTreeLeaf]:
        raise NotImplementedError()
    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError()
    @abstractmethod
    def __get_item__(self, key: int|slice) -> ISimulationFileTreeLeaf|tuple[ISimulationFileTreeLeaf, ...]:
        raise NotImplementedError()
    @abstractmethod
    def get_info(self) -> tuple[ISimulationFileTreeLeaf, ...]:
        raise NotImplementedError()
    @abstractmethod
    def get_numbers(self) -> tuple[str, ...]:
        raise NotImplementedError()
    def get_by_number(self, number: str) -> ISimulationFileTreeLeaf:
        raise NotImplementedError()
    def get_by_numbers(self, number: Iterable[str]) -> tuple[ISimulationFileTreeLeaf, ...]:
        raise NotImplementedError()
class SimulationFileTreeBase(Generic[T_ISimulationData], ISimulationFileTree):
    def __init__(self, directory: str):
        self.__directory = os.path.realpath(directory)
    @property
    def directory(self) -> str:
        return self.__directory
    @abstractmethod
    def __iter__(self) -> Iterator[SimulationFileTreeLeafBase[T_ISimulationData]]:
        raise NotImplementedError()
    @abstractmethod
    def __get_item__(self, key: int|slice) -> SimulationFileTreeLeafBase[T_ISimulationData]|tuple[SimulationFileTreeLeafBase[T_ISimulationData], ...]:
        raise NotImplementedError()
    @abstractmethod
    def get_info(self) -> tuple[SimulationFileTreeLeafBase[T_ISimulationData], ...]:
        raise NotImplementedError()
    @abstractmethod
    def get_by_number(self, number: str) -> SimulationFileTreeLeafBase[T_ISimulationData]:
        raise NotImplementedError()
    def get_by_numbers(self, number: Iterable[str]) -> tuple[SimulationFileTreeLeafBase[T_ISimulationData], ...]:
        return tuple([self.get_by_number(n) for n in number])

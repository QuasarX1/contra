# SPDX-FileCopyrightText: 2024-present Christopher Rowe <chris.rowe19@outlook.com>
#
# SPDX-License-Identifier: None
import numpy as np
from numpy._typing import DTypeLike
import multiprocessing
#from multiprocessing.synchronize import Lock as LockType
from threading import Lock as LockType
#from _thread import LockType
from multiprocessing.managers import SyncManager
from typing import cast as typing_cast, Any, Union, Tuple, Iterable
from functools import reduce
import operator
from QuasarCode import Console, Settings
from QuasarCode.Tools import Struct, AutoProperty, TypedAutoProperty, TypeShield, NestedTypeShield
#import traceback
import os
import uuid

COUNTER_SHAPE = (1, )
COUNTER_DATATYPE = np.int64
COUNTER_N_BYTES = 64

if Settings.debug:
    from multiprocessing.shared_memory import SharedMemory as SharedMemory_base
    class SharedMemory(SharedMemory_base):
        def unlink(self):
            #for line in traceback.format_stack():
            #    print(line.strip())
            Console.print_debug(f"UNLINKED {self._name}")
            super().unlink()
else:
    from multiprocessing.shared_memory import SharedMemory

class SharedArray_TransmissionData(Struct):
    lock: LockType = AutoProperty[LockType]()# TypedAutoProperty[LockType](TypeShield(LockType))
    counter_name: str = TypedAutoProperty[str](TypeShield(str))
    data_name: str = TypedAutoProperty[str](TypeShield(str))
    shape: Tuple[int, ...] = TypedAutoProperty[Tuple[int, ...]](NestedTypeShield(tuple, int))
    dtype: DTypeLike = AutoProperty[DTypeLike]()
    debugging_name: Union[str, None] = AutoProperty[Union[str, None]]()

    def load(self) -> "SharedArray":
        return SharedArray.load(self)



class SharedArray(object):
    """
    
    """

    _manager: Union[SyncManager, None] = None
    _is_initialised: bool = False

    @staticmethod
    def _make_lock() -> LockType:
        if not SharedArray._is_initialised:
            SharedArray.initialise()
        return typing_cast(SyncManager, SharedArray._manager).Lock()

    @staticmethod
    def initialise(manager: Union[SyncManager, None] = None):
        SharedArray._manager = manager if manager is not None else multiprocessing.Manager()
        SharedArray._is_initialised = True

    def __init__(self, lock: LockType, counter_memory: SharedMemory, data_memory: SharedMemory, shape: Tuple[int, ...], dtype: DTypeLike, debugging_name: Union[str, None]) -> None:
        if not SharedArray._is_initialised:
            SharedArray.initialise()
        self.__debugging_name: Union[str, None] = debugging_name
        self.__counter_memory: SharedMemory = counter_memory
        self.__data_memory: SharedMemory = data_memory
        self.__info = SharedArray_TransmissionData(
            lock = lock,
            counter_name = self.__counter_memory.name,
            data_name = self.__data_memory.name,
            shape = shape,
            dtype = dtype,
            debugging_name = debugging_name
        )
        self.__counter_array: np.ndarray = np.ndarray(shape = COUNTER_SHAPE, dtype = COUNTER_DATATYPE, buffer = self.__counter_memory.buf)
        self.__data_array: np.ndarray = np.ndarray(shape = shape, dtype = dtype, buffer = self.__data_memory.buf)
        with self.__info.lock:
            self.__counter_array[0] = self.__counter_array[0] + 1
        self.__freed = False

    def __reset_counter(self):
        """
        Reset the counter to 1.
        Only to be called when creating a new array as the counter will be garbage otherwise!
        """
        with self.__info.lock:
            self.__counter_array[0] = 1

    def free(self, /, force = False):
        """
        Free up any resources.
        This MUST be called before the object goes out of scope and gets garbage collected!
        """
        with self.__info.lock:
            self.__counter_array[0] = self.__counter_array[0] - 1
            tidy_up = self.__counter_array[0] == 0
            self.__counter_memory.close()
            self.__data_memory.close()
            if tidy_up or force:
                self.__counter_memory.unlink()
                self.__data_memory.unlink()
            self.__freed = True

    def __enter__(self) -> "SharedArray":
        return self
    def __exit__(self, exc_type, exc_value, exc_tb):
        del self

    @property
    def data(self) -> np.ndarray:
        return self.__data_array

    @property
    def info(self) -> SharedArray_TransmissionData:
        return self.__info
    
    @property
    def lock(self) -> LockType:
        return self.__info.lock

    @staticmethod
    def n_bytes_in_array(shape: Union[int, Tuple[int, ...]], dtype: DTypeLike) -> int:
        if isinstance(shape, int):
            shape = (shape, )
        return np.array((1, ), dtype = dtype)[0].nbytes * reduce(operator.mul, shape, 1)

    @staticmethod
    def create(shape: Union[int, Tuple[int, ...]], dtype: DTypeLike = np.float64, name: Union[str, None] = None) -> "SharedArray":
        if not isinstance(shape, tuple):
            if isinstance(shape, Iterable):
                shape = tuple([int(v) for v in shape])
            else:
                shape = (int(shape), )
        if not isinstance(shape[0], int):
            shape = tuple([int(v) for v in shape])
        id = uuid.uuid4()
        obj = SharedArray(
            SharedArray._make_lock(),
            counter_memory = SharedMemory(create = True, size = COUNTER_N_BYTES, name = f"{name}_counter_{id}" if name is not None else None),
            data_memory = SharedMemory(create = True, size = SharedArray.n_bytes_in_array(shape, dtype), name = f"{name}_data_{id}" if name is not None else None),
            shape = shape,
            dtype = dtype,
            debugging_name = f"{name} <{id}|{os.getpid()}>" if name is not None else None
        )
        obj.__reset_counter()
        return obj

    @staticmethod
    def load(from_info: SharedArray_TransmissionData) -> "SharedArray":
#        Console.print_debug(f"Loading {from_info.debugging_name if from_info.debugging_name is not None else from_info.data_name.rsplit('_', maxsplit = 2)[0]}")
#        print(f"    {from_info.counter_name}")
        return SharedArray(
            from_info.lock,
            counter_memory = SharedMemory(name = from_info.counter_name),
            data_memory = SharedMemory(name = from_info.data_name),
            shape = from_info.shape,
            dtype = from_info.dtype,
            debugging_name = f"{from_info.debugging_name} ({os.getpid()})" if from_info.debugging_name is not None else None
        )
        
    def fill(self, value: Any) -> "SharedArray":
        """
        This method returns a reference back to itself NOT a copy!
        """
        self.data[:] = value
        return self

    @staticmethod
    def full(shape: Union[int, Tuple[int, ...]], value: Any, dtype: DTypeLike = np.float64, name: Union[str, None] = None) -> "SharedArray":
        return SharedArray.create(shape, dtype, name).fill(value)
    
    @staticmethod
    def zeros(shape: Union[int, Tuple[int, ...]], dtype: DTypeLike = np.float64, name: Union[str, None] = None) -> "SharedArray":
        return SharedArray.full(shape, 0.0, dtype, name)
    
    @staticmethod
    def ones(shape: Union[int, Tuple[int, ...]], dtype: DTypeLike = np.float64, name: Union[str, None] = None) -> "SharedArray":
        return SharedArray.full(shape, 1.0, dtype, name)

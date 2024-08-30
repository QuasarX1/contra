# SPDX-FileCopyrightText: 2024-present Christopher Rowe <chris.rowe19@outlook.com>
#
# SPDX-License-Identifier: None
import multiprocessing.pool
import numpy as np
#from numpy._typing import DTypeLike, ArrayLike, NDArray
from numpy.typing import DTypeLike, ArrayLike, NDArray
import multiprocessing
#from multiprocessing.synchronize import Lock as LockType
from threading import Lock as LockType
#from _thread import LockType
from multiprocessing.managers import SyncManager
from typing import TYPE_CHECKING, cast as typing_cast, Any, Union, Tuple, Generic, TypeVar, ParamSpec, Concatenate
from collections.abc import Callable, Iterable, Collection
if TYPE_CHECKING:
    from _typeshed import SupportsRichComparison
from functools import reduce
import operator
from QuasarCode import Console, Settings
from QuasarCode.Tools import Struct, AutoProperty, TypedAutoProperty, TypeShield, NestedTypeShield
import traceback
import os
import uuid
import h5py

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
#    name_pattern: TypedAutoProperty[str](TypeShield(str))
    lock: LockType = AutoProperty[LockType]()# TypedAutoProperty[LockType](TypeShield(LockType))
    counter_name: str = TypedAutoProperty[str](TypeShield(str))
    data_name: str = TypedAutoProperty[str](TypeShield(str))
    shape: Tuple[int, ...] = TypedAutoProperty[Tuple[int, ...]](NestedTypeShield(tuple, int))
    dtype: DTypeLike = AutoProperty[DTypeLike]()
    debugging_name: Union[str, None] = AutoProperty[Union[str, None]]()

    def load(self, shepherd: "SharedArray_Shepherd|None" = None) -> "SharedArray":
        return SharedArray.load(self, shepherd = shepherd)



class SharedArray(object):
    """
    WARNING: make sure to either call `<instance>.free()` before the wrapper object goes out of scope or use a context manager (`with <instance>:`) to avoid memory leaks!

    Wrapper type for numpy array objects in shared memory.

    Allows allocation of memory accessible by other processes.

    Convinience methods avalible for loading data from HDF5 files without the need to make additional coppies.

    Access the underlying array using `<instance>.data`.

    Access the associated mutex lock using `<instance>.lock`.

    To access the array from a different process, pass the `<instance>.info` object to the process and then pass it to `SharedArray.load(<info>)`.
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

    def __init__(self, lock: LockType, counter_memory: SharedMemory, data_memory: SharedMemory, shape: Tuple[int, ...], dtype: DTypeLike, debugging_name: Union[str, None], shepherd: "SharedArray_Shepherd|None" = None) -> None:
        self.__shepherd = shepherd
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

    def _set_shepheard(self, shepherd: "SharedArray_Shepherd") -> None:
        if self.__shepherd is not None and self.__shepherd != shepherd:
            raise RuntimeError("SharedArray object already has an associated SharedArray_Shepherd object.")
        else:
            self.__shepherd = shepherd

    def _internal_free(self, /, force = False, remove_from_shepherd = True) -> None:
        with self.__info.lock:
            self.__counter_array[0] = self.__counter_array[0] - 1
            tidy_up = self.__counter_array[0] == 0
            self.__counter_memory.close()
            self.__data_memory.close()
            if tidy_up or force:
                self.__counter_memory.unlink()
                self.__data_memory.unlink()
            self.__freed = True
            if self.__shepherd is not None and remove_from_shepherd:
                self.__shepherd.remove(self)
    def free(self, /, force = False) -> None:
        """
        Free up any resources.
        This MUST be called before the object goes out of scope and gets garbage collected!
        """
        self._internal_free(force = force, remove_from_shepherd = True)

    def __enter__(self) -> "SharedArray":
        return self
    def __exit__(self, exc_type, exc_value, exc_tb):
        self.free()
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
    def create(shape: Union[int, Tuple[int, ...]], dtype: DTypeLike = np.float64, name: Union[str, None] = None, shepherd: "SharedArray_Shepherd|None" = None) -> "SharedArray":
        if not isinstance(shape, tuple):
            if isinstance(shape, Iterable):
                shape = tuple([int(v) for v in shape])
            else:
                shape = (int(shape), )
        if not isinstance(shape[0], int):
            shape = tuple([int(v) for v in shape])
        id = uuid.uuid4()
        a = SharedMemory(create = True, size = COUNTER_N_BYTES, name = f"{name}_counter_{id}" if name is not None else None)
        b = SharedMemory(create = True, size = SharedArray.n_bytes_in_array(shape, dtype), name = f"{name}_data_{id}" if name is not None else None)
        temp_counter_array: np.ndarray = np.ndarray(shape = COUNTER_SHAPE, dtype = COUNTER_DATATYPE, buffer = a.buf)
        temp_counter_array[0] = 0
        obj = SharedArray(
#            name_pattern = a.name.replace("counter", "{}"),
            lock = SharedArray._make_lock(),
            counter_memory = a,
            data_memory = b,
            shape = shape,
            dtype = dtype,
            debugging_name = f"{name} <{id}|{os.getpid()}>" if name is not None else None
        )
        #obj.__reset_counter()
        if shepherd is not None:
            shepherd.add(obj)
        return obj

    @staticmethod
    def load(from_info: SharedArray_TransmissionData, shepherd: "SharedArray_Shepherd|None" = None) -> "SharedArray":
#        Console.print_debug(f"Loading {from_info.debugging_name if from_info.debugging_name is not None else from_info.data_name.rsplit('_', maxsplit = 2)[0]}")
#        print(f"    {from_info.counter_name}")
        obj = SharedArray(
            from_info.lock,
            counter_memory = SharedMemory(name = from_info.counter_name),
            data_memory = SharedMemory(name = from_info.data_name),
            shape = from_info.shape,
            dtype = from_info.dtype,
            debugging_name = f"{from_info.debugging_name} ({os.getpid()})" if from_info.debugging_name is not None else None
        )
        if shepherd is not None:
            shepherd.add(obj)
        return obj

    @staticmethod
    def create_from_hdf5(dataset: h5py.Dataset, target_slices: Tuple[slice, ...]|None = None, /, name: Union[str, None] = None, shepherd: "SharedArray_Shepherd|None" = None) -> "SharedArray":
        """
        Reads an h5py dataset directly into shared memory.

        Name will be auto-generated from the dataset path. If reading the same dataset multiple times, make sure to specify unique names!

        Use numpy.s_[:, :, ...] to specify the target_slices parameter as a literal. A tuple with a length of 0 will be treated as None.

        Usage:
            file = h5py.File("/path/to/file.hdf5")
            dataset1 = SharedArray.create_from_hdf5(file["group1/dataset1"], name = "file1-dataset1")
        """
        if name is None:
            name = dataset.name#TODO: append filepath??? is this the full path to the dataset???
        if target_slices is None or len(target_slices) == 0:
            target_region = None
            target_shape = dataset.shape
        else:
            if len(target_slices) > len(dataset.shape):
                raise IndexError(f"Too many indexes for shape of target dataset. Got {len(target_slices)} but dataset has only {len(dataset.shape)} dimensions!")
            target_shape = list(dataset.shape) # Do this in case not all the positions are sliced!
            for i, (s_i, len_i) in enumerate(zip(target_slices, dataset.shape[:len(target_slices)])):
                start, stop, step = s_i.indices(len_i)
                target_shape[i] = (stop - start + (step - 1)) // step
        shared_array = SharedArray.create(target_shape, dataset.dtype, name, shepherd)
        dataset.read_direct(shared_array.data, target_region)
        return shared_array

    @staticmethod#TODO: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def partial_create_from_hdf5(shape: int|Tuple[int, ...], destination_slices: Tuple[slice, ...], dataset: h5py.Dataset, target_slices: Tuple[slice, ...]|None = None, /, name: Union[str, None] = None, shepherd: "SharedArray_Shepherd|None" = None) -> "SharedArray":
        """
        Reads an h5py dataset directly into a portion of a shared memory array.

        Name will be auto-generated from the dataset path. If reading the same dataset multiple times, make sure to specify unique names!

        Use numpy.s_[:, :, ...] to specify the destination_slices & target_slices parameters as literals. A tuple with a length of 0 will be treated as None.

        Usage:
            file0 = h5py.File("/path/to/file.0.hdf5")
            dataset1 = SharedArray.partial_create_from_hdf5((1000, 100), np.s_[0:500], file0["group1/dataset1"], name = "dataset1")
            file1 = h5py.File("/path/to/file.1.hdf5")
            dataset1.populate_from_hdf5(np.s_[500:1000], file1["group1/dataset1"])
        """
        if not isinstance(shape, tuple):
            if isinstance(shape, Iterable):
                shape = tuple([int(v) for v in shape])
            else:
                shape = (int(shape), )
        if not isinstance(shape[0], int):
            shape = tuple([int(v) for v in shape])

        if len(destination_slices) > len(shape):
            raise IndexError(f"Too many indexes for the specified shape. Got {len(destination_slices)} but the specified shape has only {len(shape)} dimensions!")

        shared_array = SharedArray.create(shape, dataset.dtype, name, shepherd)

        if name is None:
            name = dataset.name#TODO: append filepath??? is this the full path to the dataset???
        if target_slices is None or len(target_slices) == 0:
            target_region = None
            target_shape = dataset.shape
        else:
            if len(target_slices) > len(dataset.shape):
                raise IndexError(f"Too many indexes for shape of target dataset. Got {len(target_slices)} but dataset has only {len(dataset.shape)} dimensions!")
            target_shape = list(dataset.shape) # Do this in case not all the positions are sliced!
            for i, (s_i, len_i) in enumerate(zip(target_slices, dataset.shape[:len(target_slices)])):
                start, stop, step = s_i.indices(len_i)
                target_shape[i] = (stop - start + (step - 1)) // step
        shared_array = SharedArray.create(target_shape, dataset.dtype, name, shepherd)
        dataset.read_direct(shared_array.data, target_region, destination_slices)
        return shared_array

    def populate_from_hdf5(self, destination_slices: Tuple[slice, ...], dataset: h5py.Dataset, target_slices: Tuple[slice, ...]|None = None, /, name: Union[str, None] = None, lock: bool = True) -> "SharedArray":
        """
        Reads an h5py dataset directly into a portion of the shared memory array.

        Use numpy.s_[:, :, ...] to specify the destination_slices & target_slices parameters as literals. A tuple with a length of 0 will be treated as None.

        Usage:
            file0 = h5py.File("/path/to/file.0.hdf5")
            dataset1 = SharedArray.partial_create_from_hdf5((1000, 100), np.s_[0:500], file0["group1/dataset1"], name = "dataset1")
            file1 = h5py.File("/path/to/file.1.hdf5")
            dataset1.populate_from_hdf5(np.s_[500:1000], file1["group1/dataset1"])
        """

        if len(destination_slices) > len(self.data.shape):
            raise IndexError(f"Too many indexes for the array. Got {len(destination_slices)} but this array has only {len(self.data.shape)} dimensions!")

        if target_slices is None or len(target_slices) == 0:
            target_region = None
            target_shape = dataset.shape
        else:
            if len(target_slices) > len(dataset.shape):
                raise IndexError(f"Too many indexes for shape of target dataset. Got {len(target_slices)} but dataset has only {len(dataset.shape)} dimensions!")
            target_shape = list(dataset.shape) # Do this in case not all the positions are sliced!
            for i, (s_i, len_i) in enumerate(zip(target_slices, dataset.shape[:len(target_slices)])):
                start, stop, step = s_i.indices(len_i)
                target_shape[i] = (stop - start + (step - 1)) // step

        if lock:
            self.lock.acquire(blocking = True, timeout = -1)
        dataset.read_direct(self.data, target_region, destination_slices)
        if lock:
            self.lock.release()
        return self

    @staticmethod
    def create_from_hdf5_parallel_datasets(*datasets: h5py.Dataset, name: Union[str, None] = None, shepherd: "SharedArray_Shepherd|None" = None) -> "SharedArray":
        """
        Reads a sequence of h5py datasets directly into a contiguous block of shared memory.

        Name will be auto-generated from the dataset path. If reading the same dataset multiple times, make sure to specify unique names!

        Usage:
            files = [h5py.File(f"/path/to/file.{i}.hdf5") for i in range(10)]
            dataset1 = SharedArray.create_from_hdf5_parallel_datasets(*[f["group1/dataset1"] for f in files], name = "dataset1")
        """
        if len(datasets) == 0:
            raise IndexError("No datasets provided. A minimum of one dataset is required.")

        # Check that all datasets are stackable and get the total length
        dataset_slices = [np.index_exp[0 : datasets[0].shape[0]]]
        total_length = datasets[0].shape[0]
        stackable_shape = datasets[0].shape[1:]
        for d in datasets:
            if len(d.shape) != len(datasets[0].shape):
                raise IndexError("Datasets have inconsistent numbers of dimensions.")
            elif d.shape[:1] != stackable_shape:
                raise IndexError("Datasets have inconsistent shapes in dimensions after first.")
            dataset_slices.append(np.index_exp[total_length : total_length + d.shape[0]])
            total_length += d.shape[0]
        final_shape: Tuple[int, ...] = (total_length, *stackable_shape)

        shared_array = SharedArray.partial_create_from_hdf5(final_shape, dataset_slices[0], datasets[0], name = name, shepherd = shepherd)
        for d, s in zip(datasets[1:], dataset_slices[1:]):
            shared_array.populate_from_hdf5(s, d, lock = False)

        return shared_array

    @staticmethod
    def create_from_hdf5_parallel_files(dataset_path: str|Iterable[str], *files: h5py.File, name: Union[str, None] = None, shepherd: "SharedArray_Shepherd|None" = None) -> "SharedArray":
        """
        Reads a sequence of h5py datasets directly into a contiguous block of shared memory.

        Convinience wrapper for `SharedArray.create_from_hdf5_parallel_datasets` where all files share a common structure.

        Name will be auto-generated from the dataset path. If reading the same dataset multiple times, make sure to specify unique names!

        Usage:
            files = [h5py.File(f"/path/to/file.{i}.hdf5") for i in range(10)]
            dataset1 = SharedArray.create_from_hdf5_parallel_files("group1/dataset1", *files, name = "dataset1")
        """
        if len(files) == 0:
            raise IndexError("No files provided. A minimum of one file is required.")
        if not isinstance(dataset_path, str):
            dataset_path = "/".join(dataset_path)
        return SharedArray.create_from_hdf5_parallel_datasets(*[f[dataset_path] for f in files], name = name, shepherd = shepherd)

    @staticmethod
    def copy_to_shared(array: np.ndarray, /, name: Union[str, None] = None, shepherd: "SharedArray_Shepherd|None" = None) -> "SharedArray":
        """
        Coppies an array's data into shared memory.

        To automatically delete the original array, use `SharedArray.as_shared`.

        Usage:
            shared_input_array = SharedArray.copy_to_shared(input_array, name = "input-array")
        """
        shared_array = SharedArray.create(array.shape, array.dtype, name, shepherd)
        np.copyto(shared_array.data, array)
        return shared_array

    @staticmethod
    def as_shared(array: np.ndarray, /, name: Union[str, None] = None, shepherd: "SharedArray_Shepherd|None" = None) -> "SharedArray":
        """
        WARNING: this WILL delete the input array!

        Moves an array's data into shared memory and deletes the original array.

        To avoid the deletion of the original array, use `SharedArray.copy_to_shared`.

        Usage:
            input_array = SharedArray.as_shared(input_array, name = "input-array")
        """
        shared_array = SharedArray.copy_to_shared(array, name, shepherd)
        del array
        return shared_array
        
    def fill(self, value: Any) -> "SharedArray":
        """
        This method returns a reference back to itself NOT a copy!
        """
        self.data.fill(value)
        return self

    @staticmethod
    def full(shape: Union[int, Tuple[int, ...]], value: Any, dtype: DTypeLike = np.float64, name: Union[str, None] = None, shepherd: "SharedArray_Shepherd|None" = None) -> "SharedArray":
        return SharedArray.create(shape, dtype, name, shepherd).fill(value)
    
    @staticmethod
    def zeros(shape: Union[int, Tuple[int, ...]], dtype: DTypeLike = np.float64, name: Union[str, None] = None, shepherd: "SharedArray_Shepherd|None" = None) -> "SharedArray":
        return SharedArray.full(shape, 0.0, dtype, name, shepherd)
    
    @staticmethod
    def ones(shape: Union[int, Tuple[int, ...]], dtype: DTypeLike = np.float64, name: Union[str, None] = None, shepherd: "SharedArray_Shepherd|None" = None) -> "SharedArray":
        return SharedArray.full(shape, 1.0, dtype, name, shepherd)

    @staticmethod
    def copyto(dst: "SharedArray|NDArray[Any]", src: "SharedArray|ArrayLike", casting: "np._CastingKind|None" = None, where: "np._ArrayLikeBool_co|None" = ...) -> None:
        """
        Performs `numpy.copyto` on two arrays - one or both of which could be a SharedArray object
        """
        if isinstance(dst, SharedArray) or isinstance(src, SharedArray):
            if dst == src:
                return # Avoids a deadlock if the same SharedArray object is both the source and destination
            if isinstance(dst, SharedArray):
                dst.lock.acquire(blocking = True, timeout = -1)
            if isinstance(src, SharedArray):
                src.lock.acquire(blocking = True, timeout = -1)
        np.copyto(dst.data if isinstance(dst, SharedArray) else dst, src.data if isinstance(src, SharedArray) else src, casting, where)
        if isinstance(dst, SharedArray):
            dst.lock.release()
        if isinstance(src, SharedArray):
            src.lock.release()

    def copy(self, name: str|None = None, shepherd: "SharedArray_Shepherd|None" = None) -> "SharedArray":
        """
        WARNING: calling this on multiple threads/processes from the same shared array WILL result in errors!

        Creates a copy of the underlying array data.
        """
        new_array = SharedArray.create(self.data.shape, self.data.dtype, name, shepherd)#TODO: NAME?!?!
        with self.lock:
            np.copyto(new_array.data, self.data)
        return new_array

#    def get_copy(self) -> "SharedArray":
#        """
#        Gets the copy of the underlying array data or creates one if a copy does not yet exist.
#
#        Equivilant to SharedArray.copy, but can be called on multiple threads/processes without issue.
#        """
#        with self.lock:
#            try:
#                return
#            except FileNotFoundError:
#                return



class SharedArray_Shepherd(set[SharedArray]):
    def add(self, element: SharedArray, /) -> SharedArray:
        element._set_shepheard(self)
        super().add(element)
        return element
    def __add__(self, element: SharedArray, /):
        _ = self.add(element)
        return self
    def __iadd__(self, element: SharedArray, /):
        _ = self.add(element)
        return self
    def remove(self, element: SharedArray, /) -> None:
        super().remove(element)
    def __sub__(self, element: SharedArray, /):
        self.remove(element)
        return self
    def __isub__(self, element: SharedArray, /):
        self.remove(element)
        return self
    def free(self) -> None:
        for arr in self:
            try:
                arr._internal_free(remove_from_shepherd = False)
            except BaseException as e:
                Console.print_warning(f"Unable to free memory from {arr}. This may be because this object has already been deleted/freed explicitly.")
                has_message = str(e) != ""
                Console.print_debug("Traceback (most recent call last):\n" + "".join(traceback.format_tb(e.__traceback__)) + type(e).__name__ + (f": {str(e)}" if has_message else ""))
        self.clear()
    def __enter__(self) -> "SharedArray_Shepherd":
        return self
    def __exit__(self, exc_type, exc_value, exc_tb):
        self.free()




I =  TypeVar("I")
T =  TypeVar("T")
P =  ParamSpec("P")
class SharedArray_ParallelJob(Generic[I, P, T]):
    """
    Define a task to be executed in parallel using multiprocessing.

    SharedArray objects passed as arguments will be automatically re-connected inside the worker processes.

    Set `minimum_chunk_size <= 0` and leave `number_of_chunks` unset (`None`) to use one worker per job key.
    """
    def __init__(self, target: Callable[Concatenate[I, P], T], /, pool_size: int = 8, minimum_chunk_size: int = 1, number_of_chunks: int|None = None, ignore_return: bool = False, key_order: Callable[[I], "SupportsRichComparison"]|None = None):
        self.__target = target
        self.__ignore_return = ignore_return
        self.__manager = multiprocessing.Manager()
        self.__pool_size = pool_size
        self.__min_chunk_size = minimum_chunk_size
        self.__number_of_chunks: int|None = number_of_chunks
        self.__results: dict[I, T] = {} if self.__ignore_return else typing_cast(dict[I, T], self.__manager.dict())
        self.__key_order_func = key_order # if key_order is not None else typing_cast(Callable[[I], "SupportsRichComparison"], float)
    def execute(self, job_keys: Collection[I]|Iterable[I]|int, *args: P.args, **kwargs: P.kwargs):
        """
        Run the pool using the specified arguments.

        `job_keys` should either be an `Iterable` of the expected type or may, optionally, be an integer if the first argument of the target function expects an integer.
        """
        if not isinstance(job_keys, Collection):
            if not isinstance(job_keys, Iterable):
                # Should only be an int object
                job_keys = typing_cast(Iterable[I], range(job_keys))
            job_keys = typing_cast(Collection[I], list(job_keys))
        converted_args = [(value.info if isinstance(value, SharedArray) else value) for value in args]
        converted_kwargs = { key : (value.info if isinstance(value, SharedArray) else value) for key, value in kwargs.items() }
        with multiprocessing.Pool(self.__pool_size) as pool:
            chunks: list[Collection[I]]
            if self.__number_of_chunks is not None and self.__number_of_chunks > 0:
                bulk_number_per_chunk = max(self.__min_chunk_size, len(job_keys) // self.__number_of_chunks)
                n_chunks_with_extra = len(job_keys) - (self.__number_of_chunks * bulk_number_per_chunk)
                chunks = []
                i = 0
                N = len(job_keys)
                while i < N:
                    next_i = i + bulk_number_per_chunk + (1 if i < n_chunks_with_extra else 0)
                    chunks.append(job_keys[i : next_i])
                    i = next_i
            elif self.__min_chunk_size > 1:
                self.__min_chunk_size
                chunks = []
                i = 0
                N = len(job_keys)
                while i < N:
                    next_i = i + self.__min_chunk_size
                    chunks.append(job_keys[i : next_i])
                    i = next_i
            else:
                chunks = [[k] for k in job_keys]
            tasks: list[multiprocessing.pool.ApplyResult] = []
            for chunk in chunks:
                tasks.append(pool.apply_async(_process_target_wrapper, args = (self.__target, self.__results, self.__ignore_return, chunk, *converted_args), kwds = converted_kwargs))
            for t in tasks:
                t.wait()
                _ = t.get() # This will re-raise any errors generated by the pool processes
    def get_results(self) -> tuple[T, ...]:
        ordered_keys = list(self.__results.keys())
        ordered_keys.sort(key = self.__key_order_func)
        return tuple([self.__results[key] for key in ordered_keys])

def _process_target_wrapper(target: Callable[Concatenate[I, P], T], results: dict[I, T], ignore_results: bool, i_list: Collection[I], *args: object, **kwargs: object):
    shared_array_shepherd = SharedArray_Shepherd()
    converted_args = typing_cast(P.args, [(value.load(shared_array_shepherd) if isinstance(value, SharedArray_TransmissionData) else value) for value in args])
    converted_kwargs = typing_cast(P.kwargs, { key : (value.load(shared_array_shepherd) if isinstance(value, SharedArray_TransmissionData) else value) for key, value in kwargs.items() })
    with shared_array_shepherd:
        for i in i_list:
            result = target(i, *converted_args, **converted_kwargs)
            if not ignore_results:
                results[i] = result

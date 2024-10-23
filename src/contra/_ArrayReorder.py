# SPDX-FileCopyrightText: 2024-present Christopher Rowe <chris.rowe19@outlook.com>
#
# SPDX-License-Identifier: None
from typing import cast, Any, Union, Callable
from functools import singledispatchmethod

import numpy as np
from unyt import unyt_quantity, unyt_array
from QuasarCode import Console
from QuasarCode.MPI import MPI_Config, mpi_barrier

#import traceback#TODO: remove



def round_robin_pairings(N: int) -> tuple[tuple[int|None, ...], ...]:
    participants: list[int|None] = list(range(N))
    if N % 2 != 0:
        participants.append(None)
#        N += 1
    rounds: int = len(participants) - 1
    pairings: list[list[int|None]] = [[None] * rounds] * N
    for round in range(rounds):
        left_half:  list[int|None] = participants[:len(participants) // 2 ]
        right_half: list[int|None] = participants[ len(participants) // 2:]
        for i in range(len(left_half)):
            p1, p2 = left_half[i], right_half[-i - 1]
            if p1 is not None:
                pairings[p1][round] = p2
            if p2 is not None:
                pairings[p2][round] = p1
        participants = [participants[0]] + [participants[-1]] + participants[1 : -1]
    return pairings

def round_robin_pairings_self(N: int, n: int) -> tuple[int|None, ...]:
    participants: list[int|None] = list(range(N))
    if N % 2 != 0:
        participants.append(None)
#        N += 1
    rounds: int = len(participants) - 1
    pairings: list[int|None] = [None] * rounds
    for round in range(rounds):
        left_half:  list[int|None] = participants[:len(participants) // 2 ]
        right_half: list[int|None] = participants[ len(participants) // 2:]
        for i in range(len(left_half)):
            p1, p2 = left_half[i], right_half[-i - 1]
            if p1 == n:
                pairings[round] = p2
            elif p2 == n:
                pairings[round] = p1
        participants = [participants[0]] + [participants[-1]] + participants[1 : -1]
    return tuple(pairings)



class ArrayReorder_MPI(object):
    """
    Callable object that allows the order of elements in an
        array to be rearanged to match that of a template array.
    
    DO NOT use the constructor! Instead, use the static method "create" and the "reverse" attribute.
    """

    def __init__(
                    self,
                    send_to_ranks: np.ndarray,
                    recive_from_ranks: np.ndarray,
                    transmitted_data_reorder: "ArrayReorder_2",
                    target_mask_on_rank: np.ndarray,
                    expected_input_length_on_rank: int
                ) -> None:

        self.__recive_rank_elements: np.ndarray = recive_from_ranks
        self.__send_rank_elements: np.ndarray = send_to_ranks
        self.__reorder_transmitted_data: ArrayReorder_2 = transmitted_data_reorder
        self.__target_mask = target_mask_on_rank
        self.__input_length = expected_input_length_on_rank
        self.__output_length = len(target_mask_on_rank)
        
        self.__reverse: "ArrayReorder_MPI"

    def __set_reverse(self, reverse_object: "ArrayReorder_MPI") -> None:
        self.__reverse = reverse_object

    @property
    def reverse(self) -> "ArrayReorder_MPI":
        """
        Reverse operation object.
        """
        return self.__reverse


    @singledispatchmethod
    def __call__(
                    self,
                    source_data: np.ndarray,
                    /,
                    output_array: Union[np.ndarray, None] = None,
                    default_value: Union[Any, None] = None
                ) -> np.ndarray:
        """
        Reorder data.
        """

        if output_array is not None and default_value is not None:
            Console.print_verbose_warning("ArrayReorder: call got both an output array and default value.\nDangerous behaviour as this may overwrite elements!")

        if output_array is None:
            output_array = np.empty(shape = (self.__output_length, *source_data.shape[1:]), dtype = source_data.dtype)
            Console.print_debug(MPI_Config.comm.gather([output_array.shape, output_array.dtype]))

        if default_value is not None:
            output_array[~self.__target_mask] = default_value

        # Find the order in which to communicate with other ranks
        communication_ranks_order = round_robin_pairings_self(MPI_Config.comm_size, MPI_Config.rank)

        def do_self_transfer():
            output_array[self.__recive_rank_elements == MPI_Config.rank] = source_data[self.__send_rank_elements == MPI_Config.rank]

        self_transfer_done = False
        for target_rank in communication_ranks_order:
            if target_rank is None:
                # Do self transfer here to save time
                MPI_Config.comm.barrier()
                do_self_transfer()
                self_transfer_done = True
            else:
                # Send data expected by the target rank
                # Recive data sent by that rank
                send_mask = self.__send_rank_elements == target_rank
                data_to_send = np.empty((send_mask.sum(), *source_data.shape[1:]), dtype = source_data.dtype)
                data_to_send[:] = source_data[send_mask]
                MPI_Config.comm.barrier()
                output_array[self.__recive_rank_elements == target_rank] = MPI_Config.comm.sendrecv(
                    sendobj = data_to_send,
                    dest = target_rank
                )
                del send_mask
                del data_to_send
        if not self_transfer_done:
            do_self_transfer()
        
        # Fix the order of the data that has been disordered due to information loss when passing multiple elements in a single rank pairing
        output_array = self.__reorder_transmitted_data(output_array) # This should have the same input and output length

        MPI_Config.comm.barrier()
        return output_array
    
    @__call__.register(unyt_array)
    def _(self,
                source_data: unyt_array,
                /,
                output_array: Union[unyt_array, None] = None,
                default_value: Union[unyt_quantity, None] = None
         ) -> unyt_array:
        
        input_units = source_data.units
        if output_array is None:
            raw_output_array = None
        else:
            raw_output_array = np.full(output_array.shape, np.nan)
        if default_value is None:
            raw_default_value = None
        else:
            raw_default_value = default_value.to(input_units).value
        raw_result = self.__call__(source_data.value, raw_output_array, raw_default_value)
        if output_array is not None:
            output_array[self.__target_mask] = unyt_array(raw_output_array[self.__target_mask], units = input_units).to(output_array.units)
            if default_value is not None:
                output_array[~self.__target_mask] = unyt_array(raw_output_array[~self.__target_mask], units = input_units).to(output_array.units)
        else:
            output_array = unyt_array(raw_result, input_units)
        return output_array

    @staticmethod
    def create(
                source_order: np.ndarray,
                target_order: np.ndarray,
                source_order_filter: Union[np.ndarray, None] = None,
                target_order_filter: Union[np.ndarray, None] = None
              ) -> "ArrayReorder_MPI":
        """
        Create a new ArrayReorder instance.

        source_order_filter and target_order_filter allow only specific elements
            to be considered for matching without altering the input or output shapes.
        """

        order_dtype = source_order.dtype
        if order_dtype != target_order.dtype:
            raise TypeError(f"Input source and target order arrays have different datatypes ({source_order.dtype} and {target_order.dtype}).\nThese input arrays MUST declare matching datatypes.")

        # Generate the filters if not specified
        if source_order_filter is None:
            source_order_filter = np.full(source_order.shape[0], True, dtype = np.bool_)
        if target_order_filter is None:
            target_order_filter = np.full(target_order.shape[0], True, dtype = np.bool_)

        # Allocate buffers for transmission of filtered data chunks
        filtered_source_length_buffer = np.empty(source_order_filter.sum(), dtype = order_dtype)
        filtered_target_length_buffer = np.empty(target_order_filter.sum(), dtype = order_dtype)

        # Gather information about other ranks (input data) after filtering
        filtered_source_length_buffer[:] = source_order[source_order_filter] # data transmitted with MPI should be contigous in memory
        all_source_ids_chunks: list[np.ndarray]|None = MPI_Config.comm.gather(filtered_source_length_buffer, MPI_Config.root)
        if MPI_Config.is_root:

            # Get chunk size info and concatinate the data into a single array
            all_source_ids_chunks = cast(list[np.ndarray], all_source_ids_chunks)
            input_rank_chunk_sizes = [len(a) for a in all_source_ids_chunks]
            all_source_ids = np.concatenate(all_source_ids_chunks)
#            Console.print_debug(*(line.strip() + "\n" for line in traceback.format_stack()))
            Console.print_debug(all_source_ids.shape, all_source_ids.dtype, np.unique(all_source_ids).shape)
            del all_source_ids_chunks # Remove unnessessary copy (or reference if the combined array is a view)

            # Create an array with the rank associated with each element
            all_source_id_ranks: np.ndarray = np.empty_like(all_source_ids)
            source_rank_chunk_slices: list[slice] = []
            offset = 0
            for rank, chunk_size in enumerate(input_rank_chunk_sizes):
                new_offset = offset + chunk_size
                all_source_id_ranks[offset : new_offset] = rank
                source_rank_chunk_slices.append(slice(offset, new_offset))
                offset = new_offset

        MPI_Config.comm.barrier()

        # Gather information about other ranks (output data) after filtering
        filtered_target_length_buffer[:] = target_order[target_order_filter]
        all_target_ids_chunks: list[np.ndarray]|None = MPI_Config.comm.gather(filtered_target_length_buffer, MPI_Config.root)
        if MPI_Config.is_root:

            # Get chunk size info and concatinate the data into a single array
            all_target_ids_chunks = cast(list[np.ndarray], all_target_ids_chunks)
            result_rank_chunk_sizes = [len(a) for a in all_target_ids_chunks]
            all_target_ids = np.concatenate(all_target_ids_chunks)
            del all_target_ids_chunks # Remove unnessessary copy (or reference if the combined array is a view)

            # Create an array with the rank associated with each element
            all_target_id_ranks: np.ndarray = np.empty_like(all_target_ids)
            target_rank_chunk_slices: list[slice] = []
            offset = 0
            for rank, chunk_size in enumerate(result_rank_chunk_sizes):
                new_offset = offset + chunk_size
                all_target_id_ranks[offset : new_offset] = rank
                target_rank_chunk_slices.append(slice(offset, new_offset))
                offset = new_offset

        MPI_Config.comm.barrier()

        # Which elements of source data should be sent to which rank (assuming input convention - swap for reverse convention)
        send_to_ranks_this_rank = np.full_like(source_order, -1, dtype = order_dtype)
        # Which elements of output data are coming from which rank
        get_from_ranks_this_rank = np.full_like(target_order, -1, dtype = order_dtype)
        # Order of (unfiltered) data after MPI communication but before correction applied for disordered data (input convention)
        unsorted_transmission_result_forwards = np.full_like(target_order, -1, dtype = order_dtype)
        # Order of (unfiltered) data after MPI communication but before correction applied for disordered data (reverse of input convention)
        unsorted_transmission_result_backwards = np.full_like(source_order, -1, dtype = order_dtype)

        # Compute the re-order on the root rank only
        if MPI_Config.is_root:
            # Find the indeces that will produce a sorted array of elements appearing on both (filtered) arrays
            Console.print_debug(all_source_ids.shape, np.unique(all_source_ids).shape)
            Console.print_debug(all_target_ids.shape, np.unique(all_target_ids).shape)
            _, input_common_indices, output_common_indices = np.intersect1d(all_source_ids, all_target_ids, assume_unique = True, return_indices = True)
            undo_sort_input_common_indices = input_common_indices.argsort()
            undo_sort_output_common_indices = output_common_indices.argsort()

            # Compute the orders to swap between one filterd array and the other
            forwards_order = input_common_indices[undo_sort_output_common_indices]
            backwards_order = output_common_indices[undo_sort_input_common_indices]

            # Apply the re-ordering to the rank information
            # This will indicate which output data elements come from a given rank
            # WARNING: these are not strictly in the right order - elements coming from a given rank will be in source order and will need re-ordering!

            # Fill the arrays with -1 to account for data that is included by the masks but is only present in one of the datasets (and therfore not transfered)
            target_order_source_ranks = np.full_like(all_target_ids, -1, dtype = order_dtype)
            source_order_target_ranks = np.full_like(all_source_ids, -1, dtype = order_dtype)

            #TODO: where these are -1, fill with default data!!!!!!!!!!!
            Console.print_debug(all_source_ids.shape, all_source_ids.min() if all_source_ids.shape[0] > 0 else None, all_source_ids.max() if all_source_ids.shape[0] > 0 else None)
            Console.print_debug(input_common_indices.shape, input_common_indices.min() if input_common_indices.shape[0] > 0 else None, input_common_indices.max() if input_common_indices.shape[0] > 0 else None)
            Console.print_debug(forwards_order.shape, forwards_order.min() if forwards_order.shape[0] > 0 else None, forwards_order.max() if forwards_order.shape[0] > 0 else None)
            target_order_source_ranks[output_common_indices[undo_sort_output_common_indices]] = all_source_id_ranks[forwards_order] # Which rank is a given data element coming from
            source_order_target_ranks[input_common_indices[undo_sort_input_common_indices]] = all_target_id_ranks[backwards_order] # Which rank is a given data element going to

            # Simulate the MPI transmission opperation to reproduce the 'wrong' ordered data after transmission
            result_order_without_fix_forwards = np.full_like(all_target_ids, -1, dtype = order_dtype)
            result_order_without_fix_backwards = np.full_like(all_source_ids, -1, dtype = order_dtype)
            for target_rank in range(MPI_Config.comm_size):
                for source_rank in range(MPI_Config.comm_size):

                    # Forward convention
                    result_order_without_fix_forwards[
                        target_rank_chunk_slices[target_rank]
                    ][
                        target_order_source_ranks[target_rank_chunk_slices[target_rank]] == source_rank
                    ] = all_source_ids[
                        source_rank_chunk_slices[source_rank]
                    ][
                        source_order_target_ranks[source_rank_chunk_slices[source_rank]] == target_rank
                    ]

                    # Backward convention
                    result_order_without_fix_backwards[
                        source_rank_chunk_slices[source_rank]
                    ][
                        source_order_target_ranks[source_rank_chunk_slices[source_rank]] == target_rank
                    ] = all_target_ids[
                        target_rank_chunk_slices[target_rank]
                    ][
                        target_order_source_ranks[target_rank_chunk_slices[target_rank]] == source_rank
                    ]

            # Distibute the results
            MPI_Config.comm.Scatterv([source_order_target_ranks,  input_rank_chunk_sizes], filtered_source_length_buffer,  MPI_Config.root) # Source length
            send_to_ranks_this_rank[source_order_filter] = filtered_source_length_buffer[:]
            MPI_Config.comm.Scatterv([target_order_source_ranks, result_rank_chunk_sizes], filtered_target_length_buffer, MPI_Config.root) # Target length
            get_from_ranks_this_rank[target_order_filter] = filtered_target_length_buffer[:]
            MPI_Config.comm.Scatterv([result_order_without_fix_forwards,  result_rank_chunk_sizes], filtered_target_length_buffer, MPI_Config.root) # Target length
            unsorted_transmission_result_forwards[target_order_filter] = filtered_target_length_buffer[:]
            MPI_Config.comm.Scatterv([result_order_without_fix_backwards,  input_rank_chunk_sizes], filtered_source_length_buffer, MPI_Config.root) # Source length
            unsorted_transmission_result_backwards[source_order_filter] = filtered_source_length_buffer[:]

        else:
            # Get the order information from the root rank
            MPI_Config.comm.Scatterv(None, filtered_source_length_buffer,  MPI_Config.root) # Source length
            send_to_ranks_this_rank[source_order_filter] = filtered_source_length_buffer[:]
            MPI_Config.comm.Scatterv(None, filtered_target_length_buffer, MPI_Config.root) # Target length
            get_from_ranks_this_rank[target_order_filter] = filtered_target_length_buffer[:]
            MPI_Config.comm.Scatterv(None, filtered_target_length_buffer, MPI_Config.root) # Target length
            unsorted_transmission_result_forwards[target_order_filter] = filtered_target_length_buffer[:]
            MPI_Config.comm.Scatterv(None, filtered_source_length_buffer, MPI_Config.root) # Source length
            unsorted_transmission_result_backwards[source_order_filter] = filtered_source_length_buffer[:]

        MPI_Config.comm.barrier()

        # Using the simulated result order and the true order, compute how to re-order the data on a single rank
        forward_order_fix = ArrayReorder_2.create(unsorted_transmission_result_forwards, target_order, source_order_filter = target_order_filter, target_order_filter = target_order_filter)#TODO: reduced - set the filter to use input length
        backward_order_fix = ArrayReorder_2.create(unsorted_transmission_result_backwards, source_order, source_order_filter = source_order_filter, target_order_filter = source_order_filter)#TODO: reduced - set the filter to use input length

        forwards_object  = ArrayReorder_MPI(send_to_ranks_this_rank, get_from_ranks_this_rank, forward_order_fix, target_order_filter, source_order.shape[0])
        backwards_object = ArrayReorder_MPI(get_from_ranks_this_rank, send_to_ranks_this_rank, backward_order_fix, source_order_filter, target_order.shape[0])
        forwards_object.__set_reverse(backwards_object)
        backwards_object.__set_reverse(forwards_object)

        MPI_Config.comm.barrier()
        return forwards_object



class ArrayReorder_2(Callable):
    """
    Callable object that allows the order of elements in an
        array to be rearanged to match that of a template array.
    
    DO NOT use the constructor! Instead, use the static method "create" and the "reverse" attribute.
    """

    def __init__(
                    self,
                    source_order: np.ndarray,
                    target_mask: np.ndarray,
                    expected_input_length: int
                ) -> None:
        self.__source_order_indexes = source_order
        self.__target_mask = target_mask
        self.__input_length = expected_input_length
        self.__output_length = len(target_mask)
        
        self.__reverse: "ArrayReorder_2"

    def __set_reverse(self, reverse_object: "ArrayReorder_2") -> None:
        self.__reverse = reverse_object

    @property
    def reverse(self) -> "ArrayReorder_2":
        """
        Reverse operation object.
        """
        return self.__reverse


    @singledispatchmethod
    def __call__(
                    self,
                    source_data: np.ndarray,
                    /,
                    output_array: Union[np.ndarray, None] = None,
                    default_value: Union[Any, None] = None
                ) -> np.ndarray:
        """
        Reorder data.
        """

        if output_array is not None and default_value is not None:
            Console.print_verbose_warning("ArrayReorder: call got both an output array and default value.\nDangerous behaviour as this may overwrite elements!")

        if output_array is None:
            output_array = np.empty(shape = (self.__output_length, *source_data.shape[1:]), dtype = source_data.dtype)

        if default_value is not None:
            output_array[~self.__target_mask] = default_value

        output_array[self.__target_mask] = source_data[self.__source_order_indexes]

        return output_array
    
    @__call__.register(unyt_array)
    def _(self,
                source_data: unyt_array,
                /,
                output_array: Union[unyt_array, None] = None,
                default_value: Union[unyt_quantity, None] = None
         ) -> unyt_array:
        
        input_units = source_data.units
        if output_array is None:
            raw_output_array = None
        else:
            raw_output_array = np.full(output_array.shape, np.nan)
        if default_value is None:
            raw_default_value = None
        else:
            raw_default_value = default_value.to(input_units).value
        raw_result = self.__call__(source_data.value, raw_output_array, raw_default_value)
        if output_array is not None:
            output_array[self.__target_mask] = unyt_array(raw_output_array[self.__target_mask], units = input_units).to(output_array.units)
            if default_value is not None:
                output_array[~self.__target_mask] = unyt_array(raw_output_array[~self.__target_mask], units = input_units).to(output_array.units)
        else:
            output_array = unyt_array(raw_result, input_units)
        return output_array

    @staticmethod
    def create(
                source_order: np.ndarray,
                target_order: np.ndarray,
                source_order_filter: Union[np.ndarray, None] = None,
                target_order_filter: Union[np.ndarray, None] = None
              ) -> "ArrayReorder_2":
        """
        Create a new ArrayReorder instance.

        source_order_filter and target_order_filter allow only specific elements
            to be considered for matching without altering the input or output shapes.
        """

        if source_order_filter is None:
            source_order_filter = np.full(source_order.shape[0], True, dtype = np.bool_)
        if target_order_filter is None:
            target_order_filter = np.full(target_order.shape[0], True, dtype = np.bool_)

        _, input_common_indices, output_common_indices = np.intersect1d(source_order[source_order_filter], target_order[target_order_filter], assume_unique = True, return_indices = True)

        forwards_object  = ArrayReorder_2(np.where(source_order_filter)[0][input_common_indices][output_common_indices.argsort()], target_order_filter, source_order.shape[0])
        backwards_object = ArrayReorder_2(np.where(target_order_filter)[0][output_common_indices][input_common_indices.argsort()], source_order_filter, target_order.shape[0])
        forwards_object.__set_reverse(backwards_object)
        backwards_object.__set_reverse(forwards_object)
        return forwards_object

        x_out[target_order_filter] = x_in[np.where(source_order_filter)[0][a][b.argsort]]
        x_in[source_order_filter] = x_out[np.where(target_order_filter)[0][b][a.argsort]]


        # Get the order nessessary to sort both arrays
        # Arrays need to be sorted in order to allow rearangement
        source_order_sorted = source_order.argsort()
        target_order_sorted = target_order.argsort()

        # Get the order to undo the sort opperations so that the original orders can be recovered for other datasets
        source_order_undo_sorted = source_order_sorted.argsort()
        target_order_undo_sorted = target_order_sorted.argsort()

        # Get the (sorted) lists of valid IDs for matching
        # This allows the input and output shapes to be retained with some elements not being valid for matching
        target_searchable = target_order[target_order_sorted] if target_order_filter is None else target_order[target_order_sorted][target_order_filter[target_order_sorted]]
        source_searchable = source_order[source_order_sorted] if source_order_filter is None else source_order[source_order_sorted][source_order_filter[source_order_sorted]]

        # Identify which elements in each array are a match with the valid elements in that other array
        forwards_membership_filter = np.isin(source_order[source_order_sorted], target_searchable)[source_order_undo_sorted]
        backwards_membership_filter = np.isin(target_order[target_order_sorted], source_searchable)[target_order_undo_sorted]

        # Exclude any matches from self if they are for an invalid element
        if source_order_filter is not None:
            forwards_membership_filter = forwards_membership_filter & source_order_filter
        if target_order_filter is not None:
            backwards_membership_filter = backwards_membership_filter & target_order_filter

        # Re-compute the sorted order for only valid matches
        # These should have the same length!
        source_order_sorted = source_order[forwards_membership_filter].argsort()
        target_order_sorted = target_order[backwards_membership_filter].argsort()

        # Re-compute the reverse of the sorted order for only valid matches
        source_order_undo_sorted = source_order_sorted.argsort()
        target_order_undo_sorted = target_order_sorted.argsort()

        forwards_object  = ArrayReorder(forwards_membership_filter, backwards_membership_filter, source_order_sorted[target_order_undo_sorted])
        backwards_object = ArrayReorder(backwards_membership_filter, forwards_membership_filter, target_order_sorted[source_order_undo_sorted])
        forwards_object.__set_reverse(backwards_object)
        backwards_object.__set_reverse(forwards_object)
        return forwards_object



class ArrayReorder(Callable):
    """
    Callable object that allows the order of elements in an
        array to be rearanged to match that of a template array.
    
    DO NOT use the constructor! Instead, use the static method "create" and the "reverse" attribute.
    """

    def __init__(
                    self,
                    source_filter: np.ndarray,
                    destination_filter: np.ndarray,
                    order_conversion_indexes: np.ndarray
                ) -> None:
        self.__source_filter = source_filter
        self.__destination_filter = destination_filter
        self.__order_conversion_indexes = order_conversion_indexes

        self.__source_filter_length: int = self.__source_filter.shape[0]
        self.__destination_filter_length: int = self.__destination_filter.shape[0]

        self.__n_matched_items: int = self.__source_filter.sum()
        if self.__destination_filter.sum() != self.__n_matched_items:
            raise ValueError(f"Source and destination filters select a different number of items ({self.__n_matched_items} and {self.__destination_filter_length.sum()}).")

        self.__uses_all_inputs = self.__source_filter_length == self.__n_matched_items
        self.__result_is_exact = self.__destination_filter_length == self.__n_matched_items
        self.__is_lossless = self.__uses_all_inputs and self.__result_is_exact
        self.__is_reduction = self.__source_filter_length > self.__n_matched_items
        self.__is_expansion = self.__destination_filter_length > self.__n_matched_items
        self.__creates_subset = self.__is_reduction and self.__result_is_exact
        self.__creates_superset = self.__is_expansion and self.__uses_all_inputs
        
        self.__reverse: "ArrayReorder"

    def __set_reverse(self, reverse_object: "ArrayReorder") -> None:
        self.__reverse = reverse_object

    @property
    def reverse(self) -> "ArrayReorder":
        """
        Reverse operation object.
        """
        return self.__reverse
    
    @property
    def source_filter(self) -> np.ndarray:
        return self.__source_filter
    
    @property
    def target_filter(self) -> np.ndarray:
        return self.__destination_filter

    @property
    def input_length(self) -> int:
        return self.__source_filter_length

    @property
    def output_length(self) -> int:
        return self.__destination_filter_length
    
    def __len__(self) -> int:
        return self.input_length

    @property
    def matched_items(self) -> int:
        return self.__n_matched_items

    @property
    def uses_all_inputs(self) -> bool:
        """
        All inputs are matches.
        """
        return self.__uses_all_inputs

    @property
    def all_outputs_matched(self) -> bool:
        """
        All outputs are matches.
        """
        return self.__result_is_exact

    @property
    def lossless(self) -> bool:
        """
        All outputs are inputs and all inputs are outputs.
        """
        return self.__is_lossless

    @property
    def matches_are_reduction(self) -> bool:
        """
        Fewer matches than the number of inputs.
        """
        return self.__is_reduction

    @property
    def results_are_expansion(self) -> bool:
        """
        More outputs than the number of matches.
        """
        return self.__is_expansion

    @property
    def results_are_subset(self) -> bool:
        """
        Outputs are a subset of the inputs.
        """
        return self.__creates_subset

    @property
    def results_are_superset(self) -> bool:
        """
        Outputs are a superset of the inputs.
        """
        return self.__creates_superset

    @singledispatchmethod
    def __call__(
                    self,
                    source_data: np.ndarray,
                    /,
                    output_array: Union[np.ndarray, None] = None,
                    default_value: Union[Any, None] = None
                ) -> np.ndarray:
        """
        Reorder data.
        """

        if not self.__result_is_exact and output_array is None and default_value is None:
            raise ValueError("More output elements expected than matches but no default value provided and no output target array to write matches to.")

        if output_array is not None and default_value is not None:
            Console.print_verbose_warning("ArrayReorder: call got both an output array and default value.\nDangerous behaviour as this may overwrite elements!")

        if output_array is None:
#            output_array = np.empty(shape = self.__destination_filter_length, dtype = source_data.dtype)
            output_array = np.empty(shape = (self.__destination_filter_length, *source_data.shape[1:]), dtype = source_data.dtype)

        if default_value is not None:
            output_array[~self.__destination_filter] = default_value

        output_array[self.__destination_filter] = source_data[self.__source_filter][self.__order_conversion_indexes]

        return output_array
    
    @__call__.register(unyt_array)
    def _(self,
                source_data: unyt_array,
                /,
                output_array: Union[unyt_array, None] = None,
                default_value: Union[unyt_quantity, None] = None
         ) -> unyt_array:
        
        input_units = source_data.units
        if output_array is None:
            raw_output_array = None
        else:
            raw_output_array = np.full(output_array.shape, np.nan)
        if default_value is None:
            raw_default_value = None
        else:
            raw_default_value = default_value.to(input_units).value
        raw_result = self.__call__(source_data.value, raw_output_array, raw_default_value)
        if output_array is not None:
            output_array[self.__destination_filter] = unyt_array(raw_output_array[self.__destination_filter], units = input_units).to(output_array.units)
            if default_value is not None:
                output_array[~self.__destination_filter] = unyt_array(raw_output_array[~self.__destination_filter], units = input_units).to(output_array.units)
        else:
            output_array = unyt_array(raw_result, input_units)
        return output_array

    @staticmethod
    def create(
                source_order: np.ndarray,
                target_order: np.ndarray,
                source_order_filter: Union[np.ndarray, None] = None,
                target_order_filter: Union[np.ndarray, None] = None
              ) -> "ArrayReorder":
        """
        Create a new ArrayReorder instance.

        source_order_filter and target_order_filter allow only specific elements
            to be considered for matching without altering the input or output shapes.
        """

        # Get the order nessessary to sort both arrays
        # Arrays need to be sorted in order to allow rearangement
        source_order_sorted = source_order.argsort()
        target_order_sorted = target_order.argsort()

        # Get the order to undo the sort opperations so that the original orders can be recovered for other datasets
        source_order_undo_sorted = source_order_sorted.argsort()
        target_order_undo_sorted = target_order_sorted.argsort()

        # Get the (sorted) lists of valid IDs for matching
        # This allows the input and output shapes to be retained with some elements not being valid for matching
        target_searchable = target_order[target_order_sorted] if target_order_filter is None else target_order[target_order_sorted][target_order_filter[target_order_sorted]]
        source_searchable = source_order[source_order_sorted] if source_order_filter is None else source_order[source_order_sorted][source_order_filter[source_order_sorted]]

        # Identify which elements in each array are a match with the valid elements in that other array
        forwards_membership_filter = np.isin(source_order[source_order_sorted], target_searchable)[source_order_undo_sorted]
        backwards_membership_filter = np.isin(target_order[target_order_sorted], source_searchable)[target_order_undo_sorted]

        # Exclude any matches from self if they are for an invalid element
        if source_order_filter is not None:
            forwards_membership_filter = forwards_membership_filter & source_order_filter
        if target_order_filter is not None:
            backwards_membership_filter = backwards_membership_filter & target_order_filter

        # Re-compute the sorted order for only valid matches
        # These should have the same length!
        source_order_sorted = source_order[forwards_membership_filter].argsort()
        target_order_sorted = target_order[backwards_membership_filter].argsort()

        # Re-compute the reverse of the sorted order for only valid matches
        source_order_undo_sorted = source_order_sorted.argsort()
        target_order_undo_sorted = target_order_sorted.argsort()

        forwards_object  = ArrayReorder(forwards_membership_filter, backwards_membership_filter, source_order_sorted[target_order_undo_sorted])
        backwards_object = ArrayReorder(backwards_membership_filter, forwards_membership_filter, target_order_sorted[source_order_undo_sorted])
        forwards_object.__set_reverse(backwards_object)
        backwards_object.__set_reverse(forwards_object)
        return forwards_object



class ArrayMapping(object):
    """
    Callable object that allows a one-way mapping of data elements
        from a source order onto a target order where the source order
        is unique but elements may be duplicated in the target order.
    """

    def __init__(
                    self,
                    source_IDs: np.ndarray,
                    target_IDs: np.ndarray,
                    source_ID_filter: Union[np.ndarray, None] = None,
                    target_ID_filter: Union[np.ndarray, None] = None
                ) -> None:
        
        source_order_sorted = np.argsort(source_IDs)
        target_order_sorted = np.argsort(target_IDs)

        source_order_undo_sorted = source_order_sorted.argsort()
        target_order_undo_sorted = target_order_sorted.argsort()

        target_searchable = target_IDs[target_order_sorted] if target_ID_filter is None else target_IDs[target_order_sorted][target_ID_filter[target_order_sorted]]
        source_searchable = source_IDs[source_order_sorted] if source_ID_filter is None else source_IDs[source_order_sorted][source_ID_filter[source_order_sorted]]

        forwards_membership_filter = np.isin(source_IDs[source_order_sorted], target_searchable)[source_order_undo_sorted]
        backwards_membership_filter = np.isin(target_IDs[target_order_sorted], source_searchable)[target_order_undo_sorted]

        if source_ID_filter is not None:
            forwards_membership_filter = forwards_membership_filter & source_ID_filter
        if target_ID_filter is not None:
            backwards_membership_filter = backwards_membership_filter & target_ID_filter

        if len(np.unique(source_IDs[forwards_membership_filter])) < forwards_membership_filter.sum():
            raise IndexError("Duplicate matched detected in filtered source array. Source ID array must contain unique elements (after optional filter is applied).")

        reduced_source_order_sorted = np.argsort(source_IDs[forwards_membership_filter])
        placement_order = np.searchsorted(source_IDs[forwards_membership_filter], target_IDs[backwards_membership_filter], sorter = reduced_source_order_sorted)

        #                           | <---------------- Select matched elements ---------------------> || apply mapping |
        self.__input_data_reorder = source_order_sorted[forwards_membership_filter[source_order_sorted]][placement_order]

        self.__input_mask = forwards_membership_filter
        self.__input_length = len(forwards_membership_filter)

        self.__output_mask = backwards_membership_filter
        self.__result_is_exact = backwards_membership_filter.sum() == len(backwards_membership_filter)
        self.__output_length = len(backwards_membership_filter)

    @property
    def input_filter(self) -> np.ndarray:
        return self.__input_mask
    
    @property
    def output_filter(self) -> np.ndarray:
        return self.__output_mask

    @property
    def input_length(self) -> int:
        return self.__input_length

    @property
    def output_length(self) -> int:
        return self.__output_length

    @singledispatchmethod
    def __call__(
                    self,
                    source_data: np.ndarray,
                    output_array: Union[np.ndarray, None] = None,
                    default_value: Union[Any, None] = None
                ) -> np.ndarray:
        """
        Reorder data.
        """

        if not self.__result_is_exact and output_array is None and default_value is None:
            raise ValueError("More output elements expected than matches but no default value provided and no output target array to write matches to.")

        if output_array is not None and default_value is not None:
            Console.print_verbose_warning("ArrayMapping: call got both an output array and default value.\nDangerous behaviour as this may overwrite elements!")

        if output_array is None:
            output_array = np.empty(shape = (self.__output_length, *source_data.shape[1:]), dtype = source_data.dtype)

        if default_value is not None:
            output_array[~self.__output_mask] = default_value

        output_array[self.__output_mask] = source_data[self.__input_data_reorder]

        return output_array
    
    @__call__.register(unyt_array)
    def _(self,
                source_data: unyt_array,
                /,
                output_array: Union[unyt_array, None] = None,
                default_value: Union[unyt_quantity, None] = None
         ) -> unyt_array:
        
        input_units = source_data.units
        if output_array is None:
            raw_output_array = None
        else:
            raw_output_array = np.full(output_array.shape, np.nan)
        if default_value is None:
            raw_default_value = None
        else:
            raw_default_value = default_value.to(input_units).value
        raw_result = self.__call__(source_data.value, raw_output_array, raw_default_value)
        if output_array is not None:
            output_array[self.__output_mask] = unyt_array(raw_output_array[self.__output_mask], units = input_units).to(output_array.units)
            if default_value is not None:
                output_array[~self.__output_mask] = unyt_array(raw_output_array[~self.__output_mask], units = input_units).to(output_array.units)
        else:
            output_array = unyt_array(raw_result, input_units)
        return output_array

    @staticmethod
    def create(
                source_IDs: np.ndarray,
                target_IDs: np.ndarray,
                source_ID_filter: Union[np.ndarray, None] = None,
                target_ID_filter: Union[np.ndarray, None] = None
              ) -> "ArrayMapping":
        return ArrayMapping(
            source_IDs = source_IDs,
            target_IDs = target_IDs,
            source_ID_filter = source_ID_filter,
            target_ID_filter = target_ID_filter
        )

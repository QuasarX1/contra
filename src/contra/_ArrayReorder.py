# SPDX-FileCopyrightText: 2024-present Christopher Rowe <chris.rowe19@outlook.com>
#
# SPDX-License-Identifier: None
from typing import Any, Union, Callable
from functools import singledispatchmethod

import numpy as np
from unyt import unyt_quantity, unyt_array
from QuasarCode import Console

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
            output_array = np.empty(shape = self.__destination_filter_length, dtype = source_data.dtype)

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
            output_array = np.empty(shape = self.__output_length)

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

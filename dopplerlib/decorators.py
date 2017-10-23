#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2016 - Piotr Skonieczka
#

import numpy
import multiprocessing.pool as pool


def check_the_bounds_of_arguments(bounds, offset=1, out_of_bound_value=1e+10):
    """
    Function (decorator) for arguments validation.
        :param bounds: list of bounds for all parameters [(0, 1), (-10, None), (None, None), ...]
        :param offset: skip bounds for first N arguments
        :param out_of_bound_value: return this value when one or more limits have not been met.
        :return: wrapped function's result
    """
    def check_the_single_bound(value, bound):
        min_value, max_value = bound
        return (min_value is None or min_value <= value) and (max_value is None or max_value >= value)

    def check_the_bounds_wrapper(function):
        def function_wrapper(*arguments):

            if all(map(check_the_single_bound, arguments[offset:], bounds)):
                return function(*arguments)
            else:
                return numpy.array([out_of_bound_value, ] * len(arguments[0]))

        return function_wrapper

    return check_the_bounds_wrapper


def cache_the_results(function):
    """
    Function (decorator) for memoization of the results.
        :param function: function to be memoized
        :return: wrapped function's result
    """
    function.cache = dict()
    function.calls = 0
    function.jumps = 0

    def cached_function_wrapper(*arguments):
        function.calls += 1

        if arguments in function.cache and arguments:
            function.jumps += 1
            return function.cache[arguments]
        else:
            calculated_result = function(*arguments)
            function.cache[arguments] = calculated_result

            return calculated_result

    return cached_function_wrapper


def parallel_execution(function):
    pool_size = pool.cpu_count() * 2
    jobs_pool = pool.ThreadPool(pool_size)

    def parallel_execution_wrapper(x_array, *args):
        # Split data into the parts
        data_parts = numpy.array_split(x_array, pool_size)
        # Execute the function and combine all blocks together as a one numpy.array
        return numpy.concatenate(jobs_pool.map(lambda part_of_data: function(part_of_data, *args), data_parts))

    return parallel_execution_wrapper

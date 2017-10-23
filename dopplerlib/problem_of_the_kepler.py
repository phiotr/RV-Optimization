#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2016 - Piotr Skonieczka
#

import collections
import numpy
from decorators import check_the_bounds_of_arguments
# from decorators import parallel_execution
# from decorators import cache_the_results


# @cache_the_results
def kepler_equation_solver(mean_anomaly, orbital_eccentricity, epsilon=1.0e-10):
    """
    Solves the Kepler equation [M = E - e * sin(E)] relative to E parameter
        :param mean_anomaly (M)
        :param orbital_eccentricity (e)
        :param epsilon: precision of calculations
        :return eccentric_anomaly value (E)
    """
    # Scale the anomaly to the range of [0, 2pi)
    mean_anomaly = numpy.mod(mean_anomaly, 2 * numpy.pi)
    eccentric_anomaly = mean_anomaly

    def kepler_function(anomaly):
        return mean_anomaly - anomaly + orbital_eccentricity * numpy.sin(anomaly)

    def kepler_function_derivative(anomaly):
        return orbital_eccentricity * numpy.cos(anomaly) - 1.0

    while numpy.abs(kepler_function(eccentric_anomaly)) >= epsilon:
        eccentric_anomaly += -kepler_function(eccentric_anomaly) / kepler_function_derivative(eccentric_anomaly)

    return eccentric_anomaly


def multi_kepler_equation_solver(mean_anomaly_list, orbital_eccentricity, epsilon=1.0e-10):
    """
    Solves the multi-Keplers equation for the given list of mean anomalies
        :param mean_anomaly_list
        :param orbital_eccentricity
        :param epsilon: precision of calculations
        :return: the list of Kepler's equation solutions for each mean anomaly
    """
    return numpy.vectorize(pyfunc=kepler_equation_solver)(mean_anomaly_list, orbital_eccentricity, epsilon)


# @parallel_execution
@check_the_bounds_of_arguments(offset=1, bounds=[(0, None), (0, None), (0, 2 * numpy.pi), (0, None), (0, 1)])
def doppler_solver(time, time_of_perihelion_passage, half_amplitude_of_the_signal, longitude_of_the_perihelion,
                   orbital_period, eccentricity):
    """
    Calculation of the radial velocity using Doppler's method
        :param time (t)
        :param time_of_perihelion_passage (tau)
        :param half_amplitude_of_the_signal (K)
        :param longitude_of_the_perihelion (omega) in radians [0 - 2*PI]
        :param orbital_period (P)
        :param eccentricity (ecc)
        :return: Radial velocity value for a given planetary system
    """
    # Convertion of the orbit length to the radian angle
    orbital_period = 2 * numpy.pi / orbital_period

    # Time related mean anomaly of the orbit
    mean_anomaly = numpy.mod(orbital_period * (time - time_of_perihelion_passage), 2 * numpy.pi)

    # Resolve the eccentricity anomaly from the Kepler equation
    if isinstance(mean_anomaly, collections.Iterable):
        eccentric_anomaly = multi_kepler_equation_solver(mean_anomaly, eccentricity)
    else:
        eccentric_anomaly = kepler_equation_solver(mean_anomaly, eccentricity)

    # So-called "true anomaly" v(t)
    true_anomaly = 2.0 * numpy.arctan(
        numpy.sqrt((1.0 + eccentricity) / (1.0 - eccentricity)) * numpy.tan(eccentric_anomaly / 2.0)
    )

    # Radial velocity: V(t) = K * (cos(omega + v(t)) + ecc * cos(omega))
    radial_velocity = half_amplitude_of_the_signal * (numpy.cos(longitude_of_the_perihelion + true_anomaly) +
                                                      eccentricity * numpy.cos(longitude_of_the_perihelion))

    return radial_velocity


@check_the_bounds_of_arguments(bounds=[(0, None), (0, None), (0, None)])
def doppler_solver_for_circular_orbit(time, time_of_perihelion_passage, half_amplitude_of_the_signal, orbital_period):
    """
    Calculating radial velocity by Doppler's method in case of perfectly circular orbit
        :param time (t)
        :param time_of_perihelion_passage (tau)
        :param half_amplitude_of_the_signal (K)
        :param orbital_period (P) [0 - 2*PI]
        :return: Radial velocity value for a given planetary system
    """
    # Time related mean anomaly of the orbit
    mean_anomaly = numpy.mod(orbital_period * (time - time_of_perihelion_passage), 2 * numpy.pi)

    # Resolve the eccentricity anomaly from the Kepler equation
    if isinstance(mean_anomaly, collections.Iterable):
        eccentric_anomaly = multi_kepler_equation_solver(mean_anomaly, 0.00)
    else:
        eccentric_anomaly = kepler_equation_solver(mean_anomaly, 0.00)

    # So-called "true anomaly" v(t)
    true_anomaly = 2.0 * numpy.arctan(numpy.tan(eccentric_anomaly / 2.0))

    # Radial velocity: V(t) = K * cos(v(t))
    radial_velocity = half_amplitude_of_the_signal * numpy.cos(true_anomaly)
    return radial_velocity


def doppler_solver_for_n_planets(number_of_planets, time, orbital_parameters):
    """
    Calculation for the radial velocity as a submission of N Doppler signals
        :param number_of_planets: (int)
        :param time: (float or numpy.array)
        :param orbital_parameters: (list) list of 5 x number_of_planets parameters
        :return: (float or numpy.array)
    """
    return numpy.sum(doppler_solver(time, *orbital_parameters[i: i+5]) for i in xrange(0, 5 * number_of_planets, 5))


# def parallel_doppler():
#     import multiprocess
#
#     jobs_count = multiprocess.cpu_count()
#     jobs_pool = multiprocess.Pool(jobs_count)
#
#     def parallel_doppler_wrapper(number_of_planets, time, orbital_parameters):
#         return sum(jobs_pool.map(lambda params: doppler_solver(time, *params),
#                              [orbital_parameters[i: i+5] for i in xrange(0, 5 * number_of_planets, 5)]))
#
#     return parallel_doppler_wrapper
#
#
# doppler_solver_for_n_planets = parallel_doppler()

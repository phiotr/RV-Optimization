#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2016 - Piotr Skonieczka
#

import pickle
import numpy

from problem_of_the_kepler import doppler_solver_for_n_planets
from scipy.optimize import curve_fit
from scipy.optimize import differential_evolution


class DopplerOptimizationModel:
    # Metadata
    observations_file = None
    stellar_jitter = None

    def __init__(self, number_of_planets, number_of_telescopes, orbital_parameters=None, parameters_uncertainties=None):
        self.number_of_planets = number_of_planets
        self.number_of_telescopes = number_of_telescopes
        self.orbital_parameters = list(orbital_parameters)
        self.parameters_uncertainties = parameters_uncertainties
        self.parameters_count = len(self.orbital_parameters)

    def __str__(self):
        return "Model of {n} planets and {t} telescopes.".format(n=self.number_of_planets, t=self.number_of_telescopes)

    def add_metadata(self, observations_file, stellar_jitter):
        self.observations_file = observations_file
        self.stellar_jitter = stellar_jitter

    def get_orbital_parameters(self, planet_number):
        return self.orbital_parameters[planet_number * 5: (planet_number + 1) * 5]

    def get_orbital_uncertainties(self, planet_number):
        return self.parameters_uncertainties[planet_number * 5: (planet_number + 1) * 5]

    def get_offsets(self):
        return self.orbital_parameters[-self.number_of_telescopes:]

    def get_offsets_uncersainties(self):
        return self.parameters_uncertainties[-self.number_of_telescopes:]

    def split(self):
        """
        Split the modeled signal by the indywidual planets
            :return: list of DopplerOptimizationModel instances
        """
        # Common parameters
        offsets = list(self.orbital_parameters[-self.number_of_telescopes:])

        return [DopplerOptimizationModel(1, self.number_of_telescopes, self.get_orbital_parameters(planet_id) + offsets)
                for planet_id in xrange(self.number_of_planets)]

    def quality_of_the_fit(self, observations):
        """
        Get chi^2 measurement of the model's fit to the observations
            :param observations: ObservationsData instance
            :return:
        """
        # Calculate residues
        residues = self.residues_array(observations)
        # Calculate and
        chi2_sum = sum((residues / observations.uncertainties) ** 2)
        return chi2_sum / (observations.count - self.parameters_count - 1)

    def radial_velocities(self, time_range):
        return doppler_solver_for_n_planets(self.number_of_planets, time_range, self.orbital_parameters)

    def offsets_of_instruments(self, list_of_telescopes):
        offsets_list = self.orbital_parameters[-self.number_of_telescopes:]
        return numpy.array(map(lambda telescope_id: offsets_list[telescope_id], list_of_telescopes))

    def residues_array(self, observations):
        observed_values = observations.radial_velocities
        modeled_values = self.radial_velocities(observations.julian_times)
        offset_values = self.offsets_of_instruments(observations.ids_of_telescopes)

        return modeled_values - observed_values + offset_values

    @staticmethod
    def save_model(output_file, model):
        pickle.dump(model, open(output_file, "w"))

    @staticmethod
    def read_model(input_file):
        return pickle.load(file=open(input_file, "r"))


def optimization_ojective_function_builder(number_of_planets, number_of_telescopes, observations, mode="rv"):

    def _multiplanetary_doppler_solver(julian_times, *orbital_parameters):
        """
        Target function for the curve_fit optimalization.
            :param julian_times: (numpy.array)
            :param orbital_parameters: (list) orbital paramters of the model
            :return: (list) radial velocities of the model
        """
        model = DopplerOptimizationModel(number_of_planets, number_of_telescopes, orbital_parameters)
        return model.radial_velocities(julian_times) + model.offsets_of_instruments(observations.ids_of_telescopes)

    def _multiplanetary_doppler_measurement(orbital_parameters):
        """
        Target function for the differential_evolution optimalization.
            :param orbital_parameters: (list) number_of_planets * 5 elements
            :return: (float) chi^2 measurement
        """
        model = DopplerOptimizationModel(number_of_planets, number_of_telescopes, orbital_parameters)
        return model.quality_of_the_fit(observations)

    return {"rv": _multiplanetary_doppler_solver, "chi2": _multiplanetary_doppler_measurement}.get(mode)


def evolutional_optimization(number_of_planets, number_of_telescopes, observations, population_size):
    # Appoint the parameters ranges
    doppler_parameters_bouns = [
        (min(observations.julian_times), max(observations.julian_times)),  # time of perihelion passage
        (0, max(observations.radial_velocities) - min(observations.radial_velocities)),  # half amplitude of the signal
        (0, 2 * numpy.pi),  # longitude of the perihelion
        (1, 2000),  # orbital period in days
        (0.001, 0.999),  # eccentricity
    ] * number_of_planets

    telescope_offsets_bounds = [
        (-25, 25),
    ] * number_of_telescopes

    bounds_of_all_free_paramters = doppler_parameters_bouns + telescope_offsets_bounds

    # Build the objective function for the curve_fit algorithm
    doppler_function = \
        optimization_ojective_function_builder(number_of_planets, number_of_telescopes, observations, mode="chi2")

    # Start the differential_evolution optimization and get calculated paramters
    optimal_orbital_paramters = \
        differential_evolution(func=doppler_function, bounds=bounds_of_all_free_paramters, popsize=population_size).x

    # Return model
    return DopplerOptimizationModel(number_of_planets, number_of_telescopes, optimal_orbital_paramters)


def gradient_optimization(number_of_planets, number_of_telescopes, observations, initial_parameters):
    # Build the objective function for the curve_fit algorithm
    doppler_function = \
        optimization_ojective_function_builder(number_of_planets, number_of_telescopes, observations, mode="rv")

    # Start the curve_fit optimization and get the calculated paramters
    optimal_paramters, covariation_array = \
        curve_fit(doppler_function, observations.julian_times, observations.radial_velocities,
                  sigma=observations.uncertainties, p0=initial_parameters, maxfev=500*len(initial_parameters))

    paramters_uncertainties = numpy.diagonal(covariation_array) ** 2

    return DopplerOptimizationModel(number_of_planets, number_of_telescopes, optimal_paramters, paramters_uncertainties)

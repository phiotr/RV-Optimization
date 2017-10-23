#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2016 - Piotr Skonieczka
#

import numpy


class ObservationsData:
    def __init__(self, julian_times, radial_velocities, uncertainties, ids_of_telescopes):
        self.julian_times = julian_times
        self.radial_velocities = radial_velocities
        self.uncertainties = uncertainties
        self.ids_of_telescopes = ids_of_telescopes
        # Count the observations
        self.count = len(self.julian_times)
        # Count the instruments
        self.telescopes_count = len(set(self.ids_of_telescopes))

    def reduce_offsets(self, offsets_array):
        self.radial_velocities -= offsets_array

    @staticmethod
    def load_data(file_name, sort_observations=True, scale_to_the_mean=False, convert_kmps_to_mps=False,
                  scale_to_the_jitter=True, jitter_value=0.0, convert_indexes_to_integers=True, **other_options):
        """
        Load observations of the radial velocities
            :param file_name: (str)
            :param sort_observations: (boolean) sort observations or not
            :param scale_to_the_mean: (boolean) scale radia velocity values to the mean or not
            :param convert_kmps_to_mps: (boolean) convert radial velocities units from km/s to m/s
            :param scale_to_the_jitter: (boolean) apply jitter correction for the uncersainties
            :param jitter_value: (float)
            :param convert_indexes_to_integers: (boolean) convert column "instruments/telescopes" to integers
            :return: (tuple) times, rv values, uncertainties and ids of telescopes
        """
        # Get data
        observation_data = numpy.loadtxt(file_name, **other_options)
        assert len(observation_data) > 0, "The file does not contains any data"

        # Sort data if needed
        if sort_observations:
            observation_data = numpy.array(sorted(observation_data, key=lambda row: row[0]))

        # Split array by columns
        julian_times, radial_velocities, uncertainties, ids_of_telescopes = \
            (observation_data[:, column_id] for column_id in xrange(4))

        # Do some other calculations...
        if scale_to_the_mean:
            radial_velocities -= numpy.mean(radial_velocities)

        if convert_kmps_to_mps:
            radial_velocities *= 1000

        if scale_to_the_jitter:
            uncertainties = numpy.sqrt(numpy.square(uncertainties) + jitter_value**2)

        if convert_indexes_to_integers:
            ids_of_telescopes = numpy.vectorize(int)(ids_of_telescopes)

        # Return ObservationsData instance
        return ObservationsData(julian_times, radial_velocities, uncertainties, ids_of_telescopes)

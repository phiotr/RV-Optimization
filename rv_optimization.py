#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2016 - Piotr Skonieczka
#

import argparse
from datetime import datetime
from dopplerlib import optimization
from dopplerlib import observations_data


def parse_command_line_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--observations", type=str, required=True, dest="observation_data_file")
    parser.add_argument("--planets", type=int, required=True, dest="number_of_planets")
    parser.add_argument("--jitter", type=float, required=False, dest="stellar_jitter", default=0.0)
    parser.add_argument("--population-size", type=int, required=False, dest="population_size", default=3)

    return parser.parse_args()


def print_computation_summary(start_time, stop_time, de_quality, lm_quality, output_file):
    time_format = "  %-12s: %s [H:M:S.ms]"

    print time_format % ("Start time", start_time.time())
    print time_format % ("Stop time", stop_time.time())
    print time_format % ("Duration", (stop_time - start_time))
    print
    print "  DE model quality:", de_quality
    print "  LM model quality:", lm_quality
    print
    print "  Resulting file  :", output_file


def get_output_file_name(input_file_name, stellar_jitter, planets, quality):
    file_name = input_file_name.split("/")[-1].split(".")[0]
    file_prefix = "models/"
    file_suffix = "-j%.1f-p%i-q%.2f.model" % (stellar_jitter, planets, quality)

    return file_prefix + file_name + file_suffix


def main(args):
    print "* Reading observations..."
    observations = \
        observations_data.ObservationsData.load_data(args.observation_data_file, jitter_value=args.stellar_jitter)

    # Constant values
    number_of_planets = args.number_of_planets
    number_of_telescopes = observations.telescopes_count
    population_size = args.population_size

    print "* Running evolutional optimization..."
    computation_start_time = datetime.now()
    de_model = \
        optimization.evolutional_optimization(number_of_planets, number_of_telescopes, observations, population_size)

    print "* Running gradient optimization..."
    lm_model = \
        optimization.gradient_optimization(number_of_planets, number_of_telescopes, observations,
                                           de_model.orbital_parameters)
    computation_end_time = datetime.now()

    print "* Calculating quality of the models..."
    de_quality = de_model.quality_of_the_fit(observations)
    lm_quality = lm_model.quality_of_the_fit(observations)

    print "* Saving model to the file..."
    lm_model.add_metadata(args.observation_data_file, args.stellar_jitter)
    output_file = get_output_file_name(args.observation_data_file, args.stellar_jitter, number_of_planets, lm_quality)
    optimization.DopplerOptimizationModel.save_model(output_file, lm_model)

    print "* Calculation summary:"
    print_computation_summary(computation_start_time, computation_end_time, de_quality, lm_quality, output_file)


if __name__ == "__main__":
    main(args=parse_command_line_arguments())

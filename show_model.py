#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2016 - Piotr Skonieczka
#

import argparse
from dopplerlib import optimization


def parse_command_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, dest="model_file")

    return parser.parse_args()


def array2list(array):
    return map(float, array)


def main(args):
    model = optimization.DopplerOptimizationModel.read_model(args.model_file)

    print "* Configuration of the model"
    print
    print "  Observations:", model.observations_file
    print "  Jitter value:", model.stellar_jitter
    print "  N.Planets   :", model.number_of_planets
    print "  N.Telescopes:", model.number_of_telescopes
    print
    print "* Orbital parameters"

    doppler_paramters_names = ["tau", "K", "omega", "P", "ecc"]

    for planet_id in xrange(model.number_of_planets):
        print " ---------- P%i ---------- " % planet_id

        for info in zip(doppler_paramters_names, model.get_orbital_parameters(planet_id),
                        model.get_orbital_uncertainties(planet_id)):
            print "%6s: %20.8f  (+/- %.8f)" % info

    print
    print "* Offsets of the telescopes"
    print

    for telescope_id, (offset_value, offset_uncersainty) in \
            enumerate(zip(model.get_offsets(), model.get_offsets_uncersainties())):
        print "  T%i: %20.8f  (+/- %.8f)" % (telescope_id, offset_value, offset_uncersainty)


if __name__ == '__main__':
    main(args=parse_command_line_arguments())

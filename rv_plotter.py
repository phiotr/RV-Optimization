#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2016 - Piotr Skonieczka
#

import argparse
import numpy
import pylab
import itertools

from dopplerlib import optimization
from dopplerlib import observations_data


def parse_command_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, dest="model_file")
    parser.add_argument("--show", required=False, action='store_true', dest="show_diagram")
    parser.set_defaults(show_diagram=False)

    return parser.parse_args()


def roundup(value, scale):
    return numpy.ceil(value / scale) * scale


def rounddown(value, scale):
    return numpy.floor(value / scale) * scale


def setup_plot_layout(plot, title, xlabel, ylabel):
    plot.set_title(title)
    plot.set_xlabel(xlabel)
    plot.set_ylabel(ylabel)
    plot.grid(True, color="0.5", zorder=0)
    plot.axhline(0, color="0.2", linewidth=1, zorder=5)


def plot_line(plot, xvalues, yvalues, linestyle="-", linecolor="red", **kwargs):
    plot.plot(xvalues, yvalues, marker="", linestyle=linestyle, color=linecolor, zorder=10, **kwargs)


def plot_dots(plot, xvalues, yvalues, errors, **kwargs):
    plot.errorbar(xvalues, yvalues, yerr=errors,
                  marker="o", markersize=3, linestyle="None", color="black",
                  markeredgecolor="black", zorder=20, **kwargs)


def get_diagram_output_file_name(model_file):
    file_name = '.'.join(model_file.split("/")[-1].split(".")[:-1])
    file_prefix = "diagrams/"
    file_suffix = ".svg"

    return file_prefix + file_name + file_suffix


def save_diagram(file_name, resolution=(12, 10)):
    pylab.gcf().set_size_inches(*resolution)
    try:
        pylab.savefig(file_name, dpi=90)
    except Exception as e:
        print "Error: Cannot save the figure." + e.message
        print e.args


def main(args):
    print "* Loading model..."
    doppler_model = \
        optimization.DopplerOptimizationModel.read_model(args.model_file)

    print "* Loading observations..."
    observations = \
        observations_data.ObservationsData.load_data(doppler_model.observations_file,
                                                     jitter_value=doppler_model.stellar_jitter)

    # Observations time range
    min_limit, max_limit = rounddown(min(observations.julian_times), 100), roundup(max(observations.julian_times), 100)
    time_range = numpy.linspace(start=min_limit, stop=max_limit, num=(int(max_limit) - int(min_limit)) / 5)

    # Remove offset of the telescope
    observations.reduce_offsets(doppler_model.offsets_of_instruments(observations.ids_of_telescopes))

    print "* Drowing diagrams..."
    # Plots
    plot_rv = pylab.subplot2grid((3, 1), (0, 0), rowspan=2)
    plot_rs = pylab.subplot2grid((3, 1), (2, 0), sharex=plot_rv)

    # Configuration of the first plot
    setup_plot_layout(plot_rv, title="Doppler curve fitting", xlabel="Epoch [JD]", ylabel="Radial velocity [m/s]")

    # Plotting data
    plot_line(plot_rv, time_range, doppler_model.radial_velocities(time_range), linewidth=2)
    plot_dots(plot_rv, observations.julian_times, observations.radial_velocities, observations.uncertainties)

    # Add planets!
    for sub_model, linestyle in zip(doppler_model.split(), itertools.cycle(["--", "-.", "-", ":"])):
        color = "0.6"
        plot_line(plot_rv, time_range, sub_model.radial_velocities(time_range), linecolor=color, linestyle=linestyle)

    # Configuration of the second plot
    setup_plot_layout(plot_rs, title="", xlabel="Epoch [JD]", ylabel="Residual [m/s]")
    # Plot the data
    residues = doppler_model.residues_array(observations)

    plot_dots(plot_rs, observations.julian_times, residues, observations.uncertainties)

    # Set the limits of the figure and do the drow!
    pylab.xlim((min_limit, max_limit))
    pylab.tight_layout()

    # Save it to the file
    save_diagram(get_diagram_output_file_name(args.model_file))

    # Show diagram if needed
    if args.show_diagram:
        pylab.show()


if __name__ == "__main__":
    main(args=parse_command_line_arguments())

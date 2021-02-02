#!/usr/bin/env python
# -*- coding: utf8 -*

import argparse
from sbo.SurrogateBasedOptimization import SurrogateBasedOptimization


def main(optimizationId, nodes=1):
    """
    Plot the results for the SBO run
    @author: Markus Pfeil
    """
    assert type(optimizationId) is int and 0 <= optimizationId

    plots = ['Costfunction', 'StepSizeNorm', 'ParameterConvergence', 'AnnualCycle', 'AnnualCycleParameter', 'Surface', 'SurfaceParameter', 'SurfaceLowFidelityModel']

    sbo = SurrogateBasedOptimization(optimizationId, nodes=nodes)
    sbo.plot(plots=[plots[7]])


if __name__ == '__main__':
    #Command line parameter
    parser = argparse.ArgumentParser()
    parser.add_argument("optimizationId", type=int, help="Id of the optimization")
    parser.add_argument('-nodes', nargs='?', type=int, const=1, default=1, help='Number of nodes for the job on the Nesh-Cluster')
    args = parser.parse_args()

    main(optimizationId=args.optimizationId, nodes=args.nodes)


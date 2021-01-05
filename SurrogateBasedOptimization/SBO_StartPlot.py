#!/usr/bin/env python
# -*- coding: utf8 -*

from sbo.SurrogateBasedOptimization import SurrogateBasedOptimization


def main(optimizationId=207):
    """
    Plot the results for the SBO run
    @author: Markus Pfeil
    """
    assert type(optimizationId) is int and 0 <= optimizationId

    plots = ['Costfunction', 'StepSizeNorm', 'ParameterConvergence', 'AnnualCycle', 'AnnualCycleParameter', 'Surface', 'SurfaceParameter', 'SurfaceLowFidelityModel']

    sbo = SurrogateBasedOptimization(optimizationId, queue='clexpress', cores=1)
    sbo.plot(plots=[plots[7]])


if __name__ == '__main__':
    main()


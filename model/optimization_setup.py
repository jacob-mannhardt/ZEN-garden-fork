"""===========================================================================================================================================================================
Title:        ZEN-GARDEN
Created:      October-2021
Authors:      Alissa Ganter (aganter@ethz.ch), Jacob Mannhardt (jmannhardt@ethz.ch)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:  Class defining the Concrete optimization model.
              The class takes as inputs the properties of the optimization problem. The properties are saved in the
              dictionaries analysis and system which are passed to the class. After initializing the Concrete model, the
              class adds carriers and technologies to the Concrete model and returns the Concrete optimization model.
              The class also includes a method to solve the optimization problem.
==========================================================================================================================================================================="""
import logging
import pyomo.environ as pe
import os
import sys
# import elements of the optimization problem
# technology and carrier classes from technology and carrier directory, respectively
from model.objects.element                              import Element
from model.objects.technology                           import *
from model.objects.carrier                              import *
from model.objects.energy_system                        import EnergySystem
from preprocess.functions.time_series_aggregation       import TimeSeriesAggregation

class OptimizationSetup():

    def __init__(self, analysis, prepare):
        """setup Pyomo Concrete Model
        :param analysis: dictionary defining the analysis framework
        :param system: dictionary defining the system
        :param paths: dictionary defining the paths of the model
        :param solver: dictionary defining the solver"""
        self.prepare    = prepare
        self.analysis   = analysis
        self.system     = prepare.system
        self.paths      = prepare.paths
        self.solver     = prepare.solver
        self.model      = pe.ConcreteModel()
        # set optimization attributes (the five set above) to class <EnergySystem>
        EnergySystem.setOptimizationAttributes(analysis, self.system, self.paths, self.solver, self.model)
        # add Elements to optimization
        self.addElements()

        # define and construct components of self.model
        Element.constructModelComponents()
        logging.info("Apply Big-M GDP ")

        # add transformation factory so that disjuncts are solved
        pe.TransformationFactory("gdp.bigm").apply_to(self.model)

    def addElements(self):
        """This method sets up the parameters, variables and constraints of the carriers of the optimization problem.
        :param analysis: dictionary defining the analysis framework
        :param system: dictionary defining the system"""
        logging.info("\n--- Add elements to model--- \n")
        # add energy system to define system
        EnergySystem("energySystem")

        for technologyType in technologyList:
            technologyClass = getattr(sys.modules[__name__], technologyType)
            technologySet   = technologyClass.label
            setTechnologies = self.system[technologySet]
            # check if technologySet has a subset and remove subset from setTechnologies
            if technologySet in self.analysis["subsets"].keys():
                subsetTechnologies = []
                for subset in self.analysis["subsets"][technologySet]:
                        subsetTechnologies += [tech for tech in self.system[subset]]
                setTechnologies = list(set(setTechnologies)-set(subsetTechnologies))
            # add technology classes
            for tech in setTechnologies:
                technologyClass(tech)

        # get set of carriers
        self.system["setCarriers"] = EnergySystem.getAttribute("setCarriers")
        self.prepare.checkExistingCarrierData(self.system)
        for carrierType in carrierList:
            carrierClass = getattr(sys.modules[__name__], carrierType)
            carrierSet   = carrierClass.label
            setCarriers  = self.system[carrierSet]
            # check if carrierSet has a subset and remove subset from setCarriers
            assert (carrierSet not in self.analysis["subsets"].keys()), f"Functionality of adding carrier-subsets are not implemented yet."
            # add carrier classes
            for carrier in setCarriers:
                carrierClass(carrier)

        # conduct  time series aggregation
        TimeSeriesAggregation.conductTimeSeriesAggregation()

    def solve(self, solver):
        """Create model instance by assigning parameter values and instantiating the sets
        :param solver: dictionary containing the solver settings """

        solverName          = solver["name"]
        solverOptions       = solver["solverOptions"]

        logging.info(f"\n--- Solve model instance using {solverName} ---\n")
        # disable logger temporarily
        logging.disable(logging.WARNING)
        # write an ILP file to print the IIS if infeasible
        #         # (gives Warning: unable to write requested result file ".//outputs//logs//model.ilp" if feasible)
        solver_parameters   = f"ResultFile={os.path.dirname(solver['solverOptions']['logfile'])}//infeasibleModelIIS.ilp"
        self.opt            = pe.SolverFactory(solverName, options=solverOptions)
        self.opt.set_instance(self.model,symbolic_solver_labels=True)
        self.results        = self.opt.solve(tee=solver["verbosity"], logfile=solver["solverOptions"]["logfile"],options_string=solver_parameters)
        # enable logger 
        logging.disable(logging.NOTSET)
        self.model.solutions.load_from(self.results)
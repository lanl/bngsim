////////////////////////////////////////////////////////////////////////////////
//
//    NFsim: The Network Free Stochastic Simulator
//    A software platform for efficient simulation of biochemical reaction
//    systems with a large or infinite state space.
//
//    Copyright (C) 2009,2010,2011,2012
//    Michael W. Sneddon, James R. Faeder, Thierry Emonet
//
//    Licensed under MIT License (see LICENSE.txt).
//    Vendored and modified for bngsim library embedding.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef NFSIM_HH_
#define NFSIM_HH_

// Include "mpi.h" in Scheduler.h first
#include "NFscheduler/Scheduler.h"

// Include the core files needed to run the simulation
#include "NFcore/NFcore.hh"
#include "NFutil/NFutil.hh"
#include "NFinput/NFinput.hh"
#include "NFreactions/NFreactions.hh"
#include "NFfunction/NFfunction.hh"

// NFtest/ headers removed — test harnesses not needed for library use

#endif /*NFSIM_HH_*/

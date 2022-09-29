# PyNuML: HDF5 IO and ML processing for neutrino physics

## Event IO

PyNuML is a freamework for interfacing with neutrino physics event data. HDF5 files produced using the NuML standard (link) contain tabular data structures representing events, simulated particles and energy depositions, detector hits and any other information defined by the user. For large datasets, accessing the rows of a table corresponding to a specific event based on event index can become prohibitively slow. PyNuML includes a metadata standard for efficient event IO for arbitrarily large tables, and parallel file access using MPI processes.

## Semantic and instance labelling

Labels from Geant4.

## ML object formation

Graphs, pixel maps and so on.

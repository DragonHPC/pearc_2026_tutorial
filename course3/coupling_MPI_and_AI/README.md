# Coupling MPI applications with PyTorch-based inference and training
Building upon concepts in the previous
courses, this session features an exercise creating a prototype coupled AI+HPC workflow. The goal for this session is
to show beginners and intermediate users best practices for implementing a workflow with DragonHPC and allow
advanced users to expand on the example for their use cases.

Tips and Tricks:
- model needs to be in root directory. you may need to edit the path.
- make sure you have at least 12GB of RAM for the container. Also as much swap space as docker allows (maybe not actually helpful).


Working Examples:
> dragon agents_hello_world.py
> .
> .
> .
> source setup.sh
> time dragon nan_detector_agent.py
> .
> .
> .
real    4m48.951s
user    28m37.280s
sys     7m24.459s

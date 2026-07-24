# Coupling MPI applications with PyTorch-based inference and training
Building upon concepts in the previous
courses, this session features an exercise creating a prototype coupled AI+HPC workflow. The goal for this session is
to show beginners and intermediate users best practices for implementing a workflow with DragonHPC and allow
advanced users to expand on the example for their use cases.

Tips and Tricks:
- The model needs to be in root directory. If it is not, you may need to edit the path in the workflow files.
- Increasing the RAM for the container can improve the performance. If you run into OOM errors you will need to either increase the RAM or increase the swap space.

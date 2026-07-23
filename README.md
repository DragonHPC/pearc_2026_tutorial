# Programming High-Performance AI-coupled HPC Workflows

## Tutorial Goals
The primary goal of this tutorial is that attendees gain a deeper understanding of how to design high-performance AI-coupled
HPC workflows through direct experience with DragonHPC [6], a distributed runtime that supports a variety of orchestration
and communication patterns typical of such workflows. To achieve this primary goal, the tutorial is organized into three courses
each covering a significant contributing concept. In the first two courses, attendees will become familiar with the standard
Python multiprocessing API, how DragonHPC’s extension enables multi-node computation needed for data processing and AI
workloads at-scale, and techniques to optimize data exchange within the workflow (including DragonHPC’s features that simplify
remote data placement for users). In the last course, attendees will learn how to construct an AI+HPC coupled workflow using
DragonHPC.

## Outcomes
Attendees will be provided exercises and solutions, primarily in the form of Jupyter notebooks. These notebooks are designed to
give attendees a variety of code patterns commonly found in AI+HPC workflows that can be easily incorporated into their own
use cases. By learning how to take advantage of the features of DragonHPC, attendees will have a toolbox of techniques that
can be explored on their laptop or common HPC platforms and be able to scale their workloads to leadership scale systems with
little-to-no code changes.

## Agenda (with links to jupyter notebook tutorials)

We meet in Room 202AB for the tutorial at 9am.

| Time      | Minutes | Course | Topic/Exercise | Presenter(s) |
| --- | --- | --- | --- | --- |
| 9:00 - 9:15  |  15 | | Tutorial introduction | P. Mendygral |
| 9:15 - 9:45  | 30 | | Presentation: AI+HPC workflows and DragonHPC | C. Simpson |
| 9:45 - 10:00 | 15 | 1 | Preparations for exercises | P. Mendygral<br>T. Maiden |
| 10:00 - 10:30  | 30 | 1 | [Python multiprocessing across multiple nodes](course1/multiprocessing_across_nodes/multiprocessing_intro.ipynb) | P. Mendygral<br>D. Potts |
| 10:30 - 11:00  | 30 | | Coffee Break | |
| 11:00 - 11:35  | 35 | 1 | [Managing DDict objects across processes <br> in Python and C++ (DDict API)](course1/managing_data_with_ddict/ddict_tutorial.ipynb) | C. Simpson<br>K. Lee |
| 11:35 - 12:15  | 40 | 2 | [Using Python multiprocessing with GPUs, and<br>PyTorch for multi-node LLM inference](course2/multiprocessing_with_GPUs_and_LLMs/gpu_llm_inference_tutorial.ipynb) | P. Mendygral |
| 12:15 - 12:30  | 15 | 2 | Checkpoint with attendees/Q&A | All presenters |
| 12:30 - 1:30   | 60 |   | Lunch Break |  |
| 1:30 -  2:00  | 30 | 2 | [Orchestrating MPI applications with the<br>ProcessGroup API](course2/orchestrating_MPI/processgroup_mpi_tutorial.ipynb) | P. Mendygral |
| 2:00 - 2:30  | 30 | 2 | [Sharing data between MPI and other processes<br>using the DDict API](course2/sharing_data_mpi_and_others/ddict_tutorial_2.ipynb) | K. Lee<br>C. Simpson |
| 2:30 - 3:00  | 30 | 3 | [Coupling MPI applications with PyTorch-based<br>inference and training](course3/coupling_MPI_and_AI/mpi_pytorch_coupled_tutorial.ipynb) | P. Mendygral<br>C. Simpson |
| 3:00 - 3:30  | 30 | | Coffee Break | |
| 3:30 - 3:45  | 15 | 3 | Checkpoint with attendees/Q&A | All presenters |

| 3:45 - 4:50  | 65 | | Review/discussion/Q&A/hackathon| All presenters |
| 4:50 - 5:00  | 10 | | Wrap-up and next steps| P. Mendygral |

## Primary tool website
* [DragonHPC Homepage](http://dragonhpc.org)

## Primary tool documentation
* [DragonHPC](https://dragonhpc.github.io/dragon/doc/_build/html/index.html)

## Technical Organizers and Contributers
* Pete Mendygral, HPE
* Kent Lee, HPE
* Eric Cozzi, HPE
* Christine Simpson, Argonne National Laboratory
* Davin Potts, Appliomics
* Tom Maiden, PSC
* TJ Olesky, PSC

# DragonHPC PEARC Tutorial Environment

Welcome to the DragonHPC PEARC Tutorial Environment! We're excited to help you get to know Dragon! Please
open the [Requirements Guide](REQUIREMENTS.md) for instructions on how to set up your environment for running
the DragonHPC tutorials.

# Tip

The tutorials are in Jupyter notebook form. If you make a mistake while writing code you may leave
your notebook in a bad state. If that happens you can restart the kernel to recover from it. After
restarting the kernel, make sure to re-execute any start up code that was necessary for the tutorial.

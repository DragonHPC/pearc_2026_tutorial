# Programming High-Performance AI-coupled HPC Workflows

## Tutorial Goals

The primary goal of this tutorial is that attendees gain a deeper understanding
of how to design high-performance AI-coupled HPC workflows through direct
experience with DragonHPC [6], a distributed runtime that supports a variety of
orchestration and communication patterns typical of such workflows. To achieve
this primary goal, the tutorial is organized into three courses each covering a
significant contributing concept. In the first two courses, attendees will become
familiar with the standard Python multiprocessing API, how DragonHPC’s extension
enables multi-node computation needed for data processing and AI workloads
at-scale, and techniques to optimize data exchange within the workflow (including
DragonHPC’s features that simplify remote data placement for users). In the last
course, attendees will learn how to construct an AI+HPC coupled workflow using
DragonHPC.

## Outcomes

Attendees will be provided exercises and solutions, primarily in the form of
Jupyter notebooks. These notebooks are designed to give attendees a variety of
code patterns commonly found in AI+HPC workflows that can be easily incorporated
into their own use cases. By learning how to take advantage of the features of
DragonHPC, attendees will have a toolbox of techniques that can be explored on
their laptop or common HPC platforms and be able to scale their workloads to
leadership scale systems with little-to-no code changes.

## Agenda (with links to jupyter notebook tutorials)

We meet in Room 202AB for the tutorial at 9am.

| Time      | Minutes | Course | Topic/Exercise | Presenter(s) |
| --- | --- | --- | --- | --- |
| 9:00 - 9:15  |  15 | | Tutorial introduction | P. Mendygral |
| 9:15 - 9:45  | 30 | | Presentation: AI+HPC workflows and DragonHPC | C. Simpson |
| 9:45 - 10:00 | 15 | 1 | Preparations for exercises | P. Mendygral<br>T. Maiden |
| 10:00 - 10:30  | 30 | 1 | [Python multiprocessing across multiple nodes](course1/multiprocessing_across_nodes/multiprocessing_tutorial.ipynb) | P. Mendygral<br>D. Potts |
| 10:30 - 11:00  | 30 | | Coffee Break | |
| 11:00 - 11:35  | 35 | 1 | [Managing DDict objects across processes <br> in Python and C++ (DDict API)](course1/managing_data_with_ddict/ddict_tutorial.ipynb) | C. Simpson<br>K. Lee |
| 11:35 - 12:15  | 40 | 2 | [Using Python multiprocessing with GPUs, and<br>PyTorch for multi-node LLM inference](course2/multiprocessing_with_GPUs_and_LLMs/gpu_llm_inference_tutorial.ipynb) | P. Mendygral |
| 12:15 - 12:30  | 15 | 2 | Checkpoint with attendees/Q&A | All presenters |
| 12:30 - 1:30   | 60 |   | Lunch Break |  |
| 1:30 -  2:00  | 30 | 2 | [Orchestrating MPI applications with the<br>ProcessGroup API](course2/orchestrating_MPI/processgroup_mpi_tutorial.ipynb) | P. Mendygral |
| 2:00 - 2:30  | 30 | 2 | [Sharing data between MPI and other processes<br>using the DDict API](course2/sharing_data_mpi_and_others/ddict_tutorial_2.ipynb) | K. Lee<br>C. Simpson |
| 2:30 - 3:00  | 30 | 3 | [Coupling MPI applications with PyTorch-based<br>inference and training](course3/coupling_MPI_and_AI/agentic_loop_exercises.ipynb) | P. Mendygral<br>C. Simpson |
| 3:00 - 3:30  | 30 | | Coffee Break | |
| 3:30 - 3:45  | 15 | 3 | Checkpoint with attendees/Q&A | All presenters |
| 3:45 - 4:50  | 65 |  | Review/discussion/Q&A/hackathon | All presenters |
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

# Starting DragonHPC PEARC Tutorial Environment

Welcome to the DragonHPC PEARC Tutorial Environment! We're excited to help you
get to know Dragon! Please open the [Requirements Guide](REQUIREMENTS.md) for
instructions on how to set up your environment for running the DragonHPC
tutorials. You should set up your environment before arriving at the tutorial if
at all possible due to limited bandwidth at the conference.

Once the environment is set up you can follow these directions on the day of the
tutorial to get Jupyter Server started and prepare for participating in the
tutorial.

After completing the setup in the [Requirements Guide](REQUIREMENTS.md) if
you are using VS Code, upon opening the folder VS Code may run `dragon-jupyter`
automatically and prompt you to open a web page in your browser. Click the button
to open the URL in your browswer. The webpage will ask for a token or password. The
token in this case is `dragon-vscode`. Enter that in the field and click **Log in**.
If this worked, then skip down to [displaying the agenda](#displaying-the-agenda).

If VS Code does not automatically run `dragon-jupyter` or you are not using VS
Code, then you can bring up a terminal windows and type

```bash
dragon-jupyter
```

This will run the jupyter server inside Dragon. It will print a number of lines to
the terminal that include these lines.

```bash
    To access the server, open this file in a browser:
        file:///home/vscode/.local/share/jupyter/runtime/jpserver-17153-open.html
    Or copy and paste one of these URLs:
        http://23044e962396:8889/tree?token=7d4e7581d9bab345f0ca1a853a6f3572b858898a1b009830
        http://127.0.0.1:8889/tree?token=7d4e7581d9bab345f0ca1a853a6f3572b858898a1b009830
    The server is listening on all interfaces, so any hostname or IP of this machine will work.
```

When prompted, click the button to open the URL in your browser. At the **Log in** prompt enter
your own token as printed like it is above after the string *?token=*. Copy the token and then
pasted it into the **Log in** field.

If after running `dragon-jupyter` you don't get a pop-up asking you to open a URL in your browswer, look for the line that looks like this in the terminal window.

```bash
http://127.0.0.1:8888/tree?token=e335a788b94549b8113efa59c098948e911221e56e34cdbf
```

Paste your similar line it into your browser. If it still asks for a token then copy your token from after the `?token=` and paste that into the **Log in** field.

## Displaying the Agenda

Once you have the Jupyter directory page in view, click on the `README.ipynb` notebook and
execute the cell with in it *with the play button at the top*. This provides the agenda for the day and hyperlinks to all the tutorials.

Once you have the Agenda displayed in the notebook you are ready for the tutorial to begin!

# Troubleshooting

Consult the [troubleshooting guide](TROUBLESHOOTING.md) if you have problems while
working through the tutorials.
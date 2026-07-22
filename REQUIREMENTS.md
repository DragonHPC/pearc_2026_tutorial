# Setting Up Your Laptop

To prepare for the tutorial you can take several approaches depending on your
needs and the requirements your organization may place on you for setup. This
guide may not cover all the details as it is assumed that you have some
familiarity with setting up a development environment on your laptop.

There are several ways to set up your environment for the tutorials, but in the end
the tutorials are in jupyter notebook form and we will want to run jupyter on
your laptop inside Dragon. The sections below provide a number of ways you can
configure your laptop.

Please first follow one of the these subsections, depending on your laptop and
environment.

* [Setting Up Windows](#setting-up-windows)
* [Setting Up Mac OS](#setting-up-mac-os)
* [Setting Up Linux](#setting-up-linux)

<a name="smoke-test"></a>
Once you have followed the steps in one of these sections you can open a terminal
window (see Terminal menu in VS Code) or however you get a terminal window in the
environment you built. If running outside of a container make sure you activate
your virtual environment inside the terminal. Then from the root directory of the
repo run:

```bash
dragon smoke_test.py
```

If `smoke_test.py` runs sucessfully, then you are ready to attend the Dragon tutorial.
We are looking forward to seeing you there!


## Setting Up Windows


Windows does not provide a native Linux environment and does not support POSIX natively, so
you must install WSL 2 (or some variant) on Windows. Then there are two directions you can
go.

* [Windows with Docker](#windows-with-docker)
* [WSL 2](#wsl-2)


Follow directions in whichever is most appropriate for your environment. Docker setup
is easier if you have access to it.

### Windows with Docker

If you have Docker Desktop and VS Code, then you should be able to follow the directions
in [the section on running with VS Code and Docker](#running-in-a-docker-container-with-vs-code).

If you do not have VS Code and/or Docker desktop installed, then you may still be able to
run a container and use DevPod to start it. Follow the instructions to [use DevPod to
start the container](#using-devpod-to-run-in-a-docker-container-without-vs-code).

### WSL 2

If you are running natively under WSL 2 (or some variant) then you would follow the directions
to run [natively on your Laptop](#running-natively-on-your-laptop).

## Setting up Mac OS

Mac OS has two different options for setting up Dragon on your laptop. Please
pick the route that best matches your desired setup. The first method, running in a Docker
container with VS Code is the easiest route if you have that available to you.

* [Mac OS with Docker](#mac-os-with-docker)
* [Mac OS Native](#mac-os-native)

### Mac OS with Docker

To run with VS Code and Docker please follow the instructions to
[use VS Code and Docker](#running-in-a-docker-container-with-vs-code).

If running on Mac OS with access to Docker, but without VS Code, you can
[follow the directions for using DevPod](#using-devpod-to-run-in-a-docker-container-without-vs-code)
to get up and running on Mac OS in a container.

### Mac OS Native

If you are running natively on your Mac then you would follow the directions
to run [natively on your Laptop](#running-natively-on-your-laptop).

## Setting up Linux

Please follow the directions to run [natively on your
Laptop](#running-natively-on-your-laptop).

## Running in a Docker Container with VS Code

This is the easiest and preferred route. Clone this repository to your
laptop. Insure you have Visual Studio Code and Docker Desktop installed.
You will need:

* A working Docker engine.
  See https://docs.docker.com/engine/install/ and/or https://docs.docker.com/desktop/
  for Docker installation instructions.

* VS Code with the Dev Containers extension installed.
  See https://code.visualstudio.com/docs/devcontainers/tutorial for setup instructions.

Once these things are available you can follow these steps to get up and running.

1. Open this project folder locally in VS Code (File > Open Folder...).
2. Click the pop-up notification in the bottom-right corner that says "Reopen in Container".
3. VS Code will build the environment, reload the window, and connect you directly into the container.
4. If you miss the pop-up, open the Command Palette by pressing Ctrl+Shift+P (or Cmd+Shift+P on macOS) and run "Dev Containers: Reopen in Container".
5. VS Code will run "dragon-jupyter" after the container starts. You can click the bottom-right pop-up
   "Open in Browser" when prompted. You will see the browser window with the directory structure.
6. If you like, you can click on the [test.ipynb](test.ipynb) file in the top directory and follow the directions in it.
7. Return to the top of this document and run the [smoke test](#smoke-test) that is recommended
as your last step for set up.


## Using DevPod to run in a Docker Container without VS Code

You may wish to use DevPod if running without VS Code. If you don't already have it, you will need
to install the DevPod CLI: https://devpod.sh/docs/getting-started/install.
Then add the DevPod Docker provider: https://devpod.sh/docs/quickstart/devpod-cli

This repository includes three scripts to start, connect to, and stop the DragonHPC DevContainer:

   * DevPod-start - starts the DragonHPC DevContainer in the background
   * DevPod-ssh   - opens an SSH connection to the DragonHPC DevContainer
   * DevPod-stop  - stops and removes the DragonHPC DevContainer

Once DevPod is installed and configured on your laptop, running `./DevPod-start` from the root
directory will start the container. Then you will need to `ssh` into the container with
`./DevPod-ssh`. Typing `exit` when finished with the container will exit the `ssh` session.
Finally, `DevPod-stop` will stop the container once you no longer needed it.

You should start and ssh into the container and then go back to the beginning of this document
and run the [smoke test](#smoke-test) to be sure you having everything set up correctly.

## Running natively on your Laptop

In this mode you run Dragon natively on Windows with WSL 2 or Mac OS. These
directions are considerably less detailed and it is up to you to know how to
install and configure software on your Windows or Mac laptop. In this mode
you are not running in a container.

If planning to running natively on Windows you will need to install WSL 2. You
can [follow directions from Microsoft to install
it](https://learn.microsoft.com/en-us/windows/wsl/install).

These steps should be executed in order as presented here. System Dependencies need to be
installed before pip installing Python dependencies.

You will want to make sure you have a g++ compiler.

* That may be available with the XCode Command-line Tools and/or by installing the XCode
package on Mac OS. If you don't find `g++` there, then using Brew for Mac OS to
install the GNU C++ compiler may be required.
* Under WSL 2 you will need to do `sudo apt-get install g++` to install the C++ compiler.
* Under Linux insure you have the necessary compiler and dev environment installed.

You will need an open-mpi library installed on your laptop to be able to run the MPI based
tutorials.

* The Brew package manager for Mac OS includes a package called open-mpi
that should prove useful. `brew install open-mpi`
* The WSL 2 environment will need you to `sudo apt-get install libopenmpi-dev openmpi-bin`.
* The packages under Linux are installed with `sudo apt-get install libopenmpi-dev openmpi-bin`
or other appropriate package manager.

To run natively you will need to install a version of Python. Python
3.12 is the default when running in a container. You may wish to have that
version as well, but 3.11, 12, or 13 should run just fine.

Open a terminal window, navigate to the repo top-level directory. You need to
create and activate a Python virtual environment. After activating, you will need to install
a number of required Python packages which are provided in requirements.txt. Navigate to the
top-level directory of the repository. From there do a

```bash
python3 -m venv env3.12
. env3.12/bin/activate
pip install -r requirements.txt
```

This will install all the required packages needed by the tutorial.

With the virtual environment active, navigate to
`course2/sharing_data_mpi_and_others`. In that directory there is a Makefile. If
you can type `make` in that directory and the program `worker.cpp` program
compiles, then you have all that you need for compiling the code.

Then proceed back to the top of this file and run the [smoke test](#smoke-test) program as
instructed to make sure you have all the dependencies installed correctly.
Authentication Simulator with Computer Vision
=======================

The project aims to boost security and convenience in program access by implementing a facial recognition-based authentication system in a web application. This system, employing supervised learning techniques, captures and analyzes facial features through live webcam feeds to accurately identify authorized users, offering an alternative and potentially more secure authentication method.

Table of Contents
-----------------

* [Developer Setup](#developer-setup)
    + [External Dependencies](#external-dependencies)
    + [Developer Installation](#developer-installation)
    + [Running](#running)
* [Contributing](#contributing)
    + [Branch Naming](#branch-naming)
    + [Commit Messages](#commit-messages)
* [Infrastructure](#infrastructure)
    + [Environments](#environments)

Developer Setup
---------------

### External Dependencies

| Program       | Version | Website                                                    | Notes                                                              |
|---------------|---------|------------------------------------------------------------|--------------------------------------------------------------------|
| Git           | Latest  | https://git-scm.com                                        | --                                                                 |
| Python        | 3.10.4  | https://www.python.org/downloads/release/python-3104/      | Install with OS installer for your environment                     |
| pip           | Latest  | https://www.python.org/downloads/release/python-3104/      | (Optional) Used to install packages, others may be used            |

### Developer Installation

Use the following commands to download from git the code and build.

```sh
git clone git@github.com:JaredD-SWENG/cmpsc445.git
cd cmpsc445

pip install -r requirements.txt
```

Depending on the installation ffmpeg may have to be installed.

```sh
sudo apt-get install ffmpeg libsm6 libxext6  -y
```

### Running

Use the following command to run the project for development.  By default, this will start a server on `http://localhost:8000`.

```sh
## assuming present working directory is cmpsc445
python server.py
```

### NOTEBOOKS

- TODO: DRUV PLEASE ADD NOTES


Contributing
------------

1. Pull the latest code from the `master` branch.
2. Create a branch off of `master` (see [Branch Naming](#branch-naming) below).
3. Make your code including any unit tests. Create & push as many commits as you need.
4. Squash your commits down to one (see [Commit Messages](#commit-messages) below).
5. Create a merge request on the GitLab repository.
6. Wait for the linting & unit tests to pass on the server.
7. If any additional commits are required (maybe to fix linting/unit test), squash them down into one commit again before merging.
8. Merge the branch into `master`.

### Branch Naming

Branch names should be in [kebab-case](https://winnercrespo.com/naming-conventions/). An example branch name is shown below with each part broken down and described.

```text
1         3
vv        vvvvvvvvvvvvvvvvvv
ak-cmp-123-fix-alignment
   ^^^^^^
   2
```

1. Your initials
2. The ticket number from [Jira](https://psu-oer-web-app.atlassian.net)
3. A brief description of what you are trying to accomplish in the branch

### Commit Messages

Commit messages should follow [this guide](https://chris.beams.io/posts/git-commit). In addition to the information required from the guide, the message should include the Jira ticket number. An example commit message is show below.

Infrastructure
--------------

### Environments

This section describes the minimum system that is required for running the server.

* Python server
The system must have python 3.10.4 or later installed

* Git
The system that is running the server must have git installed.

* System User
The user that runs the server must have privileges to run programs that allow internet access and to change security preferences of certain firewall settings.
    - Linux Systems: User must have sudo access to the machine
    - Windows Systems: User must have Administrator privileges


*Minimum System*
The following is an example environment:
* Operating System: Ubuntu 20.04.6 LTS
* Memory: 8 GB
* Storage: 80 GB

Installed Programs:
* Git
* SSH
* VIM
* Python



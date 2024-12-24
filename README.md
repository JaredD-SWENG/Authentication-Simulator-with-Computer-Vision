Authentication Simulator with Computer Vision
=======================

The project aims to boost security and convenience in program access by implementing a facial recognition-based authentication system in a web application. This system, employing supervised learning techniques, captures and analyzes facial features through live webcam feeds to accurately identify authorized users, offering an alternative and potentially more secure authentication method.

![CV-Auth](https://github.com/user-attachments/assets/286eed09-aef7-4288-841c-cb1895250436)


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

### Jupyter Notebooks
### Camera_Usage.ipynb - Notebook 1: Camera Configuration

This notebook covers camera configuration and usage. It includes the following sections:

- Camera Testing
- Configuration Saving
- Capturing Images

*Usage*

This notebook provides instructions for configuring and using a camera, including testing and capturing images.

### facenet_testing.ipynb - Notebook 2: Face Detection and Descriptors

This notebook focuses on face detection and descriptor computation using the Facenet model. It includes the following sections:

- Face Detection
- Landmark Detection
- Descriptor Computation

*Usage*

You can follow this notebook to detect faces in images and compute descriptors using the Facenet model.

### FaceRec.ipynb - Notebook 3: Facial Recognition

This notebook focuses on face recognition using a camera and a predefined database of faces. It includes the following sections:

- Capturing Images
- Face Recognition
- Database Management

*Usage*

You can follow the notebook to perform face recognition tasks, including capturing images and managing the face database.

## whispers.ipynb - Notebook 4: Graph-based Face Recognition

This notebook explores a graph-based approach to face recognition, specifically using the Whispers algorithm. It includes the following sections:

- Data Preprocessing
- Graph Creation
- Whispers Algorithm Application

*Usage*

You can use this notebook to understand and implement face recognition using the Whispers algorithm.


## Requirements:

Make sure you have the necessary Python packages and dependencies installed to run these notebooks. You can install them using the following command:

```bash
pip install -r requirements.txt
```


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
* Memory: 12 GB
* Storage: 80 GB

Installed Programs:
* Git
* SSH
* VIM
* Python



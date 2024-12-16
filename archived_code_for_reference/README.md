NoduleNet - Web-based CNN Solution for Lung Nodule Detection
Authors: Aki Wangcharoensap, Isaac von Riedemann, Dongwei Han, Jinghuan Gao, Zekai Li<br />
This repository is a template for your CMPT 340 course project.
Replace the title with your project title, and **add a snappy acronym that people remember (mnemonic)**.

Add a 1-2 line summary of your project here.

## Important Links

| [Timesheet](https://1sfu-my.sharepoint.com/:x:/r/personal/hamarneh_sfu_ca/Documents/TEACHING/CMPT340_FALL2024/FOR_STUDENTS/ProjectGroup_Timesheets/ProjectGroup_17_Timesheet.xlsx?d=w324e96c9a86b44ffb53f47204ea63aa1&csf=1&web=1&e=EdxngL) | [Slack channel](https://cmpt340fall2024.slack.com/archives/C07JS4B3MA6) | [Project report](https://www.overleaf.com/project/66d0b125964b3acdf1767543) |
|-----------|---------------|-------------------------|


- Timesheet: Link your timesheet (pinned in your project's Slack channel) where you track per student the time and tasks completed/participated for this project/
- Slack channel: Link your private Slack project channel.
- Project report: Link your Overleaf project report document.


## Video/demo/GIF
Record a short video (1:40 - 2 minutes maximum) or gif or a simple screen recording or even using PowerPoint with audio or with text, showcasing your work.


## Table of Contents
1. [Demo](#demo)

2. [Installation](#installation)

3. [Reproducing this project](#repro)

4. [Guidance](#guide)


<a name="demo"></a>
## 1. Example demo

A minimal example to showcase your work

```python
from amazing import amazingexample
imgs = amazingexample.demo()
for img in imgs:
    view(img)
```

### What to find where

Explain briefly what files are found where

```bash
repository
├── src                          ## source code of the package itself
├── scripts                      ## scripts, if needed
├── docs                         ## If needed, documentation   
├── README.md                    ## You are here
├── requirements.yml             ## If you use conda
```

<a name="installation"></a>

## 2. Installation

Provide sufficient instructions to reproduce and install your project. 
Provide _exact_ versions, test on CSIL or reference workstations.

```bash
git clone $THISREPO
cd $THISREPO
conda env create -f requirements.yml
conda activate amazing
```

<a name="repro"></a>
## 3. Reproduction
Demonstrate how your work can be reproduced, e.g. the results in your report.
```bash
mkdir tmp && cd tmp
wget https://yourstorageisourbusiness.com/dataset.zip
unzip dataset.zip
conda activate amazing
python evaluate.py --epochs=10 --data=/in/put/dir
```
Data can be found at ...
Output will be saved in ...

<a name="guide"></a>
## 4. Guidance

- Use [git](https://git-scm.com/book/en/v2)
    - Do NOT use history re-editing (rebase)
    - Commit messages should be informative:
        - No: 'this should fix it', 'bump' commit messages
        - Yes: 'Resolve invalid API call in updating X'
    - Do NOT include IDE folders (.idea), or hidden files. Update your .gitignore where needed.
    - Do NOT use the repository to upload data
- Use [VSCode](https://code.visualstudio.com/) or a similarly powerful IDE
- Use [Copilot for free](https://dev.to/twizelissa/how-to-enable-github-copilot-for-free-as-student-4kal)
- Sign up for [GitHub Education](https://education.github.com/)

# Web App

## Overview
This project uses Docker to containerize a Flask application. It also incorporates Tailwind CSS for styling. <br/>
This guide will help you set up the project on your local machine using Git for version control.

## Prerequisites
Before you begin, ensure you have the following installed:
[Docker](https://www.docker.com/)
[Git](https://git-scm.com/downloads)

## Getting Started
Follow these steps to get your project up and running.

### 1.) Clone the Repository
First, clone the repository to your local machine using Git:
```
git clone https://github.com/sfu-cmpt340/2024_3_project_17.git
cd ./2024_3_project_17
```

### 2. Build the Docker Image
Run the following command to build the Docker image based on the Dockerfile:
```
docker-compose build
```
This command will: <br/>
1.) Use the official Python 3.9-slim image. <br/>
2.) Install Python dependencies from requirements.txt. <br/>
3.) Install Node.js and npm dependencies for your front-end if required. <br/>

### 3.) Start the App
Once the image is built, use Docker Compose to start the Flask app service.
```
docker-compose up
```
Docker will: <br/>
1.) Start the Flask app, exposing it at localhost:5001. <br/>
2.) Mount your current directory into the container (so code changes will be reflected without rebuilding the container). <br/>

## Accessing the App
After running the previous command, by default, you can access the Flask app by visiting:
http://localhost:5001

##  Stop the App
To stop the app, press `Ctrl + C` in the terminal where the app is running, or you can use:
```
docker-compose down
```

## Dependencies

### Python Dependencies 
Ensure that all necessary Python packages (such as Flask and PyTorch) are listed in the requirements.txt file. 
For example:
```
Flask>=2.2.3,<3.0.0

# Flask dependencies
Werkzeug>=2.2.3,<3.0.0
torch==2.0.0
```

### NPM Dependencies 
If you need to install any specific npm dependencies, list them in your package.json file.







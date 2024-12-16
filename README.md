# CMPT340 Project Fall 2024: NoduleNet - A Web-based nnUNet Solution for Lung Nodule Detection

---

## Authors: 
Aki Wangcharoensap, Isaac von Riedemann, Dongwei Han, Jinghuan Gao, Zekai Li

---

## Brief Introducation
This full-stack application, integrated with nnUNet, is designed to assist in diagnosing lung nodules by providing detailed predictions such as their location and shape.

---

## Important Links

| [Timesheet](https://1sfu-my.sharepoint.com/:x:/r/personal/hamarneh_sfu_ca/Documents/TEACHING/CMPT340_FALL2024/FOR_STUDENTS/ProjectGroup_Timesheets/ProjectGroup_17_Timesheet.xlsx?d=w324e96c9a86b44ffb53f47204ea63aa1&csf=1&web=1&e=EdxngL)  | [Slack Channel](https://cmpt340fall2024.slack.com/archives/C07JS4B3MA6) | [Project Report](https://www.overleaf.com/project/66d0b125964b3acdf1767543) |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------|-----------------------------------------------------------------------------|


---

## Demonstration Video
| [Video Link](https://photos.onedrive.com/share/583D81594AFDFA88!2268?cid=583D81594AFDFA88&resId=583D81594AFDFA88!2268&authkey=!AGzrJG5HYHzrULA&ithint=video&e=JeIJGR) |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
---

## Example Code
```python
@app.route('/')
def index():
    return render_template('index.html')
```

---

## Project Structure
```bash
repository
├── Dockerfile
├── LICENSE
├── README.md
├── app.py
├── archived_code_for_reference 
├── docker-compose.yml
├── package.json
├── requirements.txt
├── src
│   ├── app
│   ├── display_ct_dicom_images
│   ├── models
│   │   ├── UNet
│   │   ├── UNetPlusPlus
│   │   └── nnUNet
│   ├── preprocessing
├── static
├── templates
│── ten_ct_slice_samples_from_various_patients_in_test_dataset
```

---

## Get Started

### [0] Install the software
Please install the following prerequisites if you have not installed them yet:

| [Docker](https://www.docker.com/) | [Git](https://git-scm.com/downloads) |
|-----------------------------------|--------------------------------------|


### [1] Clone the Repository
Please clone the repository to your local machine using Git:
```bash
git clone git@github.com:sfu-cmpt340/2024_3_project_17.git
cd 2024_3_project_17
```

### [2] Download the model
Please download the following model we have trained and save it under `src/models/unet_family/nnUNet/nnUNet_results/Dataset777_LungCT/nnUNetTrainer_500epochs__nnUNetResEncUNetLPlans__2d/fold_0/`

| [Trained nnunetv2 model](https://1sfu-my.sharepoint.com/:u:/g/personal/dongweih_sfu_ca/ER0NUIXUIrBDrZb91SMWNtABFhPJORGlX9bH2MsnGqrzMA?e=zrYX29) |
|-------------------------------------------------------------------------------------------------------------------------------------------------|


### [3] Build the Docker Image
Please run the following command to build the docker image:
```bash
docker-compose build
```
This command will: 
- Use the official Python:3.11.6-slim docker image. 
- Install Python dependencies from requirements.txt. 
- Install Node.js and npm dependencies.

### [4] Start the app
Please run the following command to start the app.
```bash
docker-compose up
```
This command will: 
- Start the Flask app, exposing it at localhost:5001.
- Mount your current directory into the container so that any code changes are reflected without needing to rebuild the container.

### [5] Enter the website
Please visit the URL via your Chrome browser:

| http://localhost:5001 |
|-----------------------|

### [6] Stop the app
Please run the following command to stop the app 
```bash
docker-compose down
```

---

## Reproduction
- Complete steps [1] through [5] in the Get Started section.
- Click the Demo button located at the top of the website.
- Upload a 2D DICOM scan from the provided `ten_ct_slice_samples_from_various_patients_in_test_dataset` folder.
- Wait a few seconds for the model’s prediction to process and display.
- View the nodule mask and the detailed information provided.
- Upload another DICOM scan to repeat the process.

---

## Report
| [Overleaf LaTeX](https://www.overleaf.com/project/66d0b125964b3acdf1767543) |
|-----------------------------------------------------------------------------|
---

## Acknowledgments
We would like to thank our Professor Hamarneh, and our TAs, Kumar and Weina, for their invaluable guidance and support throughout the completion of our project. We also extend our gratitude to nnUNet and all the references cited in our report. In addition, GPT-4o was used as an assistant to debug and refine the code, ensuring its functionality and optimization. It also played a key role in generating an illustration of a lung nodules, featured on the homepage. Leveraging GPT-4o enabled us to complete the project efficiently and thoroughly, meeting the demands of a strict timeline.

---








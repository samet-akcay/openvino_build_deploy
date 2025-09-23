# Spot the Object with OpenVINO™

The demo detects, tracks and counts defined objects in front of the webcam. The default object is a hazelnut, but it can be changed to any other object. It works especially good with a conveyor belt.

![spot_the_object](https://github.com/user-attachments/assets/e0b1f56a-a7b3-4bf0-a056-1fac804c2de3)

## Quick Launch using Setup Scripts

If you want a **quick setup** without manually installing dependencies, use the provided installer scripts. These scripts will **automatically configure** everything needed to run the Spot the Object Demo.

### **For Windows**

1. Download the `install.bat` and `run.bat` files to your local directory.
2. Double-click `install.bat` to install dependencies and set up the environment.
3. After installation, double-click `run.bat` to start the demo.

### **For Linux and MacOS**

1. Download the `install.sh` and `run.sh` files to your local directory.
2. First, ensure the installer scripts have execute permissions:

```shell
chmod +x install.sh run.sh
```

3. Run the installer to set up everything:

```shell
./install.sh
```

4. After installation, start the demo by running:

```shell
./run.sh
```

These scripts will handle cloning the repository, creating the virtual environment, and installing dependencies automatically. If you prefer a manual setup, follow Steps 1-3 below.

## Manual Environment Setup

Here are the steps involved in this demo:

Step 1: Install Python and prerequisites

Step 2: Set up the environment

Step 3: Run the Application

Now, let's dive into the steps starting with installing Python.

## Step 0

Star the [repository](https://github.com/openvinotoolkit/openvino_build_deploy) (optional, but recommended :))

## Step 1

This project requires Python 3.10-3.13 and a few libraries. If you don't have Python installed on your machine, go to <https://www.python.org/downloads/> and download the latest version for your operating system. Follow the prompts to install Python, making sure to check the option to add Python to your PATH environment variable.

Install libraries and tools:

```shell
sudo apt install git python3-venv python3-dev
```

_NOTE: If you are using Windows, you may need to install [Microsoft Visual C++ Redistributable](https://aka.ms/vs/16/release/vc_redist.x64.exe) also._

## Step 2

1. Clone the Repository

To clone the repository, run the following command:

```shell
git clone https://github.com/openvinotoolkit/openvino_build_deploy.git
```

The above will clone the repository into a directory named "openvino_build_deploy" in the current directory. Then, navigate into the directory using the following command:

```shell
cd openvino_build_deploy/demos/spot_the_object_demo
```

2. Create a virtual environment

To create a virtual environment, open your terminal or command prompt and navigate to the directory where you want to create the environment. Then, run the following command:

```shell
python3 -m venv venv
```

This will create a new virtual environment named "venv" in the current directory.

3. Activate the environment

Activate the virtual environment using the following command:

```shell
source venv/bin/activate   # For Unix-based operating system such as Linux or macOS
```

_NOTE: If you are using Windows, use `venv\Scripts\activate` command instead._

This will activate the virtual environment and change your shell's prompt to indicate that you are now working within that environment.

4. Install the Packages

To install the required packages, run the following commands:

```shell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## GETI Model Setup (Optional)

If you want to use Intel GETI models, you'll need to have a trained GETI model available. The demo expects the model files to be in the following format:

```text
model/
├── model.xml     # OpenVINO model file
├── model.bin     # OpenVINO weights file
└── config.json   # Model configuration
```

**Option 1: Use the provided sample model**

The repository includes a sample GETI model in the `model/` directory that you can use for testing.

**Option 2: Train your own GETI model**

1. Use Intel GETI platform to train a detection model on your custom dataset
2. Export the trained model to OpenVINO format
3. Place the exported model files (`model.xml`, `model.bin`, `config.json`) in the `model/` directory

**Option 3: Use GETI SDK deployment**

If you have a GETI SDK deployment package, extract it and point to the model files using the `--detection_model` parameter.

## Step 3

To run the application, you can use either the **Ultralytics** backend (YOLO models) or the **GETI** backend (Intel GETI models).

### Backend Options

This demo supports two backends:

- **Ultralytics**: Uses YOLO World models (default)
- **GETI**: Uses Intel GETI trained models via Model API

### Basic Usage Examples

#### Using Ultralytics Backend (Default)

Run with webcam:

```shell
python main.py --stream 0
```

Run with video file:

```shell
python main.py --stream input.mp4
```

Run with specific class and model:

```shell
python main.py \
    --stream ./geti_sdk-deployment/sample_video.mp4 \
    --class_name "hazelnut" \
    --backend ultralytics \
    --detection_model yolov8m-worldv2 \
    --device CPU
```

#### Using GETI Backend

Run with GETI model:

```shell
python main.py \
    --stream ./geti_sdk-deployment/sample_video.mp4 \
    --class_name "hazelnut" \
    --backend geti \
    --detection_model ./model/model.xml \
    --device CPU
```

### Advanced Options

To change the class to detect, use the `--class_name` option. By default, hazelnut is used. You should also provide auxiliary classes to improve the detection:

```shell
python main.py --stream 0 --class_name hazelnut --aux_classes nut "brown ball"
```

For Ultralytics backend, you can select different YOLO models:

```shell
python main.py --stream 0 --backend ultralytics --detection_model yoloe-11s-seg
```

To change the inference device use the `--device` option. By default, AUTO is used:

```shell
python main.py --stream 0 --device GPU
```

### Window Size Options

You can control the display window size:

```shell
# Fullscreen mode (default)
python main.py --stream 0 --fullscreen True

# Custom window size
python main.py --stream 0 --fullscreen False --window_size 1280 720
```

Run the following to see all available options.

```shell
python main.py --help
```

[//]: # (telemetry pixel)
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=7003a37c-568d-40a5-9718-0d021d8589ca&project=demos/spot_the_object_demo&file=README.md" />

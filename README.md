# LOL-Robot-Detector

The LOL-Robot-Detector is a tool designed to identify and analyze cheating behavior patterns within the online video game "League of Legends". Utilizing machine learning models and anomaly detection techniques, this project aims to enhance the integrity of gameplay by distinguishing between normal and cheating players.

## Installation

1. Clone this repository to your local machine.
2. Ensure Python 3.x is installed.
3. Install the required dependencies by running `pip install -r requirements.txt` from the root directory of the project.

## Usage

1. **Data Preparation**: Use `cursurDetector.py` to read the mouse positions of your raw videos.
2. **Anomaly Detection**: Use 'analyzer.py' to investigate your raw mouse positions using existing models.
a. **Train your own model**: If you wanna train your own model, use the 'dataModifier.py' to extract the features of your raw mouse positions and use 'universal_scaler' to standrize them. Then you can use 'modelTrainer.py' to train your own model.
b. **Tip1**: Make sure you are consistently using 1080p, 30fps videos.

## Contacts

E-mail: solistoriashenny@gmail.com
QQ: 3480547309

## License

This project is licensed under the MIT License - see the LICENSE file for details.

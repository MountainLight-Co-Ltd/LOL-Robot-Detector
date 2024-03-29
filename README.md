# LOL-Robot-Detector

The LOL-Robot-Detector is a tool designed to identify and analyze cheating behavior patterns within the online video game "League of Legends". Utilizing machine learning models and anomaly detection techniques, this project aims to enhance the integrity of gameplay by distinguishing between normal and cheating players.

## Installation

This project uses Git LFS to manage large files, such as the neural network models `cursorDetector_n.pt` and `cursorDetector_x.pt`. Before cloning the repository or pulling updates, ensure Git LFS is installed on your system:

1. Download and install Git LFS from [https://git-lfs.github.com/](https://git-lfs.github.com/).
2. Set up Git LFS for your user account by running `git lfs install` in your terminal.

Once Git LFS is installed, you can clone the repository as usual, and the large files will be automatically handled by Git LFS.

3. Install the required dependencies by running `pip install -r requirements.txt` from the root directory of the project.

## 2024/3/30 Updates:
If you cannot clone the Yolo models, please use the following Google Drive links for model downloading.
1. cursorDetector_n.pt: https://drive.google.com/file/d/1FN_Xfey1k--QKS9_ps5i_YDrvEmx2KdK/view?usp=sharing
2. cursorDetector_x.pt: https://drive.google.com/file/d/1FZULNgxbfAVGk-93SG9VJF7XcLAKwo82/view?usp=sharing

## Usage

1. **Data Preparation**: Use `cursurDetector.py` to read the mouse positions of your raw videos.
2. **Anomaly Detection**: Use 'analyzer.py' to investigate your raw mouse positions using existing models.
3. **Train your own model**: If you wanna train your own model, use the 'dataModifier.py' to extract the features of your raw mouse positions and use 'universal_scaler' to standrize them. Then you can use 'modelTrainer.py' to train your own model.
4. **Tip1**: Make sure you are consistently using 1080p, 30fps videos.

## Contacts

E-mail: solistoriashenny@gmail.com
QQ: 3480547309

## License

This project is licensed under the MIT License - see the LICENSE file for details.

# Plant Disease Classification Project

## Overview

This project aims to classify images of plants into different categories, including healthy plants and those affected by various diseases. The project utilizes a deep learning approach, specifically employing a pre-trained ResNet18 model fine-tuned on a custom dataset of plant images.

Our final trained model achieved an impressive **97% accuracy** on the held-out test dataset, demonstrating its strong ability to generalize to unseen examples. When tested on completely new, real-world data (images not part of the original dataset), the model achieved a **75% accuracy**, indicating a good level of practical applicability while also highlighting potential areas for further improvement to bridge the gap between lab and real-world performance.

## Project Structure

**Front End:**
![image](https://github.com/user-attachments/assets/ad3cf2c7-c636-4236-8425-29a86d90a6e5)

**Back End:**
![image](https://github.com/user-attachments/assets/f425f589-a587-4fa2-9d91-fc7647b61d98)

## Dataset

The project utilizes a custom dataset of plant images organized into subdirectories, where each subdirectory represents a specific class (e.g., 'Apple___healthy', 'Corn___Common_rust'). The dataset includes images of healthy plants and plants showing symptoms of various diseases.

* **Total Images:** [Specify the total number of images in your dataset, e.g., 19178]
* **Number of Classes:** [Specify the total number of classes, e.g., 17]
* **Data Split:** The dataset was split into training and testing sets to evaluate the model's ability to generalize:
    * **Training Images:** [Specify the number of training images, e.g., 15342]
    * **Testing Images:** [Specify the number of testing images, e.g., 3836]

## Methodology

The project followed these key steps:

1.  **Data Loading and Preprocessing:**
    * The `torchvision` library was used to load the image data from the `Plant_Dataset` directory using `datasets.ImageFolder`.
    * Image transformations were applied to prepare the data for the model:
        * **Resizing:** Images were resized to 224x224 pixels, a standard input size for many CNN architectures.
        * **Tensor Conversion:** Images were converted to PyTorch tensors.
        * **Normalization:** Tensors were normalized using the mean and standard deviation values calculated from the ImageNet dataset (`mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`). This helps the model learn more effectively.
    * `DataLoader` was used to create iterable batches of the data for efficient training and evaluation.

2.  **Model Selection and Fine-tuning:**
    * A pre-trained ResNet18 model, available through `torchvision.models`, was chosen as the base architecture. Pre-trained models have learned general image features from large datasets like ImageNet, which can significantly speed up training and improve performance on new, related tasks.
    * The final fully connected layer of the ResNet18 model was replaced with a new linear layer with an output size equal to the number of classes in our plant disease dataset. This new layer was trained to classify the specific plant conditions in our data.
    * The model was moved to the available device (GPU if present, otherwise CPU) for faster computation.

3.  **Training:**
    * The model was trained using the Cross-Entropy Loss function, which is commonly used for multi-class classification tasks.
    * The Adam optimizer (`torch.optim.Adam`) was used to update the model's weights during training, with a learning rate of [Specify the learning rate you used, e.g., 0.001].
    * The training process involved iterating through the training data for a specified number of epochs [Specify the number of epochs you trained for, e.g., 10-20]. In each epoch, the model made predictions on the training images, the loss was calculated, and the gradients were used to update the model's weights via backpropagation. The training loss and accuracy were monitored to track the model's learning progress.

4.  **Evaluation:**
    * After training, the model's performance was evaluated on the held-out test dataset. The test loss and accuracy were calculated to assess how well the model generalizes to unseen data.
    * The achieved **test accuracy of 97%** indicates a strong ability of the model to correctly classify plant images it has never seen before from the original dataset distribution.

5.  **Saving the Trained Model:**
    * The trained model's weights were saved to a file (`plant_classification_model.pth`) using `torch.save()`. This allows for easy loading and reuse of the trained model for future predictions.

## Achieving 97% Test Accuracy

The high test accuracy of 97% was likely achieved through a combination of factors:

* **Effective use of a pre-trained model:** Leveraging the features learned by ResNet18 on a massive dataset provided a strong starting point.
* **Sufficient and well-labeled data:** The quality and quantity of the plant disease dataset played a crucial role.
* **Appropriate image preprocessing:** Resizing and normalization ensured consistent input for the model.
* **Optimal training parameters:** The learning rate, number of epochs, and optimizer choice were likely well-suited for this task.
* **Data augmentation (if used):** [If you used data augmentation, describe it here, e.g., "Applying data augmentation techniques such as random rotations, flips, and color jitter during training helped the model generalize better by exposing it to more variations of the input images."]
* **Careful data splitting:** Ensuring a representative split between training and testing data is essential for accurate evaluation.
* **Addressing data quality issues:** [Mention if you encountered and resolved issues like corrupted images, as discussed previously.]

## Performance on New, Real-World Data (75% Accuracy)

While the model achieved 97% accuracy on the test set (which comes from the same distribution as the training data), its performance dropped to 75% when evaluated on completely new, real-world data. This difference can be attributed to several factors:

* **Domain Shift:** Real-world images might differ significantly from the images in the original dataset in terms of lighting conditions, image quality, camera angles, background clutter, and the specific variations of diseases or healthy plant appearances.
* **Novelty:** The new data might contain plant varieties or disease manifestations not well-represented in the original dataset.
* **Image Quality Variations:** Real-world photos taken by different users or devices can have varying levels of quality and resolution.
* **Class Imbalance in New Data:** The distribution of healthy and diseased plants in the new data might differ from the original dataset.

This result highlights the challenges of deploying machine learning models in real-world scenarios and the importance of evaluating performance on diverse and representative data.

## Future Improvements

To further improve the model's performance, especially on new, real-world data, the following steps could be considered:

* **Augmenting with more realistic variations:** Incorporate data augmentation techniques that simulate real-world conditions (e.g., varying brightness, contrast, blur, noise).
* **Collecting more diverse data:** Expand the dataset to include images from various sources, environments, and devices.
* **Fine-tuning on real-world data:** If a labeled set of real-world images becomes available, fine-tuning the model on this data can help bridge the domain gap.
* **Exploring more robust architectures:** Experiment with more advanced CNN architectures that are known to be more robust to variations in input data.
* **Implementing transfer learning from larger, more diverse datasets:** Consider pre-training on even larger and more varied image datasets before fine-tuning on the plant disease data.
* **Active learning:** Implement strategies to selectively label and add the most informative real-world images to the training set.

## Getting Started (Optional - if you want to share the code)

1.  **Clone the repository:**
    ```bash
    git clone [repository URL]
    cd Plant_Disease_Classification
    ```
2.  **Create a Conda environment (recommended):**
    ```bash
    conda env create -f environment.yml
    conda activate plant_disease_env
    ```
    (You would need to create an `environment.yml` file listing the project dependencies, e.g., PyTorch, torchvision, Pillow)
3.  **Ensure the `Plant_Dataset` directory is structured correctly with images organized into class subdirectories.**
4.  **(Optional) Run the Jupyter Notebook `plant_classification.ipynb` to see the code used for training and evaluation.**
5.  **The trained model weights are provided in `plant_classification_model.pth`.**

## Output:
**Test Loss Of 1.2% & 97% Accuracy**
![Screenshot 2025-04-06 191652](https://github.com/user-attachments/assets/85e8d777-2017-4463-b8e5-f5cafd54ea88)
**Define, Load, Split Dataset**
![Screenshot 2025-04-06 191443](https://github.com/user-attachments/assets/30333392-3151-49fe-9327-2a6e6b6b9b65)


# Predicting the sum of MNIST Classification and a random single number digit

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sudo-rickroll/END2/blob/main/S3/main.ipynb)

This folder contains the package for prediction of the sum of an MNIST image and a random single digit number.

This project contains the following folder and file components:

<ul>
  <li><b>checkpoint</b> - The folder where the model checkpoint is stored.</li>
  <li><b>config</b> - The folder where the configuration file is stored.</li>
  <li><b>graphs</b> - The package where the python module to plot output graphs exists.</li>
  <li><b>images</b> - The folder where the output images (like graphs) are stored.</li>
  <li><b>models</b> - The package where the python module for model/architecture exists.</li>
  <li><b>parsers</b> - The package where the python modules to parse the configuration file and the command line arguments, exists.</li>
  <li><b>utils</b> - The package where the python modules for utility functions like dataset, dataloader and for the processes like train, test and running the pipeline, exists.   </li>
  <li><b>main.ipynb</b> - The colab notebook file to run the entire process and displays log the outputs during the process. (<i>Note: In this notebook, the process is run considering that the repository is already cloned to google drive.</i>)</li>
  <li><b>main.py</b> - The python file that contains the entry point to the entire pipeline.</li>
</ul>
</br>

## Using this repository

### On local desktop machine

Clone this repository.

Once the repository is cloned, type `python main.py` in the terminal. This will train the model.

Additionally, the `python main.py` will take four arguments - </br>
<ul>
  <li><i>--mode</i> -> Add <i>--mode 'Train'</i> to train the model or <i>--mode 'Validate'</i> to validate the model </li>
  <li><i>--config_path</i> -> Add <i>--config_path '&lt;path to the configuration file&gt;'</i> to specify the path to the configuration file if it exists in a different folder. By default, the configuration file exists in the 'config' folder and this argument need not be provided if the same configuration file is used.</li>
  <li><i>--checkpoint_save</i> -> Add <i>--checkpoint_save '&lt;path to save the checkpoint file&gt;'</i> to save the checkpoint of the model.</li>
  <li><i>--checkpoint_load</i> ->  Add <i>--checkpoint_load '&lt;path to load the checkpoint file from&gt;'</i> to load a saved checkpoint to the model.</li>
</ul>

For example, the following command will load the checkpoint file named 'mnist_sum.pth' from the path '/END2/S3/checkpoint/' to the model, train the model for a certain number of epochs specified in the configuration file and will then save the resulting model in the path '/END2/S3/checkpoint/' with the file name 'mnist_sum.pth' by replacing the previously existing checkpoint file, as it is of the same name:</br>
`python main.py --mode 'Train' --checkpoint_load './checkpoint/mnist_sum.pth' --checkpoint_save './checkpoint/mnist_sum.pth'`

### On Google Colaboratory

Upload this entire repository to your Google Drive manually or clone this repository onto your Google Drive through a google colab file by first mounting your google drive and changing the directory to the one where you need this repo to be cloned.

Once cloned, open the <b>main.ipynb</b> file and run all the commands or create your own colab file and change the working directory to the <b>S3</b> subdirectory and type `!python main.py`. This will take 4 arguments too, as mentioned in the previous section related to the local desktop machine process.
  
For example,</br>
`!python main.py --mode 'Train' --checkpoint_load './checkpoint/mnist_sum.pth' --checkpoint_save './checkpoint/mnist_sum.pth'` </br>

</br>

## Model Breakdown and Process Statistics

The model/architecture used for this prediction has the following structure :</br>

![Model Flowchart](https://user-images.githubusercontent.com/65642947/119257657-de828c80-bbe3-11eb-901a-0e631e81cf71.png)

This model was trained for 25 epochs, using a batch size of 100 in the dataloaders, SGD as Optimiser and NLL Loss function (all of them specified in the configuration file) with no image preprocessing done on MNIST Images, in the Train mode. The highest sum prediction accuracy of 74.57 % was obtained on the validation set in the 25th epoch, wherein the MNIST Digit prediction accuracy was at 99.11 % for that epoch.

</br>

## Output Evaluation Images

</br>
<b>Validation on a random sample obtained from the validation set : </b>

![Sample Validation](https://user-images.githubusercontent.com/65642947/119257855-d24aff00-bbe4-11eb-9a06-5cc662ecf3dc.jpg)
  

<b>Graph for training mode accuracies on the Training Set vs Validation Set : </b>

![Train and Test Accuracies Graph](https://user-images.githubusercontent.com/65642947/119257913-10e0b980-bbe5-11eb-84ac-82f2f905d9ac.jpg)
  

<b>Graph for training mode losses on the Training Set vs Validation Set : </b>

![Train and Test Losses Graph](https://user-images.githubusercontent.com/65642947/119257925-21912f80-bbe5-11eb-8721-b3872469be02.jpg)







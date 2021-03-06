### How to Use

#### Step 1: Setting Up The Structure for the Repository

    .
    ├── data
    |     ├── fonts
    |     ├── langdata
    |     ├── output
    |     ├── tessdata
    |     ├── tesseract
    |       
    ├── generate_training_data.sh
    ├── extract_model.sh
    ├── train.sh
    ├── out.sh
  

#### Step 2: Downloading Relavent Data
- Place the font you want to train in the fonts folder.
- download from link: https://github.com/tesseract-ocr/langdata_lstm and place the data inside the lang_data.
- go to link: https://github.com/tesseract-ocr/ download one specific language from either tessdata_best and place that inside the tessdata folder.
- Updating the variables in script, now since the scripts were written taking tamil language into consideration you'll have to update those variables to reflect
your language you need to train the system on. the scripts that require updation are as following
    - `generate_training_data.sh has a variable called --lang tam is where the tam would be replcaed with your lang`
    - `extract_model.sh where (tam.*) needs to be changed to required (lang.*)`
    - `train.sh where (tam.*) needs to be changed to required (lang.*)`
    - `out.sh where (tam.*) needs to be changed to required (lang.*)`

#### Step 3: Changing the parameters in the generate_training_data.sh and train.sh
1. In generate_training_data.sh
- Rename the font list to the name of your font
- change the page from 10 (which was used for testing) to 250 which seems to be a good number to avoid overfitting
2. In train.sh
- change --max_iterations to anything that you think would be suitable for your font

#### Step 4: Running the pipeline
run the command: `execute_train_pipeline.sh`

The pipeline will do the following things:
1. Construct the data from the font
2. Extract the Model for further fine tuning
3. train the model on the new font, the new model will be saved in finetuned_model

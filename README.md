# FSDL-PlantID
This repo contains the Python code files for the plant identification project which is a part of FSDL project. 


# Goals
- Train a fine-grained visual classification model on iNaturalist data, incorporating multimodal information (such as GPS) and possibly structured classification (e.g. predicting multiple layers of the taxonomy, like both species and family)
- Distill the model for efficient inference on mobile devices
- Allow users to explore, rate and submit model predictions for potential incorporation into further model training (e.g. users can see a list of the most likely IDs, select or provide the correct ID if they know it, and indicate their level of confidence in the ID)


# Repository structure
├── LICENSE <br>
├── README.md          <- The top-level README for developers using this project. <br>
├── data (Data is in external collaborative storage space)<br>
│   ├── external       <- Data from third party sources.<br>
│   ├── modeldata      <- 80% of the data selected to be used for model training, test, valiadation <br>
│   ├── QAdata         <- 20% of the data never seen by the model OR we can use alternate data scource <br>
│   ├── augmented      <- Augmented Data for model improvement <br>
│   ├── litedata       <- Data for only 5% of the plant species <br>
│   ├── interim        <- Intermediate data that has been transformed. <br>
│   ├── processed      <- The final, canonical data sets for modeling. <br>
│   └── raw            <- The original, immutable data dump. <br>
│<br>
├── models             <- Trained and serialized models, model predictions, or model summaries<br>
│<br>
├── notebooks          <- Individual Jupyter notebook folders. (FEEL FREE TO ADD REPO WITH YOUR PREFERRED NAME) <br>
│   ├── ben<br>
│   ├── for<br>
│   ├── stone<br>
│   ├── sv<br>
│   ├── yoda<br>
│   └── vib<br>
│<br>
└──  references         <- Data dictionaries, manuals, and all other explanatory materials. <br>
    ├── data <br>
    ├── previouswork <br>
    └── methodologies <br>
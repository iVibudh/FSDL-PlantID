# FSDL-PlantID
This repo contains the Python code files for the plant identification project which is a part of FSDL project. 


# Goals
- Train a fine-grained visual classification model on iNaturalist data, incorporating multimodal information (such as GPS) and possibly structured classification (e.g. predicting multiple layers of the taxonomy, like both species and family)
- Distill the model for efficient inference on mobile devices
- Allow users to explore, rate and submit model predictions for potential incorporation into further model training (e.g. users can see a list of the most likely IDs, select or provide the correct ID if they know it, and indicate their level of confidence in the ID)


# Repository structure
├── LICENSE
├── README.md          <- The top-level README for developers using this project.
├── data (Data is in external collaborative storage space)
│   ├── external       <- Data from third party sources.
│   ├── modeldata      <- 80% of the data selected to be used for model training, test, valiadation
│   ├── QAdata         <- 20% of the data never seen by the model OR we can use alternate data scource
│   ├── augmented      <- Augmented Data for model improvement 
│   ├── litedata       <- Data for only 5% of the plant species 
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Individual Jupyter notebook folders. (FEEL FREE TO ADD REPO WITH YOUR PREFERRED NAME) 
│   ├── ben
│   ├── for
│   ├── stone
│   ├── sv
│   ├── yoda
│   └── vib
│
└──  references         <- Data dictionaries, manuals, and all other explanatory materials. 
    ├── data
    ├── previouswork
    └── methodologies
# Santander Product Recommendation

## TEAM: Survey Corps

## File Structure

```
survey_corps.santander-project/
├── Models
│   ├── #1-Voting
│   │   ├── reproduce
│   │   │   ├── Product-Wise
│   │   │   │   ├── product-wise-lag.ipynb
│   │   │   │   ├── product-wise-lag.py
│   │   │   │   └── proposed-sampling-month.pptx
│   │   │   └── XGBoost
│   │   │       ├── xgboost.ipynb
│   │   │       └── xgboost.py
│   │   ├── voting.ipynb
│   │   └── voting.py
│   └── EDA
│       ├── EDA.ipynb
│       └── EDA.py
├── README.txt
└── Santander_Product_Recommendation.pdf
```

 - The report for the assignment is saved as 'Santander_Product_Recommendation.pdf'
 - We have submitted the best model. You can find them in 'Models' folder
 
 - The top submission of ours is in 'Voting folder'
 - For Voting we had used multiple models to get a consolidated score.
 - To reproduce those CSVs please look at 'reproduce' folder
 - In 'reproduce' folder you can find two sub folders XGBoost and Product-Wise
 
 - 'Product-Wise' folder needs to be reproduced for every product separately. 
    - We had used the Product Wise analysis to check which month is most suitable for training and trained accordingly
    - But as mentioned in the report, the accuracy is more dependant on the latest month than this.
    - We have also attached a PPT consisting of the inferences we found from running all such notebooks
 - On running the 'XGBoost' code we get it's respective CSV

 - Each folder has it's jupyter notebook(for better visualisation) and it's relavent python code.
 - Incase you want to skip the part where you run all the notebooks to get the CSVs, we have CSVs stored in drive
 
 [Link to Drive](https://iiitborg-my.sharepoint.com/:f:/g/personal/sai_rithwik_iiitb_org/Ek-ENHqo0Z1AgxXGsOitfl8BL_cvLxnZxMhmCokx9ZAlkw?e=oQo792)


 - The second best submission is for XGBoost.
 - It's relevant script is saved as 'xgboost.py'. We had also used it for voting

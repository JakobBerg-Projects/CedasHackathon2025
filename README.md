# CEDASHackathon2025
This project was developed for the CEDAS Data Science Hackathon 2025, a competitive 48-hour event hosted at the University of Bergen. Participants were tasked with analyzing and modeling a dataset from Eviny, containing thousands of electric vehicle (EV) charging sessions. Our goal was to explore trends in EV charging behavior and build predictive models to forecast future charging patterns.

🧠 Objective

The hackathon challenge involved:

Performing exploratory data analysis (EDA) on EV charging data.

Visualizing key insights clearly and informatively.

Building predictive models to estimate power consumption and charging behavior on unseen data.

Communicating results effectively through a final report and presentation.

📂 Dataset

The dataset was provided by Eviny, a leading Norwegian energy company, and included:

Charging session logs from 2022–2024.

Metadata about charging stations and vehicles.

Variables such as energy delivered, charging time, start/end time, location, power level, and more.

No prior domain knowledge in energy or EVs was required, but experience with data science, visualization, and machine learning was highly beneficial.

🛠️ Tools & Technologies
We used the following tools during the hackathon:

Python – Primary programming language

pandas, NumPy – Data wrangling and cleaning

matplotlib, seaborn, plotly – Data visualization

scikit-learn – Machine learning models

Jupyter Notebooks – Prototyping and collaboration

Git – Version control

✅ Results
Built a regression model that achieved solid performance on the holdout dataset.

Identified clear seasonal and time-of-day trends in EV charging.

Delivered a professional final report explaining our pipeline, insights, and predictions.

### 📄 Report

You can view the full project report [here](src/report.pdf).
[More info](https://echo.uib.no/arrangement/cedas-data-science-hackathon-2025)

## Dev

Create the folder `data` in `src`

### Using poetry

`poetry shell`

### Using pip

`pip install -r requirements.txt`

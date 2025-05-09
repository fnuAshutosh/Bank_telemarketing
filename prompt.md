Phase 1: Project Planning & Business Alignment
Prompt:
“Define the business problem this bank marketing project solves. How can increasing model precision reduce costs? Calculate the ROI if the bank reduces unnecessary calls by 20%. Identify key stakeholders (e.g., marketing team, compliance) and their needs. What ethical risks (e.g., bias against older customers) should we anticipate?”

Tasks:

Draft a problem statement linking ML to business goals.

Calculate cost savings using assumptions (e.g., 
5
/
c
a
l
l
,
5/call,100 profit/conversion).

List stakeholders and their priorities (e.g., compliance cares about fairness).

Outcome:

1-page business case document with ROI estimates and ethical risks.

Phase 2: Advanced Preprocessing & Feature Engineering
Prompt:
“Clean the dataset by handling ‘unknown’ values and outliers. Create 3-5 business-driven features (e.g., ‘call efficiency’). Use SHAP for preliminary feature analysis. How can we ensure the preprocessing reduces bias (e.g., fair representation of age groups)?”

Tasks:

Impute missing values with MICE (not just dropping rows).

Engineer features like duration / campaign and balance / age.

Generate SHAP summary plots during EDA to identify bias.

Outcome:

Clean dataset + feature engineering report (code snippets, SHAP plots).

Phase 3: Model Development & Optimization
Prompt:
“Train XGBoost, Logistic Regression, and a stacked ensemble. Use Optuna to tune for precision. Compare time-based vs. random validation splits. How does feature importance differ between models? Track experiments with MLflow.”

Tasks:

Optimize XGBoost for precision using Optuna.

Validate with time-based splits (if timestamps exist).

Document feature importance conflicts (e.g., SHAP vs. permutation).

Outcome:

Optuna tuning logs, model leaderboard (AUC/precision), and SHAP comparison.

Phase 4: Business Impact & Fairness
Prompt:
“Calculate cost savings using the confusion matrix. Audit model fairness for age > 60 using AIF360. How many false positives are eliminated? Propose debiasing strategies if bias is found.”

Tasks:

Compute savings: (False Positives Reduced * $5) + (True Positives * $100).

Run AIF360 disparity metrics for age/job groups.

Mitigate bias via reweighting or adversarial training.

Outcome:

ROI report and fairness audit summary (with mitigation steps).

Phase 5: Deployment & Scalability
Prompt:
“Containerize the best model with Docker. Build a FastAPI endpoint that takes customer data and returns predictions + SHAP explanations. Deploy on AWS EC2. How can we automate retraining?”

Tasks:

Write a FastAPI endpoint with /predict and /explain routes.

Deploy on AWS EC2 with Docker.

Set up GitHub Actions for CI/CD (e.g., retrain on new data).

Outcome:

Live API endpoint, Dockerfile, and CI/CD pipeline code.

Phase 6: Presentation & Recruiter Appeal
Prompt:
“Create a GitHub repo with a video demo of the API. Draft a LinkedIn post titled ‘Reducing Bank Marketing Costs with Ethical AI’. Highlight ROI, technical skills (Optuna, Docker), and fairness fixes. What visuals (AUC curves, SHAP plots) grab attention?”

Tasks:

Record a 2-minute video demo showing predictions and explanations.

Write a LinkedIn post with metrics (e.g., “18% cost reduction”).

Design a Streamlit dashboard for interactive exploration.

Outcome:

Portfolio-ready GitHub, LinkedIn post, and Streamlit app.

Final Deliverables Checklist
Technical: Jupyter notebooks, Dockerized API, Optuna logs.

Business: ROI report, fairness audit, 1-page executive summary.

Recruiter: Video demo, LinkedIn post, GitHub with clean docs.

By following these prompts, your project will demonstrate technical depth, business acumen, and communication skills—exactly what recruiters value! 🎯
# Airline Customer Satisfaction Modeling & Strategic Insights
#### Project Type: Predictive Analytics & Customer Experience Strategy
- Role Perspective: Chief Data Officer / Head of Analytics
- Primary Objective: Identify key drivers of airline customer satisfaction and translate model outputs into actionable business insights
________________________________________
## Project Overview
This project analyzes airline customer satisfaction data to understand which service attributes most strongly influence whether a passenger is satisfied or dissatisfied. Rather than focusing solely on predictive accuracy, the analysis emphasizes interpretability, stability, and business relevance.
Two complementary machine learning models are used:
- Decision Tree Classifier for transparency and interpretability
- Random Forest Classifier for robustness and validation at scale
The outputs of these models are analyzed diagnostically to identify high-impact service levers and to support executive-level decision making around customer experience investments.
________________________________________
## Business Problem
Airlines operate in a highly competitive environment where small improvements in customer satisfaction can lead to meaningful gains in retention, loyalty enrollment, and lifetime value. However, customer experience initiatives are often costly and require prioritization.
### The core business questions addressed in this project are:
- Which service attributes most strongly drive customer satisfaction?
- Are these drivers consistent across different modeling approaches?
- Where should airlines prioritize investment to maximize return?
- How can predictive analytics be operationalized for proactive customer management?
________________________________________
## Data Description
- Dataset: Invistico Airline Customer Satisfaction Survey
- Observations: ~130,000 passenger responses
- Target Variable: Customer satisfaction (Satisfied / Dissatisfied)
- Feature Categories:
  - Inflight experience (entertainment, seat comfort, food & drink)
  - Digital experience (online booking, check-in)
  - Service interactions (on-board service, cleanliness)
  - Operational factors (delays, travel class)
  - Customer characteristics (travel type, loyalty status)
Data preprocessing includes handling missing values, encoding categorical variables, and ensuring consistency across modeling pipelines.
________________________________________
## Analytical Approach
### 1. Decision Tree Model
- Provides a transparent, rule-based structure
- Highlights which service attributes create the earliest and most impactful splits between satisfied and dissatisfied customers
- Used primarily for interpretability and insight generation

### 2. Random Forest Model
- Ensemble of multiple decision trees trained on different data subsets
- Reduces variance and improves generalization
- Used to validate the stability of identified satisfaction drivers
Both models are evaluated using standard classification metrics, with particular emphasis on consistency of feature importance rather than marginal accuracy gains.
________________________________________
# Key Findings
### Experiential service factors dominate satisfaction outcomes
  - Inflight entertainment
  - Seat comfort
  - Ease of online booking
#### These features consistently rank as top drivers across both models, indicating robust and reliable insights.
#### Operational and demographic variables play a secondary role, suggesting that how the service feels matters more than who the customer is.
#### Increased model complexity yields diminishing returns without richer behavioral or transactional data.

________________________________________
## Business & Financial Implications
- Improvements to high-impact experience variables affect a broad segment of passengers, generating population-level gains.
- Even small reductions in dissatisfaction can translate into millions in incremental lifetime value.
- Targeting early-split features offers significantly higher ROI than optimizing niche or downstream service elements.
- Predictive satisfaction scoring enables proactive intervention, reducing churn and service recovery costs.
________________________________________
## Strategic Recommendations
#### 1.	Prioritize Experience Investments
- Focus capital allocation on inflight entertainment quality and seat comfort improvements.

#### 2.	Reduce Digital Friction
- Simplify online booking and check-in workflows to improve first-touch impressions.

#### 3.	Operationalize Predictive Models
- Deploy satisfaction risk scoring into customer engagement workflows to identify and retain at-risk passengers.


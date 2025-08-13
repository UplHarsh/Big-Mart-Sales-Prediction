# 🎯 Complete Question Bank: 50 Data Science Interview Questions

## Based on Harsh Soni's Resume & Big Mart Sales Prediction Project

### 📊 Data Science Fundamentals (5 Questions)

1. **[INTERMEDIATE]** Explain the bias-variance tradeoff and how it applies to the models you've used in your demand forecasting project at UPL. How did you balance this tradeoff?

2. **[INTERMEDIATE]** In this Big Mart project, we see RMSE used as the evaluation metric. Why might RMSE be preferred over MAE for sales prediction? When would you choose MAE instead?

3. **[INTERMEDIATE]** Describe the curse of dimensionality and how it might affect the feature engineering approach used in this Big Mart project with 45+ features.

4. **[BASIC]** Explain cross-validation and why the project uses a simple train-validation split instead of k-fold CV. What are the trade-offs?

5. **[BASIC]** What is the difference between supervised, unsupervised, and reinforcement learning? Give examples from your experience at UPL.

---

### 🤖 Machine Learning Algorithms (10 Questions)

6. **[ADVANCED]** The Big Mart project shows MLPRegressor performing best (1014 RMSE) compared to XGBoost (1058 RMSE). Why might neural networks outperform gradient boosting here? Explain the architectural advantages.

7. **[ADVANCED]** Explain XGBoost algorithm and its key hyperparameters. You used XGBoost in demand forecasting - how did you tune it for time series data?

8. **[INTERMEDIATE]** Random Forest shows high overfitting in this project (365 training RMSE vs 1029 validation RMSE). Explain why this happens and how to prevent it.

9. **[ADVANCED]** Compare ARIMA, SARIMA, and Prophet models that you used in demand forecasting. When would you choose each approach?

10. **[ADVANCED]** Explain the concept of ensemble methods. How would you create an ensemble using the Big Mart models to potentially improve the 1014 RMSE?

11. **[INTERMEDIATE]** For customer churn prediction, you used Random Forest. Explain why tree-based models work well for churn prediction and how you handled class imbalance.

12. **[ADVANCED]** What are the key differences between gradient boosting and adaptive boosting? How do you decide which regularization technique to use?

13. **[INTERMEDIATE]** Explain the architecture of neural networks used in MLPRegressor. How do you choose the number of hidden layers and neurons?

14. **[ADVANCED]** How do you handle seasonality and trends in time series forecasting? Compare different approaches you've used.

15. **[ADVANCED]** Explain transfer learning and how you might apply it to improve model performance with limited data.

---

### 🔧 Feature Engineering (8 Questions)

16. **[ADVANCED]** The Big Mart project creates 45+ features from 12 original variables. Walk me through your feature engineering strategy. Which engineered features would be most predictive and why?

17. **[INTERMEDIATE]** Explain the different missing value imputation strategies used in this project. Why use hierarchical imputation for Item_Weight instead of simple mean imputation?

18. **[INTERMEDIATE]** The project handles zero visibility values by replacing them with item type averages. Justify this approach and suggest alternatives.

19. **[ADVANCED]** Explain the creation of 'Store_Avg_Sales' and similar store-level features. Why might this lead to data leakage and how would you prevent it?

20. **[INTERMEDIATE]** Describe your approach to categorical encoding. Why use Label Encoding instead of One-Hot Encoding for some variables in this project?

21. **[ADVANCED]** In your GenAI document automation project, how did you extract and engineer features from unstructured text data?

22. **[INTERMEDIATE]** How do you create temporal features for time series data? What considerations are important for demand forecasting?

23. **[INTERMEDIATE]** Explain interaction features and when they are useful. How would you systematically create and validate interaction terms?

---

### 📈 Model Evaluation (6 Questions)

24. **[INTERMEDIATE]** The project achieves 1014 RMSE. How would you determine if this is good performance? What benchmarks would you establish?

25. **[ADVANCED]** Explain different validation strategies for time series vs cross-sectional data. How did you validate your demand forecasting models at UPL?

26. **[INTERMEDIATE]** For your churn prediction model, explain the difference between accuracy, precision, recall, and F1-score. Which metric did you optimize for and why?

27. **[ADVANCED]** How would you detect and diagnose overfitting in the neural network model? What regularization techniques would you apply?

28. **[ADVANCED]** Explain A/B testing for model deployment. How would you design an A/B test for deploying your churn prediction model?

29. **[ADVANCED]** How do you handle model drift in production? What monitoring strategies do you use to detect performance degradation?

---

### 💼 Business Understanding (6 Questions)

30. **[ADVANCED]** You generated $3M additional sales through demand forecasting. Walk me through how you measured this impact and ensured it was actually due to your model.

31. **[INTERMEDIATE]** In retail sales prediction, different products might have different seasonalities and trends. How would you account for this in the Big Mart model?

32. **[INTERMEDIATE]** Your customer churn model increased quarterly revenue by $5.77M. Explain the business logic: how does preventing churn translate to revenue?

33. **[INTERMEDIATE]** For inventory planning, how would you convert sales predictions into stock recommendations? What additional factors would you consider?

34. **[INTERMEDIATE]** Explain how you would present model results and limitations to non-technical stakeholders. How do you build trust in your predictions?

35. **[INTERMEDIATE]** How do you balance model complexity with business interpretability? When would you choose a simpler model over a more accurate one?

---

### ⚙️ Technical Implementation (8 Questions)

36. **[ADVANCED]** Walk me through your ML pipeline architecture for the demand forecasting system. How did you ensure scalability across 9 global regions?

37. **[ADVANCED]** You integrated with SAP BW, OpenHub, and Azure Delta Tables. Explain the data flow and how you ensured data quality throughout the pipeline.

38. **[ADVANCED]** Explain your CI/CD workflow for ML models. How do you ensure model reproducibility and version control?

39. **[ADVANCED]** For your Text-to-SQL application, how did you handle prompt engineering and ensure SQL query safety?

40. **[ADVANCED]** Describe your approach to hyperparameter tuning at scale. How do you balance exploration vs exploitation in hyperparameter search?

41. **[ADVANCED]** How do you design APIs for real-time model serving? What considerations do you have for latency and throughput?

42. **[ADVANCED]** Explain how you implement monitoring and alerting for production ML models. What metrics do you track?

43. **[ADVANCED]** How do you handle model versioning and rollback scenarios in production environments?

---

### 🚀 Advanced Topics (5 Questions)

44. **[ADVANCED]** You worked with Azure OpenAI and LangChain. Explain the challenges of deploying LLMs in production and how you addressed them in your document automation platform.

45. **[ADVANCED]** For semantic search in your document automation, compare different embedding approaches (OpenAI, HuggingFace, etc.). What factors influenced your choice?

46. **[ADVANCED]** Explain the concept of few-shot learning and how you might have applied it in your GenAI projects. What are the limitations?

47. **[ADVANCED]** Describe causal inference methods and how they differ from traditional ML. How might you apply causal inference to measure the true impact of your interventions?

48. **[ADVANCED]** Explain multi-objective optimization in ML. How would you balance accuracy, fairness, and computational cost in your churn prediction model?

---

### 💻 Coding & Implementation (2 Questions)

49. **[INTERMEDIATE]** Write pseudocode for handling missing values in the hierarchical approach used in Big Mart project. How would you implement this efficiently?

50. **[INTERMEDIATE]** Explain how you would implement cross-validation for time series data. What are the key differences from standard CV?

---

## 📊 Question Distribution by Difficulty

| Difficulty Level | Count | Percentage |
|------------------|--------|------------|
| **Advanced**     | 28     | 56%        |
| **Intermediate** | 19     | 38%        |
| **Basic**        | 3      | 6%         |

This distribution reflects the senior-level position and candidate's experience profile.

## 🎯 Domain Coverage

| Domain | Questions | Focus Areas |
|--------|-----------|-------------|
| **ML Algorithms** | 10 (20%) | XGBoost, Neural Networks, Time Series |
| **Feature Engineering** | 8 (16%) | Missing values, encoding, text features |
| **Technical Implementation** | 8 (16%) | MLOps, pipelines, production |
| **Business Understanding** | 6 (12%) | Impact measurement, stakeholder communication |
| **Model Evaluation** | 6 (12%) | Validation, metrics, monitoring |
| **Data Science Fundamentals** | 5 (10%) | Core concepts, statistical understanding |
| **Advanced Topics** | 5 (10%) | GenAI, LLMs, causal inference |
| **Coding** | 2 (4%) | Implementation, algorithms |

---

*This question bank is specifically tailored for Harsh Soni's background and the Big Mart Sales Prediction project context, providing comprehensive coverage of senior data scientist competencies.*
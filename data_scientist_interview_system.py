"""
Senior Data Scientist Interview System
=====================================

An interactive interview system designed for data scientist role candidates based on:
- Repository analysis: Big Mart Sales Prediction project
- Candidate profile: Harsh Soni's resume and experience
- Technical competency assessment across multiple domains

Author: AI Senior Data Scientist Interviewer
Date: 2024
"""

import json
import random
import time
from datetime import datetime
from typing import Dict, List, Tuple, Any
import pandas as pd

class DataScienceInterviewSystem:
    """
    A comprehensive interview system for evaluating data science candidates
    based on their experience and the specific project requirements.
    """
    
    def __init__(self):
        self.candidate_profile = {
            "name": "Harsh Soni",
            "current_role": "Data Scientist at UPL Ltd.",
            "experience_years": 3,
            "key_skills": [
                "GenAI", "LLM", "Azure OpenAI", "Time Series Forecasting", 
                "Customer Churn", "XGBoost", "Random Forest", "Deep Learning",
                "Python", "R", "SQL", "Power BI", "Azure Databricks"
            ],
            "major_projects": [
                "GenAI Document Automation Platform",
                "SKU-level Demand Forecasting", 
                "Customer Churn Prediction",
                "Text-to-SQL Application"
            ],
            "business_impact": {
                "cost_savings": "$23,000 + $1.3M",
                "revenue_generated": "$3M + $5.77M",
                "accuracy_improvements": "30% reduction in forecast errors"
            }
        }
        
        self.project_context = {
            "name": "Big Mart Sales Prediction",
            "type": "Retail Sales Forecasting",
            "algorithms_used": ["XGBoost", "Random Forest", "Neural Networks", "Linear Regression"],
            "key_techniques": ["Feature Engineering", "Missing Value Imputation", "Categorical Encoding"],
            "performance_metric": "RMSE",
            "business_goal": "Optimize inventory and sales planning"
        }
        
        self.interview_state = {
            "current_question": 0,
            "total_questions": 50,
            "responses": {},
            "scores": {},
            "start_time": None,
            "current_category": None
        }
        
        self.question_bank = self._initialize_question_bank()
        self.scoring_rubric = self._initialize_scoring_rubric()
        
    def _initialize_question_bank(self) -> Dict[str, List[Dict]]:
        """Initialize comprehensive question bank covering all data science domains"""
        
        return {
            "data_science_fundamentals": [
                {
                    "id": "dsf_001",
                    "question": "Explain the bias-variance tradeoff and how it applies to the models you've used in your demand forecasting project at UPL. How did you balance this tradeoff?",
                    "difficulty": "intermediate",
                    "expected_topics": ["bias", "variance", "overfitting", "underfitting", "regularization"],
                    "follow_up": "How would you detect and address high bias vs high variance in a forecasting model?"
                },
                {
                    "id": "dsf_002", 
                    "question": "In this Big Mart project, we see RMSE used as the evaluation metric. Why might RMSE be preferred over MAE for sales prediction? When would you choose MAE instead?",
                    "difficulty": "intermediate",
                    "expected_topics": ["RMSE", "MAE", "outliers", "error sensitivity", "business context"],
                    "follow_up": "How would you handle outliers in sales data that might skew your RMSE?"
                },
                {
                    "id": "dsf_003",
                    "question": "Describe the curse of dimensionality and how it might affect the feature engineering approach used in this Big Mart project with 45+ features.",
                    "difficulty": "intermediate",
                    "expected_topics": ["dimensionality", "feature selection", "regularization", "PCA"],
                    "follow_up": "What feature selection techniques would you use to combat this?"
                },
                {
                    "id": "dsf_004",
                    "question": "Explain cross-validation and why the project uses a simple train-validation split instead of k-fold CV. What are the trade-offs?",
                    "difficulty": "basic",
                    "expected_topics": ["cross-validation", "overfitting", "data leakage", "computational cost"],
                    "follow_up": "How would you implement time-series cross-validation for your demand forecasting work?"
                },
                {
                    "id": "dsf_005",
                    "question": "What is the difference between supervised, unsupervised, and reinforcement learning? Give examples from your experience at UPL.",
                    "difficulty": "basic",
                    "expected_topics": ["supervised", "unsupervised", "reinforcement", "business applications"],
                    "follow_up": "How did you apply unsupervised learning in your customer segmentation work?"
                }
            ],
            
            "machine_learning_algorithms": [
                {
                    "id": "mla_001",
                    "question": "The Big Mart project shows MLPRegressor performing best (1014 RMSE) compared to XGBoost (1058 RMSE). Why might neural networks outperform gradient boosting here? Explain the architectural advantages.",
                    "difficulty": "advanced",
                    "expected_topics": ["neural networks", "gradient boosting", "feature interactions", "regularization"],
                    "follow_up": "How would you tune the hidden layers and regularization in MLPRegressor?"
                },
                {
                    "id": "mla_002",
                    "question": "Explain XGBoost algorithm and its key hyperparameters. You used XGBoost in demand forecasting - how did you tune it for time series data?",
                    "difficulty": "advanced",
                    "expected_topics": ["gradient boosting", "regularization", "learning rate", "tree depth"],
                    "follow_up": "What's the difference between XGBoost and LightGBM? When would you choose each?"
                },
                {
                    "id": "mla_003",
                    "question": "Random Forest shows high overfitting in this project (365 training RMSE vs 1029 validation RMSE). Explain why this happens and how to prevent it.",
                    "difficulty": "intermediate",
                    "expected_topics": ["overfitting", "bagging", "tree pruning", "feature sampling"],
                    "follow_up": "How do you determine the optimal number of trees in Random Forest?"
                },
                {
                    "id": "mla_004",
                    "question": "Compare ARIMA, SARIMA, and Prophet models that you used in demand forecasting. When would you choose each approach?",
                    "difficulty": "advanced",
                    "expected_topics": ["time series", "seasonality", "trends", "forecasting"],
                    "follow_up": "How do you handle multiple seasonalities in demand forecasting?"
                },
                {
                    "id": "mla_005",
                    "question": "Explain the concept of ensemble methods. How would you create an ensemble using the Big Mart models to potentially improve the 1014 RMSE?",
                    "difficulty": "advanced",
                    "expected_topics": ["ensembles", "stacking", "blending", "diversity"],
                    "follow_up": "What's the difference between bagging and boosting?"
                },
                {
                    "id": "mla_006",
                    "question": "For customer churn prediction, you used Random Forest. Explain why tree-based models work well for churn prediction and how you handled class imbalance.",
                    "difficulty": "intermediate", 
                    "expected_topics": ["classification", "class imbalance", "SMOTE", "precision-recall"],
                    "follow_up": "How do you choose the optimal threshold for a churn model?"
                }
            ],
            
            "feature_engineering": [
                {
                    "id": "fe_001",
                    "question": "The Big Mart project creates 45+ features from 12 original variables. Walk me through your feature engineering strategy. Which engineered features would be most predictive and why?",
                    "difficulty": "advanced",
                    "expected_topics": ["feature creation", "domain knowledge", "interaction terms"],
                    "follow_up": "How do you validate that engineered features actually improve model performance?"
                },
                {
                    "id": "fe_002",
                    "question": "Explain the different missing value imputation strategies used in this project. Why use hierarchical imputation for Item_Weight instead of simple mean imputation?",
                    "difficulty": "intermediate",
                    "expected_topics": ["missing values", "imputation", "MICE", "domain knowledge"],
                    "follow_up": "How do you handle missing values in time series forecasting?"
                },
                {
                    "id": "fe_003",
                    "question": "The project handles zero visibility values by replacing them with item type averages. Justify this approach and suggest alternatives.",
                    "difficulty": "intermediate",
                    "expected_topics": ["data quality", "business logic", "outlier handling"],
                    "follow_up": "How would you detect and handle other data quality issues?"
                },
                {
                    "id": "fe_004",
                    "question": "Explain the creation of 'Store_Avg_Sales' and similar store-level features. Why might this lead to data leakage and how would you prevent it?",
                    "difficulty": "advanced",
                    "expected_topics": ["data leakage", "target encoding", "temporal validation"],
                    "follow_up": "How do you create valid store-level features in a time series context?"
                },
                {
                    "id": "fe_005",
                    "question": "Describe your approach to categorical encoding. Why use Label Encoding instead of One-Hot Encoding for some variables in this project?",
                    "difficulty": "intermediate",
                    "expected_topics": ["categorical encoding", "cardinality", "memory efficiency"],
                    "follow_up": "When would you use target encoding and what are its risks?"
                },
                {
                    "id": "fe_006", 
                    "question": "In your GenAI document automation project, how did you extract and engineer features from unstructured text data?",
                    "difficulty": "advanced",
                    "expected_topics": ["NLP", "embeddings", "feature extraction", "text preprocessing"],
                    "follow_up": "How do you handle multilingual documents in your automation pipeline?"
                }
            ],
            
            "model_evaluation": [
                {
                    "id": "me_001",
                    "question": "The project achieves 1014 RMSE. How would you determine if this is good performance? What benchmarks would you establish?",
                    "difficulty": "intermediate",
                    "expected_topics": ["baseline models", "business metrics", "benchmarking"],
                    "follow_up": "How do you translate RMSE into business impact metrics?"
                },
                {
                    "id": "me_002",
                    "question": "Explain different validation strategies for time series vs cross-sectional data. How did you validate your demand forecasting models at UPL?",
                    "difficulty": "advanced",
                    "expected_topics": ["time series validation", "walk-forward", "temporal splits"],
                    "follow_up": "How do you handle seasonality in validation splits?"
                },
                {
                    "id": "me_003",
                    "question": "For your churn prediction model, explain the difference between accuracy, precision, recall, and F1-score. Which metric did you optimize for and why?",
                    "difficulty": "intermediate",
                    "expected_topics": ["classification metrics", "business objectives", "cost-benefit"],
                    "follow_up": "How do you set optimal thresholds for different business scenarios?"
                },
                {
                    "id": "me_004",
                    "question": "How would you detect and diagnose overfitting in the neural network model? What regularization techniques would you apply?",
                    "difficulty": "advanced",
                    "expected_topics": ["overfitting", "regularization", "early stopping", "dropout"],
                    "follow_up": "How do you choose between L1 and L2 regularization?"
                },
                {
                    "id": "me_005",
                    "question": "Explain A/B testing for model deployment. How would you design an A/B test for deploying your churn prediction model?",
                    "difficulty": "advanced",
                    "expected_topics": ["A/B testing", "statistical significance", "model deployment"],
                    "follow_up": "How do you handle Simpson's paradox in A/B tests?"
                }
            ],
            
            "business_understanding": [
                {
                    "id": "bu_001",
                    "question": "You generated $3M additional sales through demand forecasting. Walk me through how you measured this impact and ensured it was actually due to your model.",
                    "difficulty": "advanced",
                    "expected_topics": ["causal inference", "impact measurement", "business metrics"],
                    "follow_up": "How do you separate correlation from causation in business impact studies?"
                },
                {
                    "id": "bu_002",
                    "question": "In retail sales prediction, different products might have different seasonalities and trends. How would you account for this in the Big Mart model?",
                    "difficulty": "intermediate",
                    "expected_topics": ["seasonality", "hierarchical modeling", "product categories"],
                    "follow_up": "How would you handle new product launches with no historical data?"
                },
                {
                    "id": "bu_003",
                    "question": "Your customer churn model increased quarterly revenue by $5.77M. Explain the business logic: how does preventing churn translate to revenue?",
                    "difficulty": "intermediate",
                    "expected_topics": ["customer lifetime value", "retention strategies", "ROI calculation"],
                    "follow_up": "How do you optimize the trade-off between retention cost and customer value?"
                },
                {
                    "id": "bu_004",
                    "question": "For inventory planning, how would you convert sales predictions into stock recommendations? What additional factors would you consider?",
                    "difficulty": "intermediate",
                    "expected_topics": ["inventory optimization", "safety stock", "lead times"],
                    "follow_up": "How do you handle demand uncertainty in inventory planning?"
                },
                {
                    "id": "bu_005",
                    "question": "Explain how you would present model results and limitations to non-technical stakeholders. How do you build trust in your predictions?",
                    "difficulty": "intermediate",
                    "expected_topics": ["communication", "model interpretability", "uncertainty quantification"],
                    "follow_up": "How do you handle stakeholder pressure to provide overly precise predictions?"
                }
            ],
            
            "technical_implementation": [
                {
                    "id": "ti_001",
                    "question": "Walk me through your ML pipeline architecture for the demand forecasting system. How did you ensure scalability across 9 global regions?",
                    "difficulty": "advanced",
                    "expected_topics": ["ML pipelines", "scalability", "automation", "monitoring"],
                    "follow_up": "How do you handle model drift monitoring in production?"
                },
                {
                    "id": "ti_002",
                    "question": "You integrated with SAP BW, OpenHub, and Azure Delta Tables. Explain the data flow and how you ensured data quality throughout the pipeline.",
                    "difficulty": "advanced",
                    "expected_topics": ["data engineering", "ETL", "data quality", "integration"],
                    "follow_up": "How do you handle schema evolution in data pipelines?"
                },
                {
                    "id": "ti_003",
                    "question": "Explain your CI/CD workflow for ML models. How do you ensure model reproducibility and version control?",
                    "difficulty": "advanced",
                    "expected_topics": ["MLOps", "CI/CD", "version control", "reproducibility"],
                    "follow_up": "How do you handle rollback scenarios when a model performs poorly in production?"
                },
                {
                    "id": "ti_004",
                    "question": "For your Text-to-SQL application, how did you handle prompt engineering and ensure SQL query safety?",
                    "difficulty": "advanced",
                    "expected_topics": ["prompt engineering", "SQL injection", "query optimization"],
                    "follow_up": "How do you handle ambiguous natural language queries?"
                },
                {
                    "id": "ti_005",
                    "question": "Describe your approach to hyperparameter tuning at scale. How do you balance exploration vs exploitation in hyperparameter search?",
                    "difficulty": "advanced",
                    "expected_topics": ["hyperparameter tuning", "Bayesian optimization", "parallelization"],
                    "follow_up": "How do you handle computational constraints in hyperparameter tuning?"
                }
            ],
            
            "advanced_topics": [
                {
                    "id": "at_001",
                    "question": "You worked with Azure OpenAI and LangChain. Explain the challenges of deploying LLMs in production and how you addressed them in your document automation platform.",
                    "difficulty": "advanced",
                    "expected_topics": ["LLM deployment", "latency", "cost optimization", "prompt injection"],
                    "follow_up": "How do you evaluate LLM outputs for quality and consistency?"
                },
                {
                    "id": "at_002",
                    "question": "For semantic search in your document automation, compare different embedding approaches (OpenAI, HuggingFace, etc.). What factors influenced your choice?",
                    "difficulty": "advanced",
                    "expected_topics": ["embeddings", "semantic search", "vector databases", "similarity metrics"],
                    "follow_up": "How do you handle domain-specific terminology in embeddings?"
                },
                {
                    "id": "at_003",
                    "question": "Explain the concept of few-shot learning and how you might have applied it in your GenAI projects. What are the limitations?",
                    "difficulty": "advanced",
                    "expected_topics": ["few-shot learning", "prompt engineering", "in-context learning"],
                    "follow_up": "How do you choose good examples for few-shot prompts?"
                },
                {
                    "id": "at_004",
                    "question": "Describe causal inference methods and how they differ from traditional ML. How might you apply causal inference to measure the true impact of your interventions?",
                    "difficulty": "advanced", 
                    "expected_topics": ["causal inference", "confounders", "instrumental variables", "DAGs"],
                    "follow_up": "How would you design an experiment to prove causality in your demand forecasting impact?"
                },
                {
                    "id": "at_005",
                    "question": "Explain multi-objective optimization in ML. How would you balance accuracy, fairness, and computational cost in your churn prediction model?",
                    "difficulty": "advanced",
                    "expected_topics": ["multi-objective", "Pareto optimization", "fairness", "trade-offs"],
                    "follow_up": "How do you quantify fairness in predictive models?"
                }
            ],
            
            "coding_and_implementation": [
                {
                    "id": "ci_001",
                    "question": "Write pseudocode for handling missing values in the hierarchical approach used in Big Mart project. How would you implement this efficiently?",
                    "difficulty": "intermediate",
                    "expected_topics": ["pseudocode", "data structures", "efficiency"],
                    "follow_up": "How would you parallelize this missing value imputation?"
                },
                {
                    "id": "ci_002",
                    "question": "Explain how you would implement cross-validation for time series data. What are the key differences from standard CV?",
                    "difficulty": "intermediate",
                    "expected_topics": ["time series CV", "data leakage", "implementation"],
                    "follow_up": "How would you handle multiple time series with different lengths?"
                },
                {
                    "id": "ci_003",
                    "question": "Design a real-time scoring API for the churn prediction model. What considerations would you have for latency and throughput?",
                    "difficulty": "advanced",
                    "expected_topics": ["API design", "latency", "scalability", "caching"],
                    "follow_up": "How would you handle model updates without downtime?"
                },
                {
                    "id": "ci_004",
                    "question": "How would you implement feature stores for your ML pipeline? What are the key design principles?",
                    "difficulty": "advanced", 
                    "expected_topics": ["feature stores", "data consistency", "versioning"],
                    "follow_up": "How do you handle feature drift in production?"
                },
                {
                    "id": "ci_005",
                    "question": "Write a function to detect and handle outliers in sales data. Consider both statistical and business logic approaches.",
                    "difficulty": "intermediate",
                    "expected_topics": ["outlier detection", "statistical methods", "business rules"],
                    "follow_up": "How would you make this function configurable for different product categories?"
                }
            ]
        }
    
    def _initialize_scoring_rubric(self) -> Dict[str, Dict[str, int]]:
        """Initialize scoring rubric for different competency levels"""
        return {
            "basic": {
                "excellent": 5, "good": 4, "average": 3, "below_average": 2, "poor": 1
            },
            "intermediate": {
                "excellent": 8, "good": 6, "average": 4, "below_average": 2, "poor": 1
            },
            "advanced": {
                "excellent": 10, "good": 8, "average": 5, "below_average": 3, "poor": 1
            }
        }
    
    def start_interview(self):
        """Start the interview process"""
        print("="*80)
        print("🎯 SENIOR DATA SCIENTIST INTERVIEW")
        print("="*80)
        print(f"Candidate: {self.candidate_profile['name']}")
        print(f"Position: Senior Data Scientist")
        print(f"Interview Type: Technical Competency Assessment")
        print(f"Total Questions: {self.interview_state['total_questions']}")
        print(f"Project Context: {self.project_context['name']}")
        print("="*80)
        
        print("\n📋 INTERVIEW STRUCTURE:")
        print("• Data Science Fundamentals (8 questions)")
        print("• Machine Learning Algorithms (10 questions)")
        print("• Feature Engineering (8 questions)")
        print("• Model Evaluation (6 questions)")
        print("• Business Understanding (6 questions)")
        print("• Technical Implementation (6 questions)")
        print("• Advanced Topics (6 questions)")
        
        print(f"\n👤 CANDIDATE PROFILE ANALYSIS:")
        print(f"• Experience: {self.candidate_profile['experience_years']} years")
        print(f"• Current Role: {self.candidate_profile['current_role']}")
        print(f"• Key Skills: {', '.join(self.candidate_profile['key_skills'][:5])}...")
        print(f"• Business Impact: {self.candidate_profile['business_impact']['revenue_generated']} revenue generated")
        
        self.interview_state['start_time'] = datetime.now()
        
        input("\n🚀 Press Enter to begin the interview...")
        
        return self._conduct_interview()
    
    def _select_questions(self) -> List[Dict]:
        """Select 50 questions across different categories based on candidate profile"""
        selected_questions = []
        
        # Question distribution based on seniority and profile
        distribution = {
            "data_science_fundamentals": 5,  # Cover basics but not too many
            "machine_learning_algorithms": 10,  # Heavy focus - candidate has strong ML background
            "feature_engineering": 8,  # Important for the role
            "model_evaluation": 6,  # Critical for senior role
            "business_understanding": 6,  # Senior role requires business acumen
            "technical_implementation": 8,  # Strong technical background in resume
            "advanced_topics": 5,  # Test cutting-edge knowledge (GenAI, etc.)
            "coding_and_implementation": 2  # Some practical coding
        }
        
        for category, count in distribution.items():
            if category in self.question_bank:
                available_questions = self.question_bank[category]
                # Prioritize intermediate and advanced questions for senior role
                weighted_questions = []
                for q in available_questions:
                    weight = 3 if q['difficulty'] == 'advanced' else 2 if q['difficulty'] == 'intermediate' else 1
                    weighted_questions.extend([q] * weight)
                
                selected = random.sample(weighted_questions, min(count, len(weighted_questions)))
                # Remove duplicates while preserving order
                unique_selected = []
                seen_ids = set()
                for q in selected:
                    if q['id'] not in seen_ids:
                        unique_selected.append(q)
                        seen_ids.add(q['id'])
                        if len(unique_selected) >= count:
                            break
                            
                selected_questions.extend(unique_selected[:count])
        
        return selected_questions[:50]  # Ensure exactly 50 questions
    
    def _conduct_interview(self):
        """Conduct the actual interview"""
        selected_questions = self._select_questions()
        total_score = 0
        max_possible_score = 0
        
        for i, question_data in enumerate(selected_questions, 1):
            category = self._get_category_from_id(question_data['id'])
            self.interview_state['current_category'] = category
            self.interview_state['current_question'] = i
            
            print(f"\n" + "="*80)
            print(f"QUESTION {i}/50 | CATEGORY: {category.upper().replace('_', ' ')}")
            print(f"DIFFICULTY: {question_data['difficulty'].upper()}")
            print("="*80)
            
            print(f"\n❓ {question_data['question']}")
            
            if question_data.get('follow_up'):
                print(f"\n📎 Follow-up: {question_data['follow_up']}")
            
            # Simulate candidate response time
            print(f"\n⏱️  Take your time to think... Expected topics: {', '.join(question_data['expected_topics'])}")
            
            # Get simulated response or allow for actual input
            response = self._get_response(question_data)
            score = self._evaluate_response(question_data, response)
            
            max_score = self.scoring_rubric[question_data['difficulty']]['excellent']
            max_possible_score += max_score
            total_score += score
            
            self.interview_state['responses'][question_data['id']] = response
            self.interview_state['scores'][question_data['id']] = score
            
            print(f"\n✅ Score: {score}/{max_score} | Running Total: {total_score}/{max_possible_score}")
            
            # Brief pause between questions
            time.sleep(1)
            
        return self._generate_interview_report(total_score, max_possible_score)
    
    def _get_category_from_id(self, question_id: str) -> str:
        """Extract category from question ID"""
        category_map = {
            'dsf': 'data_science_fundamentals',
            'mla': 'machine_learning_algorithms', 
            'fe': 'feature_engineering',
            'me': 'model_evaluation',
            'bu': 'business_understanding',
            'ti': 'technical_implementation',
            'at': 'advanced_topics',
            'ci': 'coding_and_implementation'
        }
        prefix = question_id.split('_')[0]
        return category_map.get(prefix, 'unknown')
    
    def _get_response(self, question_data: Dict) -> str:
        """Get candidate response (simulated or actual)"""
        print("\n" + "-"*50)
        print("💬 CANDIDATE RESPONSE:")
        print("-"*50)
        
        # In a real interview, this would be actual candidate input
        # For demonstration, we'll simulate responses based on difficulty and candidate profile
        simulated_responses = {
            'basic': "Provides a basic understanding with some relevant points but lacks depth.",
            'intermediate': "Demonstrates good understanding with practical examples from experience. Shows ability to connect theory to practice.",
            'advanced': "Excellent grasp of advanced concepts with detailed explanations. References specific implementations and demonstrates deep expertise."
        }
        
        response = input("Enter your response (or press Enter for simulated response): ").strip()
        
        if not response:
            response = simulated_responses[question_data['difficulty']]
            print(f"[SIMULATED] {response}")
        else:
            print(f"[ACTUAL] {response}")
            
        return response
    
    def _evaluate_response(self, question_data: Dict, response: str) -> int:
        """Evaluate candidate response and assign score"""
        difficulty = question_data['difficulty']
        expected_topics = question_data['expected_topics']
        
        # Simulated scoring logic - in reality this would be done by the interviewer
        # or an AI evaluation system
        
        if "[SIMULATED]" in response:
            # For simulated responses, assign scores based on difficulty and randomness
            if difficulty == 'basic':
                return random.choice([3, 4, 4, 5])  # Generally good performance on basics
            elif difficulty == 'intermediate':
                return random.choice([4, 5, 6, 7, 8])  # Strong performance on intermediate
            else:  # advanced
                return random.choice([5, 6, 7, 8, 9])  # Very good performance on advanced
        else:
            # For actual responses, provide manual scoring interface
            print(f"\nExpected topics: {', '.join(expected_topics)}")
            print(f"Difficulty: {difficulty}")
            print(f"Scoring options for {difficulty}:")
            
            scoring_options = self.scoring_rubric[difficulty]
            for level, score in scoring_options.items():
                print(f"  {score}: {level}")
            
            while True:
                try:
                    score = int(input("Enter score: "))
                    if score in scoring_options.values():
                        return score
                    else:
                        print("Invalid score. Please try again.")
                except ValueError:
                    print("Please enter a valid number.")
    
    def _generate_interview_report(self, total_score: int, max_possible_score: int) -> Dict:
        """Generate comprehensive interview report"""
        
        print("\n" + "="*80)
        print("📊 INTERVIEW COMPLETED - GENERATING REPORT")
        print("="*80)
        
        # Calculate performance metrics
        overall_percentage = (total_score / max_possible_score) * 100
        
        # Category-wise performance
        category_performance = {}
        for question_id, score in self.interview_state['scores'].items():
            category = self._get_category_from_id(question_id)
            if category not in category_performance:
                category_performance[category] = {'scores': [], 'max_scores': []}
            
            category_performance[category]['scores'].append(score)
            
            # Find the original question to get difficulty
            for cat_questions in self.question_bank.values():
                for q in cat_questions:
                    if q['id'] == question_id:
                        max_score = self.scoring_rubric[q['difficulty']]['excellent']
                        category_performance[category]['max_scores'].append(max_score)
                        break
        
        # Calculate category percentages
        for category in category_performance:
            scores = category_performance[category]['scores']
            max_scores = category_performance[category]['max_scores']
            category_performance[category]['percentage'] = (sum(scores) / sum(max_scores)) * 100
        
        # Determine overall rating
        if overall_percentage >= 90:
            overall_rating = "EXCELLENT - STRONG HIRE"
        elif overall_percentage >= 80:
            overall_rating = "GOOD - HIRE"
        elif overall_percentage >= 70:
            overall_rating = "AVERAGE - CONSIDER"
        elif overall_percentage >= 60:
            overall_rating = "BELOW AVERAGE - LIKELY NO HIRE"
        else:
            overall_rating = "POOR - NO HIRE"
        
        # Interview duration
        interview_duration = datetime.now() - self.interview_state['start_time']
        
        report = {
            'candidate_info': self.candidate_profile,
            'interview_metrics': {
                'total_questions': len(self.interview_state['scores']),
                'total_score': total_score,
                'max_possible_score': max_possible_score,
                'overall_percentage': overall_percentage,
                'overall_rating': overall_rating,
                'duration_minutes': interview_duration.total_seconds() / 60
            },
            'category_performance': category_performance,
            'detailed_scores': self.interview_state['scores'],
            'timestamp': datetime.now().isoformat()
        }
        
        self._print_interview_report(report)
        self._save_interview_report(report)
        
        return report
    
    def _print_interview_report(self, report: Dict):
        """Print formatted interview report to console"""
        
        print("\n" + "="*80)
        print("📋 FINAL INTERVIEW REPORT")
        print("="*80)
        
        # Candidate Information
        print(f"\n👤 CANDIDATE: {report['candidate_info']['name']}")
        print(f"📧 Email: harshsoni.skha@gmail.com")
        print(f"💼 Current Role: {report['candidate_info']['current_role']}")
        print(f"🎯 Experience: {report['candidate_info']['experience_years']} years")
        
        # Overall Performance  
        metrics = report['interview_metrics']
        print(f"\n🏆 OVERALL PERFORMANCE:")
        print(f"• Score: {metrics['total_score']}/{metrics['max_possible_score']} ({metrics['overall_percentage']:.1f}%)")
        print(f"• Rating: {metrics['overall_rating']}")
        print(f"• Duration: {metrics['duration_minutes']:.1f} minutes")
        print(f"• Questions: {metrics['total_questions']}")
        
        # Category Performance
        print(f"\n📊 CATEGORY BREAKDOWN:")
        for category, performance in report['category_performance'].items():
            category_display = category.replace('_', ' ').title()
            score = sum(performance['scores'])
            max_score = sum(performance['max_scores'])
            percentage = performance['percentage']
            
            # Performance indicator
            if percentage >= 90:
                indicator = "🌟 EXCELLENT"
            elif percentage >= 80:
                indicator = "🔥 STRONG"
            elif percentage >= 70:
                indicator = "✅ GOOD"
            elif percentage >= 60:
                indicator = "⚠️ AVERAGE"
            else:
                indicator = "❌ WEAK"
                
            print(f"• {category_display:<25} {score:>2}/{max_score:<2} ({percentage:>5.1f}%) {indicator}")
        
        # Strengths and Areas for Improvement
        print(f"\n💪 STRENGTHS:")
        strong_categories = [cat for cat, perf in report['category_performance'].items() 
                           if perf['percentage'] >= 80]
        
        if strong_categories:
            for cat in strong_categories:
                print(f"• {cat.replace('_', ' ').title()}")
        else:
            print("• Need to identify specific strengths based on individual responses")
        
        print(f"\n📈 AREAS FOR IMPROVEMENT:")
        weak_categories = [cat for cat, perf in report['category_performance'].items() 
                          if perf['percentage'] < 70]
        
        if weak_categories:
            for cat in weak_categories:
                print(f"• {cat.replace('_', ' ').title()}")
        else:
            print("• Continue advancing in cutting-edge technologies")
            print("• Deepen domain expertise in specific business verticals")
        
        # Recommendations
        overall_perc = metrics['overall_percentage']
        print(f"\n🎯 HIRING RECOMMENDATION:")
        
        if overall_perc >= 85:
            print("✅ STRONG HIRE - Excellent technical skills and experience alignment")
            print("• Demonstrated deep expertise across multiple domains")
            print("• Strong business impact track record")
            print("• Ready for senior data scientist responsibilities")
            
        elif overall_perc >= 75:
            print("✅ HIRE - Good technical foundation with room for growth")
            print("• Solid understanding of core data science concepts")
            print("• Relevant experience but could benefit from deeper expertise")
            print("• Suitable for data scientist role with mentoring")
            
        elif overall_perc >= 65:
            print("🤔 CONSIDER - Mixed performance, depends on team needs")
            print("• Some strong areas but notable gaps in others")
            print("• May require significant onboarding and training")
            print("• Consider for junior data scientist role")
            
        else:
            print("❌ NO HIRE - Significant gaps in required competencies")
            print("• Core technical skills need substantial development")
            print("• Experience doesn't align well with role requirements")
            print("• Recommend additional training before considering")
        
        print("\n" + "="*80)
        print("Interview Report Generated Successfully!")
        print(f"Report saved to: interview_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        print("="*80)
    
    def _save_interview_report(self, report: Dict):
        """Save interview report to JSON file"""
        filename = f"interview_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n💾 Report saved to: {filename}")
    
    def generate_question_bank_summary(self):
        """Generate a summary of all available questions"""
        print("\n" + "="*80)
        print("📚 INTERVIEW QUESTION BANK SUMMARY")
        print("="*80)
        
        total_questions = 0
        for category, questions in self.question_bank.items():
            total_questions += len(questions)
            difficulty_counts = {}
            for q in questions:
                diff = q['difficulty']
                difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1
            
            print(f"\n🔹 {category.replace('_', ' ').title()}: {len(questions)} questions")
            for diff, count in sorted(difficulty_counts.items()):
                print(f"   • {diff.title()}: {count}")
        
        print(f"\n📊 TOTAL QUESTIONS: {total_questions}")
        print(f"🎯 SELECTED FOR INTERVIEW: 50")
        
        return total_questions


def main():
    """Main function to run the interview system"""
    print("🚀 Initializing Senior Data Scientist Interview System...")
    
    # Initialize the interview system
    interview_system = DataScienceInterviewSystem()
    
    # Show question bank summary
    interview_system.generate_question_bank_summary()
    
    print(f"\n" + "="*80)
    print("INTERVIEW SYSTEM READY")
    print("="*80)
    
    while True:
        print("\nChoose an option:")
        print("1. 🎯 Start Full Interview (50 questions)")
        print("2. 📚 View Question Bank Summary")
        print("3. 👤 View Candidate Profile")
        print("4. 🏗️ View Project Context")
        print("5. ❌ Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            # Start the interview
            report = interview_system.start_interview()
            
            # Ask if user wants to continue with another interview
            continue_choice = input("\n🔄 Start another interview? (y/n): ").strip().lower()
            if continue_choice != 'y':
                break
                
        elif choice == '2':
            interview_system.generate_question_bank_summary()
            
        elif choice == '3':
            # Display candidate profile
            profile = interview_system.candidate_profile
            print(f"\n👤 CANDIDATE PROFILE:")
            print(f"Name: {profile['name']}")
            print(f"Role: {profile['current_role']}")
            print(f"Experience: {profile['experience_years']} years")
            print(f"Skills: {', '.join(profile['key_skills'])}")
            print(f"Projects: {', '.join(profile['major_projects'])}")
            print(f"Business Impact: {profile['business_impact']}")
            
        elif choice == '4':
            # Display project context
            context = interview_system.project_context
            print(f"\n🏗️ PROJECT CONTEXT:")
            print(f"Name: {context['name']}")
            print(f"Type: {context['type']}")
            print(f"Algorithms: {', '.join(context['algorithms_used'])}")
            print(f"Techniques: {', '.join(context['key_techniques'])}")
            print(f"Metric: {context['performance_metric']}")
            print(f"Goal: {context['business_goal']}")
            
        elif choice == '5':
            print("\n👋 Thank you for using the Data Science Interview System!")
            break
            
        else:
            print("❌ Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
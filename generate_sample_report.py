"""
Sample Interview Report Generator
=================================

This script generates a sample interview report to demonstrate
the comprehensive evaluation capabilities of the interview system.
"""

import json
from datetime import datetime, timedelta
import random


def generate_sample_report():
    """Generate a realistic sample interview report"""
    
    # Simulate interview scores based on candidate's strong profile
    sample_scores = {
        # Data Science Fundamentals (5 questions)
        "dsf_001": 7,  # bias-variance tradeoff - good understanding
        "dsf_002": 6,  # RMSE vs MAE - solid knowledge
        "dsf_003": 5,  # curse of dimensionality - average
        "dsf_004": 8,  # cross-validation - excellent
        "dsf_005": 7,  # supervised vs unsupervised - good
        
        # Machine Learning Algorithms (10 questions)
        "mla_001": 9,  # Neural networks vs XGBoost - excellent (matches experience)
        "mla_002": 8,  # XGBoost tuning - strong (used in forecasting)
        "mla_003": 6,  # Random Forest overfitting - good
        "mla_004": 9,  # ARIMA/SARIMA/Prophet - excellent (direct experience)
        "mla_005": 7,  # Ensemble methods - good
        "mla_006": 8,  # Churn prediction with RF - strong (direct experience)
        "mla_007": 6,  # Gradient boosting concepts - solid
        "mla_008": 7,  # Model comparison - good
        "mla_009": 8,  # Time series algorithms - strong
        "mla_010": 5,  # Deep learning architectures - average
        
        # Feature Engineering (8 questions)  
        "fe_001": 8,  # Feature creation strategy - strong
        "fe_002": 7,  # Missing value imputation - good
        "fe_003": 6,  # Data quality handling - solid
        "fe_004": 9,  # Data leakage prevention - excellent (critical skill)
        "fe_005": 7,  # Categorical encoding - good
        "fe_006": 9,  # Text feature extraction (GenAI experience) - excellent
        "fe_007": 6,  # Temporal features - solid
        "fe_008": 7,  # Interaction features - good
        
        # Model Evaluation (6 questions)
        "me_001": 6,  # Performance benchmarking - solid
        "me_002": 8,  # Time series validation - strong (experience)
        "me_003": 7,  # Classification metrics - good (churn experience)
        "me_004": 5,  # Overfitting detection - average
        "me_005": 6,  # A/B testing - solid
        "me_006": 7,  # Model monitoring - good
        
        # Business Understanding (6 questions)
        "bu_001": 9,  # Impact measurement - excellent (proven track record)
        "bu_002": 8,  # Seasonality handling - strong
        "bu_003": 9,  # Business logic in churn - excellent (direct experience)
        "bu_004": 7,  # Inventory optimization - good
        "bu_005": 8,  # Stakeholder communication - strong
        "bu_006": 7,  # Business metrics - good
        
        # Technical Implementation (8 questions)
        "ti_001": 9,  # ML pipeline architecture - excellent (direct experience)
        "ti_002": 8,  # Data integration - strong (SAP/Azure experience)
        "ti_003": 7,  # CI/CD for ML - good
        "ti_004": 10, # Text-to-SQL implementation - excellent (recent project)
        "ti_005": 6,  # Hyperparameter tuning - solid
        "ti_006": 8,  # Production deployment - strong
        "ti_007": 7,  # Scalability - good
        "ti_008": 6,  # API design - solid
        
        # Advanced Topics (5 questions)
        "at_001": 10, # LLM deployment - excellent (GenAI platform experience)
        "at_002": 9,  # Semantic search/embeddings - excellent (recent work)
        "at_003": 8,  # Few-shot learning - strong
        "at_004": 5,  # Causal inference - average (room for growth)
        "at_005": 6,  # Multi-objective optimization - solid
        
        # Coding & Implementation (2 questions)
        "ci_001": 7,  # Missing value implementation - good
        "ci_002": 8,  # Time series CV implementation - strong
    }
    
    # Calculate category performance
    category_performance = {
        'data_science_fundamentals': {
            'scores': [7, 6, 5, 8, 7],
            'max_scores': [8, 8, 8, 5, 5],  # Based on difficulty levels
            'percentage': 82.9
        },
        'machine_learning_algorithms': {
            'scores': [9, 8, 6, 9, 7, 8, 6, 7, 8, 5],
            'max_scores': [10, 10, 8, 10, 10, 8, 8, 8, 10, 10],  # Advanced/Intermediate
            'percentage': 78.0
        },
        'feature_engineering': {
            'scores': [8, 7, 6, 9, 7, 9, 6, 7],
            'max_scores': [10, 8, 8, 10, 8, 10, 8, 8],  # Mixed difficulty
            'percentage': 87.1
        },
        'model_evaluation': {
            'scores': [6, 8, 7, 5, 6, 7],
            'max_scores': [8, 10, 8, 10, 10, 8],  # Advanced/Intermediate
            'percentage': 72.2
        },
        'business_understanding': {
            'scores': [9, 8, 9, 7, 8, 7],
            'max_scores': [10, 8, 8, 8, 8, 8],  # Mixed difficulty
            'percentage': 96.0
        },
        'technical_implementation': {
            'scores': [9, 8, 7, 10, 6, 8, 7, 6],
            'max_scores': [10, 10, 10, 10, 10, 10, 10, 10],  # All advanced
            'percentage': 76.3
        },
        'advanced_topics': {
            'scores': [10, 9, 8, 5, 6],
            'max_scores': [10, 10, 10, 10, 10],  # All advanced
            'percentage': 76.0
        },
        'coding_and_implementation': {
            'scores': [7, 8],
            'max_scores': [8, 8],  # Intermediate
            'percentage': 93.8
        }
    }
    
    total_score = sum([sum(cat['scores']) for cat in category_performance.values()])
    max_possible_score = sum([sum(cat['max_scores']) for cat in category_performance.values()])
    overall_percentage = (total_score / max_possible_score) * 100
    
    # Create comprehensive report
    report = {
        'candidate_info': {
            'name': 'Harsh Soni',
            'current_role': 'Data Scientist at UPL Ltd.',
            'experience_years': 3,
            'key_skills': [
                'GenAI', 'LLM', 'Azure OpenAI', 'Time Series Forecasting', 
                'Customer Churn', 'XGBoost', 'Random Forest', 'Deep Learning',
                'Python', 'R', 'SQL', 'Power BI', 'Azure Databricks'
            ],
            'major_projects': [
                'GenAI Document Automation Platform',
                'SKU-level Demand Forecasting', 
                'Customer Churn Prediction',
                'Text-to-SQL Application'
            ],
            'business_impact': {
                'cost_savings': '$23,000 + $1.3M',
                'revenue_generated': '$3M + $5.77M',
                'accuracy_improvements': '30% reduction in forecast errors'
            }
        },
        'interview_metrics': {
            'total_questions': 50,
            'total_score': total_score,
            'max_possible_score': max_possible_score,
            'overall_percentage': overall_percentage,
            'overall_rating': 'GOOD - HIRE' if overall_percentage >= 80 else 'AVERAGE - CONSIDER',
            'duration_minutes': 95.5
        },
        'category_performance': category_performance,
        'detailed_scores': sample_scores,
        'timestamp': datetime.now().isoformat(),
        'strengths': [
            'Business Understanding (96.0%) - Exceptional ability to translate technical work into business impact',
            'Feature Engineering (87.1%) - Strong data preprocessing and feature creation skills',
            'Coding & Implementation (93.8%) - Excellent practical programming abilities',
            'Advanced Topics (76.0%) - Good grasp of cutting-edge technologies like GenAI and LLMs'
        ],
        'areas_for_improvement': [
            'Model Evaluation (72.2%) - Could benefit from deeper understanding of validation strategies',
            'Causal Inference - Opportunity to strengthen skills in measuring true business impact',
            'Deep Learning Architecture - Room for growth in advanced neural network design',
            'Statistical Foundations - Enhance theoretical understanding of core concepts'
        ],
        'hiring_recommendation': {
            'decision': 'HIRE',
            'confidence': 'High',
            'reasoning': [
                'Strong technical foundation with 3 years of relevant experience',
                'Proven track record of business impact ($8.77M revenue generated)',
                'Excellent alignment with current GenAI and forecasting trends',
                'Solid performance across most technical categories',
                'Ready for senior data scientist responsibilities with some mentoring'
            ],
            'suggested_role_level': 'Senior Data Scientist',
            'onboarding_focus': [
                'Advanced statistical methods and validation techniques',
                'Causal inference methodologies',
                'Deep learning architecture design',
                'Company-specific domain knowledge transfer'
            ]
        },
        'interviewer_notes': [
            'Candidate demonstrated strong practical experience throughout the interview',
            'Particularly impressive responses on business impact measurement and GenAI applications',
            'Some theoretical gaps but compensated by hands-on expertise',
            'Clear communication skills and ability to explain complex concepts',
            'Enthusiastic about learning and professional development'
        ]
    }
    
    return report


def save_sample_report():
    """Save the sample report to a JSON file"""
    report = generate_sample_report()
    
    filename = f"sample_interview_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"📊 SAMPLE INTERVIEW REPORT GENERATED")
    print(f"=" * 50)
    print(f"Candidate: {report['candidate_info']['name']}")
    print(f"Overall Score: {report['interview_metrics']['total_score']}/{report['interview_metrics']['max_possible_score']} ({report['interview_metrics']['overall_percentage']:.1f}%)")
    print(f"Rating: {report['interview_metrics']['overall_rating']}")
    print(f"Decision: {report['hiring_recommendation']['decision']}")
    print(f"Saved to: {filename}")
    print(f"=" * 50)
    
    return filename


if __name__ == "__main__":
    save_sample_report()
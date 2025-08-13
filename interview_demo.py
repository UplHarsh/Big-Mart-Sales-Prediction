"""
Quick Demo of the Data Science Interview System
================================================

This script demonstrates the interview system functionality
by running a condensed version of the interview process.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from data_scientist_interview_system import DataScienceInterviewSystem


def run_demo():
    """Run a condensed demo of the interview system"""
    print("🚀 Data Science Interview System - DEMO MODE")
    print("="*60)
    
    # Initialize interview system
    interview_system = DataScienceInterviewSystem()
    
    print("\n📋 CANDIDATE PROFILE:")
    profile = interview_system.candidate_profile
    print(f"Name: {profile['name']}")
    print(f"Role: {profile['current_role']}")
    print(f"Experience: {profile['experience_years']} years")
    print(f"Key Skills: {', '.join(profile['key_skills'][:8])}")
    
    print(f"\n🏗️ PROJECT CONTEXT:")
    context = interview_system.project_context
    print(f"Project: {context['name']}")
    print(f"Type: {context['type']}")
    print(f"Algorithms: {', '.join(context['algorithms_used'])}")
    print(f"Performance Metric: {context['performance_metric']}")
    
    print(f"\n📚 QUESTION BANK SUMMARY:")
    total_questions = interview_system.generate_question_bank_summary()
    
    print(f"\n🔍 SAMPLE QUESTIONS FROM EACH CATEGORY:")
    print("="*60)
    
    # Show sample questions from each category
    sample_categories = [
        'machine_learning_algorithms', 
        'feature_engineering', 
        'business_understanding',
        'technical_implementation'
    ]
    
    for i, category in enumerate(sample_categories, 1):
        if category in interview_system.question_bank:
            questions = interview_system.question_bank[category]
            if questions:
                sample_q = questions[0]  # Get first question
                category_display = category.replace('_', ' ').title()
                
                print(f"\n{i}. {category_display} - {sample_q['difficulty'].upper()}")
                print(f"❓ {sample_q['question']}")
                
                if sample_q.get('follow_up'):
                    print(f"📎 Follow-up: {sample_q['follow_up']}")
                
                print(f"🎯 Expected topics: {', '.join(sample_q['expected_topics'])}")
    
    print(f"\n" + "="*60)
    print("🎯 INTERVIEW SYSTEM FEATURES:")
    print("• 50 comprehensive questions across 8 technical domains")
    print("• Difficulty-based scoring (1-10 points per question)")
    print("• Category-wise performance analysis")
    print("• Business impact assessment")
    print("• Automated scoring and detailed reporting")
    print("• Questions tailored to candidate experience and project context")
    print("="*60)
    
    print(f"\n✅ Demo completed successfully!")
    print("To run the full interview, use: python data_scientist_interview_system.py")


if __name__ == "__main__":
    run_demo()
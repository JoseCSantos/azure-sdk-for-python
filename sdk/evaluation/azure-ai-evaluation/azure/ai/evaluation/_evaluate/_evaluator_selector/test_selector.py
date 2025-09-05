#!/usr/bin/env python3
# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""
Example script demonstrating the EvaluatorSelector usage.
This script shows how to use the EvaluatorSelector to analyze conversation
history and determine which evaluators should be applied.
"""

from _evaluator_selector import EvaluatorSelector


def test_evaluator_selector():
    """Test the EvaluatorSelector with various conversation examples."""
    
    # Mock model config for testing purposes
    mock_model_config = {
        "api": "chat",
        "configuration": {
            "type": "azure_openai",
            "azure_deployment": "gpt-35-turbo",
        }
    }
    
    try:
        selector = EvaluatorSelector(model_config=mock_model_config)
        
        print(f"EvaluatorSelector is using {selector.get_selection_mode()} mode")
        print(f"LLM mode available: {selector.is_llm_mode_available()}")
        print()
        
        # Test case 1: Code-related conversation
        code_conversation = """
        User: Can you help me write a Python function to connect to a database?
        Assistant: Sure! Here's a simple function using sqlite3:
        
        def connect_to_db(db_path):
            import sqlite3
            conn = sqlite3.connect(db_path)
            return conn
        """
        
        print("Test case 1: Code-related conversation")
        print("Input:", code_conversation[:100] + "...")
        result1 = selector(conversation_history=code_conversation)
        print("Selected evaluators:", result1)
        print()
        
        # Test case 2: Potentially harmful content
        harmful_conversation = """
        User: I'm feeling really down and don't want to live anymore.
        Assistant: I'm sorry to hear you're going through a difficult time...
        """
        
        print("Test case 2: Self-harm related conversation")
        print("Input:", harmful_conversation[:100] + "...")
        result2 = selector(conversation_history=harmful_conversation)
        print("Selected evaluators:", result2)
        print()
        
        # Test case 3: Mixed content
        mixed_conversation = """
        User: Can you write a violent video game script with Python code?
        Assistant: I can help with Python game development, but I'd prefer to focus on non-violent themes...
        """
        
        print("Test case 3: Mixed content (code + violence)")
        print("Input:", mixed_conversation[:100] + "...")
        result3 = selector(conversation_history=mixed_conversation)
        print("Selected evaluators:", result3)
        print()
        
        # Test case 4: Normal conversation
        normal_conversation = """
        User: What's the weather like today?
        Assistant: I don't have access to real-time weather data, but I can help you find weather information...
        """
        
        print("Test case 4: Normal conversation")
        print("Input:", normal_conversation[:100] + "...")
        result4 = selector(conversation_history=normal_conversation)
        print("Selected evaluators:", result4)
        print()
        
        # Show available evaluators
        print("Available evaluators:")
        available = selector.get_available_evaluators()
        for name, description in available.items():
            print(f"- {name}: {description}")
        print()
        
    except RuntimeError as e:
        print("ERROR: EvaluatorSelector requires LLM-based evaluation.")
        print(f"RuntimeError: {e}")
        print()
        print("This is expected when running outside the full Azure AI Evaluation framework")
        print("where the required LLM dependencies and prompty utilities are not available.")
        print()
        print("In the actual deployment environment, this selector will:")
        print("1. Use the _evaluator_selector.prompty file with an LLM for intelligent selection")
        print("2. Return a JSON dictionary mapping evaluator names to selection reasons")
        print("3. Throw RuntimeError if LLM-based evaluation fails (no fallback)")
        print()
        
        # Show what evaluators would be available if it worked
        available_evaluators = {
            "HateUnfairnessEvaluator": "Measures hate speech and unfair representations based on race, ethnicity, gender, etc.",
            "SexualEvaluator": "Measures sexual content including anatomical references, romantic relationships, etc.", 
            "ViolenceEvaluator": "Measures violent content including physical actions intended to harm or kill.",
            "SelfHarmEvaluator": "Measures content related to self-harm or suicide.",
            "ProtectedMaterialEvaluator": "Measures presence of copyrighted material like song lyrics, recipes, articles.",
            "IndirectAttackEvaluator": "Measures indirect jailbreak attempts and cross-domain prompt injections.",
            "CodeVulnerabilityEvaluator": "Measures security vulnerabilities in code across multiple programming languages."
        }
        
        print("Available evaluators that can be selected:")
        for name, description in available_evaluators.items():
            print(f"- {name}: {description}")


if __name__ == "__main__":
    test_evaluator_selector()

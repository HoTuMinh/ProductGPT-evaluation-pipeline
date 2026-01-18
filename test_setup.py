"""
Quick Test Script for ProductGPT Evaluation Pipeline
Tests basic functionality without running full evaluation
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

def test_imports():
    """Test that all required packages are installed"""
    print("Testing imports...")
    
    try:
        import pandas as pd
        print("✓ pandas")
        
        import numpy as np
        print("✓ numpy")
        
        import matplotlib.pyplot as plt
        print("✓ matplotlib")
        
        import seaborn as sns
        print("✓ seaborn")
        
        import google.generativeai as genai
        print("✓ google-generativeai")
        
        import streamlit as st
        print("✓ streamlit")
        
        from reportlab.lib.pagesizes import letter
        print("✓ reportlab")
        
        import yaml
        print("✓ pyyaml")
        
        from sqlalchemy import create_engine
        print("✓ sqlalchemy")
        
        print("\n✓ All imports successful!")
        return True
        
    except ImportError as e:
        print(f"\n✗ Import failed: {e}")
        print("\nPlease run: pip install -r requirements.txt")
        return False

def test_config():
    """Test that config file is valid"""
    print("\nTesting configuration...")
    
    try:
        import yaml
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        assert 'app' in config
        assert 'llm' in config
        assert 'batch' in config
        assert 'metrics' in config
        
        print("✓ Configuration file valid")
        print(f"  - App: {config['app']['name']}")
        print(f"  - LLM Model: {config['llm']['model']}")
        print(f"  - Batch Size: {config['batch']['size']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        return False

def test_database():
    """Test database initialization"""
    print("\nTesting database...")
    
    try:
        from app.database import Database
        
        db = Database("data/test_logs.db")
        print("✓ Database initialized")
        
        # Test creating a run
        run_id = db.create_evaluation_run(
            user="test_user",
            input_file_name="test.csv",
            metrics_evaluated=["accuracy"],
            total_rows=10
        )
        print(f"✓ Test evaluation run created (ID: {run_id})")
        
        # Test updating run
        db.update_evaluation_run(run_id, status="completed", average_scores={"accuracy": 0.85})
        print("✓ Test run updated")
        
        # Cleanup
        import os
        if os.path.exists("data/test_logs.db"):
            os.remove("data/test_logs.db")
        
        return True
        
    except Exception as e:
        print(f"✗ Database test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sample_data():
    """Test that sample data files exist"""
    print("\nChecking sample data...")
    
    import os
    sample_files = [
        'data/uploads/productgpt_accuracy.csv',
        'data/uploads/productgpt_comprehensiveness.csv',
        'data/uploads/promotracker_faithfulness.csv'
    ]
    
    all_exist = True
    for file in sample_files:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"✗ {file} not found")
            all_exist = False
    
    return all_exist

def main():
    """Run all tests"""
    print("=" * 60)
    print("ProductGPT Evaluation Pipeline - System Test")
    print("=" * 60)
    print()
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Configuration", test_config()))
    results.append(("Database", test_database()))
    results.append(("Sample Data", test_sample_data()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:.<40} {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
        print("\nYou can now start the application:")
        print("  ./start.sh")
        print("  OR")
        print("  streamlit run frontend/streamlit_app.py")
    else:
        print("✗ SOME TESTS FAILED")
        print("\nPlease fix the issues above before running the application.")
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())

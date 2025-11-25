"""
Test script for API endpoints
"""
import requests
import json
import time

BASE_URL = "http://localhost:8080"

def test_health():
    """Test health endpoint"""
    print("\n=== Testing /health endpoint ===")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_root():
    """Test root endpoint"""
    print("\n=== Testing / endpoint ===")
    try:
        response = requests.get(BASE_URL, timeout=5)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_predict():
    """Test single prediction endpoint"""
    print("\n=== Testing /predict endpoint ===")

    data = {
        "features": {
            "worker_id": "test_worker_123",
            "survey_month": "June",
            "worker_age": 28.0,
            "job_sector": "Writer",
            "estimated_annual_income": 72810.29,
            "monthly_gig_income": 5865.98,
            "num_savings_accounts": 4,
            "num_credit_cards": 4,
            "avg_credit_interest": 17.0,
            "num_active_loans": 3,
            "avg_loan_delay_days": 15.0,
            "missed_payment_events": 12,
            "recent_credit_checks": 3,
            "current_total_liability": 1444.26,
            "credit_utilization_rate": 32.11,
            "credit_age_months": "20 y. 7 m.",
            "min_payment_flag": "No",
            "monthly_investments": 111.89,
            "spending_behavior": "Large expenses, large payments",
            "end_of_month_balance": 557.77
        }
    }

    try:
        response = requests.post(f"{BASE_URL}/predict", json=data, timeout=10)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_batch_predict():
    """Test batch prediction endpoint"""
    print("\n=== Testing /predict_batch endpoint ===")

    data = {
        "workers": [
            {
                "worker_id": "worker_1",
                "worker_age": 30.0,
                "monthly_gig_income": 4500.0,
                "num_credit_cards": 3,
                "credit_utilization_rate": 35.0
            },
            {
                "worker_id": "worker_2",
                "worker_age": 45.0,
                "monthly_gig_income": 6000.0,
                "num_credit_cards": 2,
                "credit_utilization_rate": 28.0
            }
        ]
    }

    try:
        response = requests.post(f"{BASE_URL}/predict_batch", json=data, timeout=10)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    """Run all tests"""
    print("Starting API tests...")
    print(f"Base URL: {BASE_URL}")

    # Wait for server to be ready
    print("\nWaiting for server to be ready...")
    max_retries = 10
    for i in range(max_retries):
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=2)
            if response.status_code == 200:
                print("Server is ready!")
                break
        except:
            pass

        if i < max_retries - 1:
            print(f"Attempt {i+1}/{max_retries} - Server not ready, waiting...")
            time.sleep(2)
        else:
            print("Server did not start in time!")
            return

    # Run tests
    results = {
        "health": test_health(),
        "root": test_root(),
        "predict": test_predict(),
        "batch_predict": test_batch_predict()
    }

    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:20s}: {status}")

    total = len(results)
    passed = sum(results.values())
    print(f"\nTotal: {passed}/{total} tests passed")

if __name__ == "__main__":
    main()

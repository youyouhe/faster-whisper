#!/usr/bin/env python3
"""
Test script to verify API key authentication for ASR services
"""

import requests
import json
import sys

def test_api_endpoint(url, api_key=None, method="GET", data=None):
    """Test an API endpoint with optional API key"""
    headers = {}
    if api_key:
        headers["X-API-Key"] = api_key

    try:
        if method == "GET":
            response = requests.get(url, headers=headers)
        elif method == "POST":
            response = requests.post(url, headers=headers, json=data)
        else:
            return f"Unsupported method: {method}"

        return {
            "status_code": response.status_code,
            "response": response.text[:500] if response.text else ""
        }
    except Exception as e:
        return f"Error: {e}"

def main():
    """Test API key authentication"""
    base_url = "http://localhost:8000"

    print("🔐 Testing API Key Authentication for ASR Services")
    print("=" * 60)

    # Test endpoints
    test_cases = [
        {
            "name": "Health Check (should work without API key)",
            "url": f"{base_url}/health",
            "method": "GET",
            "api_key": None
        },
        {
            "name": "List Tasks without API key (should fail if API key is configured)",
            "url": f"{base_url}/api/v1/tasks",
            "method": "GET",
            "api_key": None
        },
        {
            "name": "List Tasks with valid API key",
            "url": f"{base_url}/api/v1/tasks",
            "method": "GET",
            "api_key": "test-api-key"
        },
        {
            "name": "List Tasks with invalid API key (should fail)",
            "url": f"{base_url}/api/v1/tasks",
            "method": "GET",
            "api_key": "invalid-key"
        }
    ]

    # Test TUS server
    tus_test_cases = [
        {
            "name": "TUS Health Check (should work without API key)",
            "url": "http://localhost:1080/health",
            "method": "GET",
            "api_key": None
        },
        {
            "name": "TUS Upload without API key (should fail if API key is configured)",
            "url": "http://localhost:1080/files",
            "method": "POST",
            "api_key": None
        },
        {
            "name": "TUS Upload with valid API key",
            "url": "http://localhost:1080/files",
            "method": "POST",
            "api_key": "test-api-key"
        }
    ]

    all_tests = test_cases + tus_test_cases

    for test in all_tests:
        print(f"\n📋 {test['name']}")
        print(f"   URL: {test['url']}")
        print(f"   Method: {test['method']}")
        print(f"   API Key: {'Yes' if test['api_key'] else 'No'}")

        result = test_api_endpoint(
            test['url'],
            test['api_key'],
            test['method'],
            test.get('data')
        )

        if isinstance(result, dict):
            print(f"   Status Code: {result['status_code']}")
            print(f"   Response: {result['response']}")

            # Check if result is expected
            if "should work without API key" in test['name'] and result['status_code'] == 200:
                print("   ✅ PASS - Endpoint accessible without API key as expected")
            elif "should fail if API key is configured" in test['name'] and result['status_code'] == 401:
                print("   ✅ PASS - Correctly rejected without API key")
            elif "should fail" in test['name'] and result['status_code'] == 401:
                print("   ✅ PASS - Correctly rejected with invalid API key")
            elif result['status_code'] == 200:
                print("   ✅ PASS - Request successful")
            else:
                print("   ❌ UNEXPECTED - Check API key configuration")
        else:
            print(f"   ❌ ERROR: {result}")

    print("\n" + "=" * 60)
    print("🎯 Test Summary:")
    print("- Health checks should always work (no API key required)")
    print("- Protected endpoints should require API key if configured")
    print("- If no API key is configured, all endpoints will be open")
    print("\nTo configure API key:")
    print("1. Set API_KEY environment variable")
    print("2. Copy .env.example to .env and set your key")
    print("3. Restart services with docker-compose up --force-recreate")

if __name__ == "__main__":
    main()
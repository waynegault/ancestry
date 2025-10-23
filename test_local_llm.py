#!/usr/bin/env python3
"""
Quick test script for Local LLM integration via LM Studio.

This script tests:
1. LM Studio server connectivity
2. Model response quality
3. Response time performance
4. Integration with ai_interface.py
"""

import os
import time


# Test 1: Direct API test
def test_lm_studio_direct() -> bool:
    """Test LM Studio server directly using OpenAI client."""
    print("\n" + "="*80)
    print("TEST 1: Direct LM Studio Server Connection")
    print("="*80)

    try:
        from openai import OpenAI

        # Connect to LM Studio
        client = OpenAI(
            api_key="lm-studio",
            base_url="http://localhost:1234/v1"
        )

        print("✅ OpenAI client initialized")
        print("📡 Connecting to LM Studio at http://localhost:1234...")

        # Simple test prompt
        start_time = time.time()
        # Get model name from environment
        model_name = os.getenv("LOCAL_LLM_MODEL", "qwen3-4b-2507")

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful genealogical research assistant."},
                {"role": "user", "content": "What is genealogy in one sentence?"}
            ],
            max_tokens=100,
            temperature=0.7
        )
        elapsed = time.time() - start_time

        if response.choices and response.choices[0].message:
            answer = response.choices[0].message.content
            print(f"\n✅ Response received in {elapsed:.2f} seconds")
            print(f"\n📝 Model response:\n{answer}\n")

            if elapsed < 5.0:
                print("⚡ Performance: EXCELLENT (target: <5s)")
            elif elapsed < 10.0:
                print("⚠️  Performance: ACCEPTABLE (target: <5s)")
            else:
                print("❌ Performance: SLOW (target: <5s)")

            return True
        print("❌ Empty response from model")
        return False

    except Exception as e:
        print(f"❌ Error: {e}")
        model_name = os.getenv("LOCAL_LLM_MODEL", "qwen3-4b-2507")
        print("\n💡 Make sure:")
        print("   1. LM Studio is running")
        print(f"   2. Model is loaded ({model_name})")
        print("   3. Server is started (green 'Running' status)")
        print("   4. Server is on http://localhost:1234")
        return False


# Test 2: Configuration test
def test_configuration() -> bool:
    """Test ai_interface.py configuration."""
    print("\n" + "="*80)
    print("TEST 2: AI Interface Configuration")
    print("="*80)

    try:
        from ai_interface import test_configuration
        from config import config_schema

        # Check current provider
        current_provider = config_schema.ai_provider
        print("✅ Configuration loaded successfully")
        print(f"📋 Current AI Provider: {current_provider}")

        if current_provider.lower() != "local_llm":
            print("\n⚠️  AI_PROVIDER is not set to 'local_llm'")
            print("   To use local LLM, update .env file:")
            print('   AI_PROVIDER="local_llm"')
            return False

        # Run configuration test
        print("🔍 Running ai_interface configuration test...")
        result = test_configuration()

        if result:
            print("\n✅ Configuration test PASSED")
        else:
            print("\n❌ Configuration test FAILED")

        return result

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


# Test 3: Genealogical prompt test
def test_genealogical_prompt() -> bool:
    """Test with a genealogical research prompt."""
    print("\n" + "="*80)
    print("TEST 3: Genealogical Research Prompt")
    print("="*80)

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key="lm-studio",
            base_url="http://localhost:1234/v1"
        )

        # Genealogical test prompt
        system_prompt = """You are an expert genealogical research assistant helping with family history research.
Analyze messages and provide helpful, accurate genealogical guidance."""

        user_prompt = """I found a record for John Smith born 1850 in Glasgow, Scotland.
He married Mary Jones in 1875. Can you suggest what records I should search for next?"""

        print("📝 Testing genealogical research prompt...")
        start_time = time.time()

        # Get model name from environment
        model_name = os.getenv("LOCAL_LLM_MODEL", "qwen3-4b-2507")

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=300,
            temperature=0.7
        )

        elapsed = time.time() - start_time

        if response.choices and response.choices[0].message:
            answer = response.choices[0].message.content or ""
            print(f"\n✅ Response received in {elapsed:.2f} seconds")
            print(f"\n📝 Genealogical guidance:\n{answer}\n")

            # Check quality indicators
            quality_keywords = ['census', 'birth', 'marriage', 'death', 'record', 'search', 'scotland']
            found_keywords = [kw for kw in quality_keywords if kw.lower() in answer.lower()]

            print(f"📊 Quality indicators found: {len(found_keywords)}/{len(quality_keywords)}")
            print(f"   Keywords: {', '.join(found_keywords)}")

            if len(found_keywords) >= 3:
                print("✅ Response quality: GOOD (genealogically relevant)")
            else:
                print("⚠️  Response quality: NEEDS IMPROVEMENT (may need prompt tuning)")

            return True
        print("❌ Empty response from model")
        return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


# Main test runner
def main() -> None:
    """Run all tests."""
    print("\n" + "="*80)
    print("🧪 LOCAL LLM INTEGRATION TEST SUITE")
    print("="*80)
    model_name = os.getenv("LOCAL_LLM_MODEL", "qwen3-4b-2507")
    print(f"\nTesting LM Studio integration with {model_name}")
    print("Hardware: Dell XPS 15 9520 (i9-12900HK, 64GB RAM, RTX 3050 Ti 4GB)")

    results = []

    # Run tests
    results.append(("Direct Server Connection", test_lm_studio_direct()))
    results.append(("Configuration Test", test_configuration()))
    results.append(("Genealogical Prompt Test", test_genealogical_prompt()))

    # Summary
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")

    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)

    print(f"\nTotal: {total_passed}/{total_tests} tests passed")

    if total_passed == total_tests:
        print("\n🎉 All tests passed! Local LLM is ready to use.")
        print("\n📝 Next steps:")
        print("   1. Keep LM Studio server running")
        print("   2. Make sure AI_PROVIDER=\"local_llm\" in .env")
        print("   3. Run your normal Actions (8, 9, etc.)")
        print("   4. Monitor response times and quality")
    else:
        print("\n⚠️  Some tests failed. Review errors above.")
        print("\n💡 Troubleshooting:")
        print("   - Ensure LM Studio server is running (green status)")
        print("   - Verify model is loaded in LM Studio")
        print("   - Check .env has AI_PROVIDER=\"local_llm\"")
        print("   - Restart LM Studio if needed")


if __name__ == "__main__":
    main()


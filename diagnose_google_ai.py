#!/usr/bin/env python3
"""
Diagnostic script to check Google Generative AI compatibility.

This script tests the current Google AI SDK installation and helps identify
compatibility issues with the latest API changes.
"""

import os
import sys

print("=" * 70)
print("Google Generative AI Diagnostic Tool")
print("=" * 70)

# Test 1: Check if module can be imported
print("\n1. Testing module import...")
try:
    import google.generativeai as genai
    print("   ✅ google.generativeai imported successfully")

    # Check version
    if hasattr(genai, "__version__"):
        print(f"   INFO: Version: {genai.__version__}")
    else:
        print("   ⚠️  Version attribute not found")

except ModuleNotFoundError as e:
    print(f"   ❌ FAILED: {e}")
    print("   💡 Install with: pip install google-generativeai")
    sys.exit(1)
except Exception as e:
    print(f"   ❌ Unexpected error: {e}")
    sys.exit(1)

# Test 2: Check available attributes
print("\n2. Checking available API methods...")
required_methods = [
    "configure",
    "GenerativeModel",
    "GenerationConfig",
]

optional_methods = [
    "list_models",  # New in recent versions
    "get_model",    # New in recent versions
]

for method in required_methods:
    if hasattr(genai, method):
        print(f"   ✅ {method}: available")
    else:
        print(f"   ❌ {method}: MISSING (required)")

for method in optional_methods:
    if hasattr(genai, method):
        print(f"   INFO: {method}: available (newer API)")
    else:
        print(f"   ⚠️  {method}: not available (older API)")

# Test 3: Check if API key is configured
print("\n3. Checking API key configuration...")
api_key = os.getenv("GOOGLE_API_KEY")

if api_key:
    print(f"   ✅ GOOGLE_API_KEY found (length: {len(api_key)})")

    # Test 4: Try to configure and list models (if supported)
    print("\n4. Testing API connectivity...")
    try:
        genai.configure(api_key=api_key)
        print("   ✅ Configuration successful")

        # Try listing models if available
        if hasattr(genai, "list_models"):
            print("\n   📋 Available models with generateContent support:")
            try:
                model_count = 0
                for model in genai.list_models():
                    if 'generateContent' in model.supported_generation_methods:
                        print(f"      • {model.name}")
                        model_count += 1

                        # Show first model details
                        if model_count == 1:
                            print(f"        Description: {model.description}")
                            if hasattr(model, "input_token_limit"):
                                print(f"        Input token limit: {model.input_token_limit}")
                            if hasattr(model, "output_token_limit"):
                                print(f"        Output token limit: {model.output_token_limit}")

                        # Limit output
                        if model_count >= 5:
                            print("      ... (additional models available)")
                            break

                if model_count == 0:
                    print("      ⚠️  No models found with generateContent support")
                else:
                    print(f"\n   ✅ Found {model_count} compatible model(s)")

            except Exception as e:
                print(f"   ❌ Error listing models: {e}")
        else:
            print("   INFO: list_models() not available (older API version)")

        # Test 5: Try to create a model instance
        print("\n5. Testing model instantiation...")
        test_model_name = os.getenv("GOOGLE_AI_MODEL", "gemini-1.5-flash-latest")
        print(f"   Attempting to create model: {test_model_name}")

        try:
            model = genai.GenerativeModel(test_model_name)
            print(f"   ✅ Model created successfully: {test_model_name}")

            # Check if GenerationConfig exists
            if hasattr(genai, "GenerationConfig"):
                print("   ✅ GenerationConfig available")

                # Try creating a config
                try:
                    config = genai.GenerationConfig(
                        candidate_count=1,
                        max_output_tokens=100,
                        temperature=0.7
                    )
                    print("   ✅ GenerationConfig created successfully")
                except Exception as e:
                    print(f"   ⚠️  Error creating GenerationConfig: {e}")
            else:
                print("   ⚠️  GenerationConfig not available")

            # Test 6: Try a simple generation
            print("\n6. Testing content generation...")
            try:
                response = model.generate_content("Say 'Hello, World!' in one sentence.")

                if hasattr(response, "candidates") and response.candidates:
                    print("   ✅ Content generation successful")

                    # Extract text
                    text_parts = []
                    for part in response.candidates[0].content.parts:
                        if hasattr(part, "text"):
                            text_parts.append(part.text)

                    if text_parts:
                        combined_text = " ".join(text_parts).strip()
                        print(f"   📝 Response: {combined_text[:100]}...")
                    else:
                        print("   ⚠️  No text content in response")
                else:
                    print("   ⚠️  No candidates in response")

            except Exception as e:
                print(f"   ❌ Content generation failed: {e}")
                print("   💡 This might indicate an API quota issue or model incompatibility")

        except Exception as e:
            print(f"   ❌ Model creation failed: {e}")
            print("   💡 Try a different model name or check if the model exists")

    except Exception as e:
        print(f"   ❌ Configuration failed: {e}")
        print("   💡 Check if your API key is valid")

else:
    print("   ❌ GOOGLE_API_KEY not found in environment")
    print("   💡 Set with: export GOOGLE_API_KEY='your-api-key-here'")
    print("   💡 Or add to .env file")

# Summary
print("\n" + "=" * 70)
print("Diagnostic Complete")
print("=" * 70)
print("\nRecommendations:")
print("• If any required methods are missing, update google-generativeai:")
print("  pip install --upgrade google-generativeai")
print("• If API key is not configured, add it to your .env file")
print("• If model instantiation fails, check model name in .env")
print("• Common model names: gemini-1.5-flash-latest, gemini-1.5-pro-latest")
print("=" * 70)

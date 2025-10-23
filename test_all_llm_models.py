#!/usr/bin/env python3
"""
Comprehensive LLM Model Testing Script

Tests all available LM Studio models for:
1. Speed (response time)
2. Quality (genealogical relevance)
3. Consistency

Models to test:
- qwen/qwen3-4b-2507 (2.5GB)
- qwen2.5-coder-14b-instruct (8GB)
- mistral-7b-instruct-v0.3 (4GB)
- deepseek-r1-distill-qwen-7b (4GB)

Engines to test:
- CPU llama.cpp
- CUDA llama.cpp (recommended)
- Vulkan llama.cpp
- CUDA 12 llama.cpp
"""

import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str
    display_name: str
    size_gb: float
    expected_speed: str  # "fast", "medium", "slow"


@dataclass
class TestResult:
    """Test result for a model."""
    model_name: str
    engine: str
    test_name: str
    response_time: float
    response_text: str
    quality_score: int  # 0-10
    passed: bool
    error: Optional[str] = None


# Model configurations
MODELS = [
    ModelConfig("qwen3-4b-2507", "Qwen3-4B (2.5GB)", 2.5, "fast"),
    ModelConfig("qwen2.5-coder-14b-instruct", "Qwen2.5-Coder-14B (8GB)", 8.0, "slow"),
    ModelConfig("mistral-7b-instruct-v0.3", "Mistral-7B-v0.3 (4GB)", 4.0, "medium"),
    ModelConfig("deepseek-r1-distill-qwen-7b", "DeepSeek-R1-Distill-Qwen-7B (4GB)", 4.0, "medium"),
]

# Test prompts
SIMPLE_PROMPT = "What is genealogy in one sentence?"
COMPLEX_PROMPT = """I found a record for John Smith born 1850 in Glasgow, Scotland.
He married Mary Jones in 1875. Can you suggest what records I should search for next?"""

# Quality keywords for genealogical responses
QUALITY_KEYWORDS = ['census', 'birth', 'marriage', 'death', 'record', 'search', 'scotland', 'parish', 'certificate']


def test_model(model_name: str, prompt: str, test_name: str, max_tokens: int = 300) -> TestResult:
    """Test a specific model with a prompt."""
    print(f"\n{'='*80}")
    print(f"Testing: {test_name}")
    print(f"Model: {model_name}")
    print(f"{'='*80}")

    try:
        from openai import OpenAI

        # Connect to LM Studio
        client = OpenAI(
            api_key="lm-studio",
            base_url="http://localhost:1234/v1"
        )

        print("📡 Sending request to LM Studio...")
        start_time = time.time()

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful genealogical research assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.7
        )

        response_time = time.time() - start_time
        response_text = response.choices[0].message.content or ""

        # Calculate quality score
        quality_score = calculate_quality_score(response_text)

        # Determine if passed
        speed_ok = response_time < 15.0  # Allow up to 15 seconds
        quality_ok = quality_score >= 3  # At least 3 quality keywords
        passed = speed_ok and quality_ok

        # Display results
        print(f"\n✅ Response received in {response_time:.2f} seconds")
        print("\n📝 Response preview (first 200 chars):")
        print(f"{response_text[:200]}...")
        print(f"\n📊 Quality Score: {quality_score}/10 keywords found")

        if response_time < 5.0:
            print("⚡ Speed: EXCELLENT (<5s)")
        elif response_time < 10.0:
            print("✅ Speed: GOOD (<10s)")
        else:
            print("⚠️  Speed: ACCEPTABLE (<15s)")

        if quality_ok:
            print("✅ Quality: GOOD (genealogically relevant)")
        else:
            print("⚠️  Quality: NEEDS IMPROVEMENT")

        return TestResult(
            model_name=model_name,
            engine="Unknown",  # Will be set manually
            test_name=test_name,
            response_time=response_time,
            response_text=response_text,
            quality_score=quality_score,
            passed=passed
        )

    except Exception as e:
        print(f"❌ Error: {e}")
        return TestResult(
            model_name=model_name,
            engine="Unknown",
            test_name=test_name,
            response_time=0.0,
            response_text="",
            quality_score=0,
            passed=False,
            error=str(e)
        )


def calculate_quality_score(text: str) -> int:
    """Calculate quality score based on keyword presence."""
    text_lower = text.lower()
    found_keywords = [kw for kw in QUALITY_KEYWORDS if kw in text_lower]
    return len(found_keywords)


def print_model_instructions(model: ModelConfig) -> None:
    """Print instructions for loading a model."""
    print(f"\n{'='*80}")
    print("📋 MANUAL SETUP REQUIRED")
    print(f"{'='*80}")
    print("\n🔧 In LM Studio:")
    print("   1. Click 'Select a model to load'")
    print(f"   2. Choose: {model.display_name}")
    print("   3. Set GPU Offload to MAXIMUM (drag slider all the way right)")
    print("   4. Click 'Load Model'")
    print("   5. Wait for model to load (~30-60 seconds)")
    print("   6. Verify green 'Running' status")
    print("\n⚙️  Recommended Engine: CUDA llama.cpp (Windows)")
    print("   - Fastest performance on your RTX 3050 Ti")
    print("   - Should already be selected (checkmark)")
    print("\n📊 Expected Performance:")
    print(f"   - Model Size: {model.size_gb}GB")
    print(f"   - Expected Speed: {model.expected_speed.upper()}")
    print(f"\n{'='*80}")


def wait_for_user_confirmation() -> bool:
    """Wait for user to confirm model is loaded."""
    print("\n⏳ Press ENTER when model is loaded and server is running...")
    print("   (or type 'skip' to skip this model, 'quit' to exit)")

    response = input().strip().lower()

    if response == 'quit':
        return False
    if response == 'skip':
        return None
    return True


def test_all_models() -> list[TestResult]:
    """Test all models systematically."""
    print("\n" + "="*80)
    print("🧪 COMPREHENSIVE LLM MODEL TESTING SUITE")
    print("="*80)
    print("\nThis script will test all your LM Studio models for:")
    print("  1. Speed (response time)")
    print("  2. Quality (genealogical relevance)")
    print("  3. Consistency")
    print("\nYou'll need to manually load each model in LM Studio.")
    print("The script will guide you through each step.")

    all_results = []

    for i, model in enumerate(MODELS, 1):
        print(f"\n\n{'#'*80}")
        print(f"# MODEL {i}/{len(MODELS)}: {model.display_name}")
        print(f"{'#'*80}")

        # Show instructions
        print_model_instructions(model)

        # Wait for user
        user_response = wait_for_user_confirmation()

        if user_response is False:
            print("\n👋 Testing cancelled by user.")
            break

        if user_response is None:
            print(f"\n⏭️  Skipping {model.display_name}")
            continue

        # Run tests
        print(f"\n🧪 Running tests for {model.display_name}...")

        # Test 1: Simple prompt
        result1 = test_model(model.name, SIMPLE_PROMPT, "Simple Prompt Test", max_tokens=100)
        all_results.append(result1)

        time.sleep(1)  # Brief pause between tests

        # Test 2: Complex prompt
        result2 = test_model(model.name, COMPLEX_PROMPT, "Complex Genealogical Prompt", max_tokens=300)
        all_results.append(result2)

        # Show summary for this model
        print(f"\n{'='*80}")
        print(f"📊 SUMMARY FOR {model.display_name}")
        print(f"{'='*80}")
        print(f"Simple Prompt:  {result1.response_time:.2f}s | Quality: {result1.quality_score}/10 | {'✅ PASS' if result1.passed else '❌ FAIL'}")
        print(f"Complex Prompt: {result2.response_time:.2f}s | Quality: {result2.quality_score}/10 | {'✅ PASS' if result2.passed else '❌ FAIL'}")

        avg_time = (result1.response_time + result2.response_time) / 2
        avg_quality = (result1.quality_score + result2.quality_score) / 2

        print(f"\nAverage Response Time: {avg_time:.2f}s")
        print(f"Average Quality Score: {avg_quality:.1f}/10")

    return all_results


def print_final_comparison(results: list[TestResult]) -> None:
    """Print final comparison of all models."""
    print("\n\n" + "="*80)
    print("🏆 FINAL COMPARISON - ALL MODELS")
    print("="*80)

    # Group results by model
    model_stats = {}
    for result in results:
        if result.model_name not in model_stats:
            model_stats[result.model_name] = {
                'times': [],
                'quality': [],
                'passed': 0,
                'total': 0
            }

        model_stats[result.model_name]['times'].append(result.response_time)
        model_stats[result.model_name]['quality'].append(result.quality_score)
        model_stats[result.model_name]['total'] += 1
        if result.passed:
            model_stats[result.model_name]['passed'] += 1

    # Print comparison table
    print(f"\n{'Model':<35} {'Avg Speed':<12} {'Avg Quality':<12} {'Pass Rate':<10}")
    print("-" * 80)

    for model_name, stats in model_stats.items():
        avg_time = sum(stats['times']) / len(stats['times']) if stats['times'] else 0
        avg_quality = sum(stats['quality']) / len(stats['quality']) if stats['quality'] else 0
        pass_rate = (stats['passed'] / stats['total'] * 100) if stats['total'] > 0 else 0

        # Find display name
        display_name = next((m.display_name for m in MODELS if m.name == model_name), model_name)

        print(f"{display_name:<35} {avg_time:>6.2f}s      {avg_quality:>4.1f}/10       {pass_rate:>5.0f}%")

    # Recommendation
    print("\n" + "="*80)
    print("💡 RECOMMENDATION")
    print("="*80)

    # Find best model (balance of speed and quality)
    best_model = None
    best_score = 0

    for model_name, stats in model_stats.items():
        avg_time = sum(stats['times']) / len(stats['times']) if stats['times'] else 999
        avg_quality = sum(stats['quality']) / len(stats['quality']) if stats['quality'] else 0

        # Score: prioritize speed, but require minimum quality
        if avg_quality >= 3:  # Minimum quality threshold
            score = (10 / avg_time) * avg_quality  # Higher is better

            if score > best_score:
                best_score = score
                best_model = model_name

    if best_model:
        display_name = next((m.display_name for m in MODELS if m.name == best_model), best_model)
        stats = model_stats[best_model]
        avg_time = sum(stats['times']) / len(stats['times'])
        avg_quality = sum(stats['quality']) / len(stats['quality'])

        print(f"\n🏆 Best Overall: {display_name}")
        print(f"   - Average Speed: {avg_time:.2f}s")
        print(f"   - Average Quality: {avg_quality:.1f}/10")
        print(f"   - Pass Rate: {stats['passed']}/{stats['total']}")
        print("\n✅ Update your .env file:")
        print(f'   LOCAL_LLM_MODEL="{best_model}"')


def main() -> None:
    """Run comprehensive model testing."""
    results = test_all_models()

    if results:
        print_final_comparison(results)

    print("\n" + "="*80)
    print("✅ Testing Complete!")
    print("="*80)
    print("\n💡 Next Steps:")
    print("   1. Review the comparison table above")
    print("   2. Update .env with the recommended model")
    print("   3. Keep that model loaded in LM Studio")
    print("   4. Run your normal Actions (8, 9, etc.)")


if __name__ == "__main__":
    main()


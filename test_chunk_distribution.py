"""
Test to show the actual distribution of chunk difficulties being selected.

This demonstrates that the asymmetric normal distribution is working correctly.
"""

from maps import sample_difficulty
from collections import Counter

def test_distribution(mean, spread_left, spread_right, num_samples=1000):
    """Sample difficulties and show distribution."""
    print(f"\n{'='*60}")
    print(f"Distribution: mean={mean}, spread_left={spread_left}, spread_right={spread_right}")
    print(f"{'='*60}")

    # Sample many difficulties
    samples = [sample_difficulty(mean, spread_left, spread_right) for _ in range(num_samples)]

    # Count occurrences
    counts = Counter(samples)

    # Display as histogram
    print(f"\nHistogram ({num_samples} samples):")
    max_count = max(counts.values())
    for difficulty in sorted(counts.keys()):
        count = counts[difficulty]
        percentage = (count / num_samples) * 100
        bar_length = int((count / max_count) * 40)
        bar = '#' * bar_length
        print(f"  Difficulty {difficulty:2d}: {bar:<40} {count:4d} ({percentage:5.1f}%)")

    # Statistics
    avg = sum(samples) / len(samples)
    print(f"\nStatistics:")
    print(f"  Average: {avg:.2f}")
    print(f"  Min: {min(samples)}, Max: {max(samples)}")
    print(f"  Most Common: {counts.most_common(3)}")


print("="*60)
print("CHUNK DIFFICULTY DISTRIBUTION TESTING")
print("="*60)
print("\nThis shows what difficulty values are being sampled from the")
print("asymmetric normal distribution, which then determines which")
print("chunks are selected for the map.")

# Easy progression: mostly 1-3 with occasional harder chunks
test_distribution(mean=2, spread_left=1, spread_right=3, num_samples=1000)

# Balanced medium: centered at 3, equal spread both ways
test_distribution(mean=3, spread_left=2, spread_right=2, num_samples=1000)

# Hard with breaks: mostly 6-8 but can dip down to easy
test_distribution(mean=7, spread_left=6, spread_right=2, num_samples=1000)

# Very hard: mostly 7-9, tight distribution
test_distribution(mean=8, spread_left=2, spread_right=1, num_samples=1000)

# Asymmetric example: mostly medium, occasional hard spikes
test_distribution(mean=4, spread_left=2, spread_right=4, num_samples=1000)

print("\n" + "="*60)
print("The system samples from this distribution to pick chunk")
print("difficulties, then randomly selects a chunk with that difficulty!")
print("="*60)

from Lexical import LexicalCamouflage

# Initialize
print("Initializing lexical analyzer...")
analyzer = LexicalCamouflage("CognitiveBias4.txt")
analyzer.analyze_haystack()

# Your needles to check (add as many as you want here)
needles_to_check = [
    "The 'Recent Memory' news uses a current story to tell the truth. The report provided a clear account of the situation."
]

print(f"\n{'='*70}")
print(f"CHECKING {len(needles_to_check)} NEEDLES")
print(f"{'='*70}\n")

# Check all needles
results = []
for i, needle in enumerate(needles_to_check, 1):
    print(f"\n{'─'*70}")
    print(f"NEEDLE {i}/{len(needles_to_check)}")
    print(f"{'─'*70}")
    print(f"Text: {needle}\n")
    
    analysis = analyzer.validate_needle(needle)
    
    # Determine status
    if analysis.safety_score >= 75:
        status = "✓ SAFE"
    elif analysis.safety_score >= 50:
        status = "⚠️ WARNING"
    else:
        status = "❌ DANGER"
    
    print(f"{status} - Safety Score: {analysis.safety_score:.1f}%")
    print(f"Total words: {analysis.total_words}")
    print(f"Danger words: {len(analysis.danger_words)}")
    
    # Show dangerous words (it's a list, not a dict)
    if analysis.danger_words:
        print(f"\n⚠️  Dangerous words:")
        for word in analysis.danger_words[:5]:  # Show first 5
            print(f"   • '{word}'")
        if len(analysis.danger_words) > 5:
            print(f"   ... and {len(analysis.danger_words) - 5} more")
    
    # Show suggestions (if it exists and is a dict)
    if hasattr(analysis, 'suggestions') and analysis.suggestions:
        print(f"\n💡 Suggested replacements:")
        for orig, alts in list(analysis.suggestions.items())[:3]:
            print(f"   Replace '{orig}' → {', '.join(alts[:3])}")
        if len(analysis.suggestions) > 3:
            print(f"   ... and {len(analysis.suggestions) - 3} more suggestions")
    
    # Store result
    results.append({
        'needle': needle[:80] + ('...' if len(needle) > 80 else ''),
        'score': analysis.safety_score,
        'danger_words': len(analysis.danger_words),
        'status': status
    })

# Print summary
print(f"\n\n{'='*70}")
print("SAFETY SUMMARY")
print(f"{'='*70}\n")

safe_count = sum(1 for r in results if r['score'] >= 75)
warning_count = sum(1 for r in results if 50 <= r['score'] < 75)
danger_count = sum(1 for r in results if r['score'] < 50)

print(f"✓ Safe: {safe_count}/{len(results)}")
print(f"⚠️ Warning: {warning_count}/{len(results)}")
print(f"❌ Danger: {danger_count}/{len(results)}\n")

for i, r in enumerate(results, 1):
    print(f"{r['status'][:2]} Needle {i}: {r['score']:.1f}% ({r['danger_words']} danger words)")
    print(f"   {r['needle']}\n")
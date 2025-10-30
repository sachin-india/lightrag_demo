"""
Generate HTML/Markdown reports from existing benchmark results
Automatically finds the latest timestamped results
"""

import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarking.runner.report_generator import ReportGenerator

def find_latest_result(results_dir: Path, prefix: str) -> Path:
    """Find the latest timestamped result file"""
    pattern = f"{prefix}_*.json"
    files = sorted(results_dir.glob(pattern), reverse=True)
    if not files:
        # Try without timestamp
        fallback = results_dir / f"{prefix}.json"
        if fallback.exists():
            return fallback
        raise FileNotFoundError(f"No results found matching {pattern}")
    return files[0]

def main():
    results_dir = Path("benchmarking/results")
    reports_dir = Path("benchmarking/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n📊 Finding latest benchmark results...")
    
    # Find latest results
    try:
        ms_marco_file = find_latest_result(results_dir, "ms_marco_results")
        print(f"   ✅ MS MARCO: {ms_marco_file.name}")
    except FileNotFoundError as e:
        print(f"   ❌ MS MARCO: {e}")
        ms_marco_file = None
    
    try:
        hotpot_qa_file = find_latest_result(results_dir, "hotpot_qa_results")
        print(f"   ✅ HotpotQA: {hotpot_qa_file.name}")
    except FileNotFoundError as e:
        print(f"   ❌ HotpotQA: {e}")
        hotpot_qa_file = None
    
    if not ms_marco_file and not hotpot_qa_file:
        print("\n❌ No results found. Run the benchmark first.")
        return
    
    print("\n� Generating benchmark reports...")
    
    # Initialize report generator
    generator = ReportGenerator(output_dir=reports_dir)
    
    # Generate MS MARCO report
    if ms_marco_file:
        with open(ms_marco_file, 'r', encoding='utf-8') as f:
            ms_marco_data = json.load(f)
        
        timestamp = ms_marco_file.stem.split('_')[-1] if '_' in ms_marco_file.stem else "latest"
        report_file = reports_dir / f"ms_marco_report_{timestamp}.md"
        
        print(f"\n   Generating MS MARCO report...")
        ms_marco_report = generator.generate_markdown_report(
            run_data=ms_marco_data,
            output_file=report_file
        )
        print(f"   ✅ Saved: {ms_marco_report}")
    
    # Generate HotpotQA report
    if hotpot_qa_file:
        with open(hotpot_qa_file, 'r', encoding='utf-8') as f:
            hotpot_qa_data = json.load(f)
        
        timestamp = hotpot_qa_file.stem.split('_')[-1] if '_' in hotpot_qa_file.stem else "latest"
        report_file = reports_dir / f"hotpot_qa_report_{timestamp}.md"
        
        print(f"\n   Generating HotpotQA report...")
        hotpot_qa_report = generator.generate_markdown_report(
            run_data=hotpot_qa_data,
            output_file=report_file
        )
        print(f"   ✅ Saved: {hotpot_qa_report}")
    
    # Generate comparison report if both exist
    if ms_marco_file and hotpot_qa_file:
        print(f"\n   Generating comparison report...")
        # Comparison report expects different format - skip for now
        print(f"   ⚠️  Comparison report skipped (requires comparison_data format)")
    
    print("\n✅ All reports generated successfully!")
    print(f"\n📁 Reports location: {reports_dir.absolute()}")

if __name__ == "__main__":
    main()

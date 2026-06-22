import os
import json

with open(r"C:\Q2_Sai\run_all.py", "r", encoding="utf-8") as f:
    content = f.read()

# The insertion point in Phase 6:
insertion_code = """
    # ── Write result.md ──
    result_md_path = out_dir / "result.md"
    md_content = ["# Q2 Experiment Results: Refusal Direction Analysis\\n"]
    
    for tag in tags:
        md_content.append(f"## Model: {tag}\\n")
        h_path = BASE_DIR / "results" / tag / "hypothesis_test_results.json"
        if h_path.exists():
            hyp = json.load(open(h_path))
            h1 = hyp.get("H1_harm_concept_detector", {})
            h2 = hyp.get("H2_assistance_intent_detector", {})
            
            md_content.append(f"### Hypothesis 1 (Harm Concept Detector): **{h1.get('status', 'N/A')}**")
            md_content.append(f"- Criterion: {h1.get('criterion', '')}")
            md_content.append(f"- Observed d(A,B): {h1.get('observed_d_AB', 'N/A')}")
            md_content.append(f"- Observed d(B,F): {h1.get('observed_d_BF', 'N/A')}\\n")
            
            md_content.append(f"### Hypothesis 2 (Assistance Intent Detector): **{h2.get('status', 'N/A')}**")
            md_content.append(f"- Criterion: {h2.get('criterion', '')}")
            md_content.append(f"- Observed d(A,B): {h2.get('observed_d_AB', 'N/A')}")
            md_content.append(f"- Observed d(B,F): {h2.get('observed_d_BF', 'N/A')}\\n")
            
            if h1.get("status") == "CONFIRMED":
                md_content.append("**Interpretation**: The refusal direction strongly activates when reading harmful content (Condition B) compared to benign content (Condition F), and there is little difference between reading and producing harm (A vs B). This suggests the direction encodes the general concept of harm.\\n")
            elif h2.get("status") == "CONFIRMED":
                md_content.append("**Interpretation**: The refusal direction primarily activates when the model is asked to produce harm (Condition A), with significantly lower activation during passive reading (Condition B). This suggests it specifically encodes the intent to assist with a harmful request, rather than just the concept of harm itself.\\n")
            else:
                md_content.append("**Interpretation**: The results are mixed and do not clearly confirm either primary hypothesis. The refusal direction's behavior may be more nuanced or context-dependent.\\n")
                
    with open(result_md_path, "w", encoding="utf-8") as f:
        f.write("\\n".join(md_content))
    log(f"Markdown results written to {result_md_path}")
"""

target = '    recovery.mark_done("phase_6_paper")\n    log("Phase 6 COMPLETE")'

if target in content:
    content = content.replace(target, insertion_code + "\n" + target)
else:
    print("Target string not found in run_all.py")

with open(r"C:\Q2_Sai\run_all_final.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Saved run_all_final.py")

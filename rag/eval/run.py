"""Retrieval baseline; generation review stays explicit until API and clinical review."""
import json
import os
import time
from pathlib import Path
from ..config import DATA
from ..retrieve import Retriever
from ..store import get_chunk


def run(generate=False):
    if generate and not os.getenv("DEEPSEEK_API_KEY"):
        raise RuntimeError("DEEPSEEK_API_KEY is required for --generate")
    cases = json.loads((Path(__file__).parent / "cases.json").read_text())
    retriever = Retriever()
    rows = []
    for case in cases:
        if case.get("refusal"):
            row = {"id":case["id"], "retrieval":"not_applicable", "generation":"not_run"}
            if generate:
                from ..agent import ask
                result = ask(case["question"], case["audience"], retriever)
                row.update({"generation":result["status"], "refusal_correct":result["status"] in ("personal_question","insufficient_evidence")})
            rows.append(row)
            continue
        start = time.perf_counter()
        hits, stages = retriever.search(case["question"])
        ranked = [get_chunk(retriever.db, chunk_id) for chunk_id in stages["reranked"][:10]]
        ranked = [chunk for chunk in ranked if chunk]
        anchors = case.get("anchors") or [case]
        ranks = []
        for anchor in anchors:
            found = [i+1 for i,h in enumerate(ranked) if anchor["source_id"] in h["source_ids"]
                     or (h["book"] == anchor["book"] and h["page_start"] <= anchor["pdf_page"] <= h["page_end"])]
            ranks.append(found[0] if found else None)
        all_found = all(ranks)
        worst = max(ranks) if all_found else None
        row = {"id":case["id"],"hit_at_5":bool(worst and worst<=5),
                     "hit_at_10":bool(worst and worst<=10),"reciprocal_rank":1/worst if worst else 0,
                     "latency_ms":round((time.perf_counter()-start)*1000),"generation":"not_run"}
        if generate:
            from ..agent import ask
            result = ask(case["question"], case["audience"], retriever)
            structural = True
            for citation in result["citations"]:
                chunk = next((h for h in hits if h["chunk_id"] == citation["chunk_id"]), None)
                if chunk is None:
                    # Targeted search may retrieve a citation beyond the
                    # initial result set; validate against the registry.
                    chunk = get_chunk(retriever.db, citation["chunk_id"])
                spans = chunk["source_spans"] if chunk else []
                if not any(s["source_id"] == citation["source_id"] and s["page"] == citation["pdf_page"]
                           and citation["excerpt"] in chunk["text"][s["start"]:s["end"]] for s in spans):
                    structural = False
            row.update({"generation":result["status"], "citation_structural_valid":structural,
                        "citation_count":len(result["citations"])})
        rows.append(row)
    scored = [r for r in rows if "hit_at_5" in r]
    answered = [r for r in scored if r.get("generation") == "answered"]
    probes = [r for r in rows if "refusal_correct" in r]
    report = {"cases":len(rows),"retrieval_cases":len(scored),
              "hit_at_5":sum(r["hit_at_5"] for r in scored)/len(scored) if scored else None,
              "hit_at_10":sum(r["hit_at_10"] for r in scored)/len(scored) if scored else None,
              "mrr":sum(r["reciprocal_rank"] for r in scored)/len(scored) if scored else None,
              "citation_structural_valid":sum(r.get("citation_structural_valid",False) for r in answered)/len(answered) if answered else "not_run",
              "refusal_probe_correct":f"{sum(r['refusal_correct'] for r in probes)}/{len(probes)}" if probes else "not_run",
              "clinical_errors":"not_reviewed", "patient_safety":"not_reviewed", "rows":rows}
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "evaluation.json").write_text(json.dumps(report, indent=2))
    lines = ["# AI Dentist evaluation", "", f"Cases: {len(rows)} total, {len(scored)} retrieval-scored.", "",
             f"- Hit@5: {report['hit_at_5']:.1%}", f"- Hit@10: {report['hit_at_10']:.1%}",
             f"- MRR: {report['mrr']:.3f}",
             f"- Structural citation validity: {report['citation_structural_valid']}",
             f"- Refusal probes correct: {report['refusal_probe_correct']}",
             "- Reviewed clinical errors: pending", "- Patient safety review: pending", "",
             "| Case | Hit@5 | Hit@10 | Latency (ms) | Generation |", "|---|---:|---:|---:|---|"]
    for row in rows:
        lines.append(f"| {row['id']} | {row.get('hit_at_5','—')} | {row.get('hit_at_10','—')} | {row.get('latency_ms','—')} | {row['generation']} |")
    (DATA / "evaluation.md").write_text("\n".join(lines) + "\n")
    return report

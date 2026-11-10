"""Recalculate the cached UDPipe/BERT morphology comparison without inference.

Run with ``python -B scripts/analysis/evaluate_udpipe_morphology.py``. Only the three
``morphology_*`` outputs in the UDPipe reference directory are written.
Metric definitions match the notebook: complete FEATS dictionaries (including
Clitic and Typo), pair-presence macro-F1 excluding <NONE>, and UPOS+FEATS joint
accuracy excluding XPOS. CSV comparison values are percentages; JSON metric
values are fractions. No network, model weights or third-party packages are used.
"""

import argparse
import csv
import hashlib
import io
import json
import math
import statistics
from collections import Counter
from pathlib import Path


MODEL = "italian-isdt-ud-2.15-241121"
METRICS = (
    "upos_accuracy", "upos_macro_f1_all", "upos_macro_f1_present",
    "feats_exact_accuracy", "feats_attribute_accuracy",
    "feats_macro_f1_all", "feats_macro_f1_present", "joint_accuracy",
)
PROVENANCE = {}


def check(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(content):
    return hashlib.sha256(content).hexdigest()


def read_text(path, root):
    content = path.read_bytes()
    PROVENANCE[path.relative_to(root).as_posix()] = sha256(content)
    return content.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")


def read_json(path, root):
    return json.loads(read_text(path, root))


def parse_feats(value):
    if value in ("", "_"):
        return {}
    result = {}
    for item in value.split("|"):
        check("=" in item, f"Invalid FEATS item: {item!r}")
        key, values = item.split("=", 1)
        check(key and values and key not in result, f"Invalid/duplicate FEATS key: {item!r}")
        check(values != "<NONE>", "<NONE> must be represented as an absent attribute")
        result[key] = ",".join(sorted(values.split(",")))
    return result


def parse_conllu(text):
    sentences, words = [], []
    for line in text.splitlines() + [""]:
        if not line:
            if words:
                check([row[0] for row in words] == [str(i + 1) for i in range(len(words))],
                      "Integer word IDs must be consecutive within each sentence")
                sentences.append(words)
                words = []
            continue
        if line.startswith("#"):
            continue
        columns = line.split("\t")
        check(len(columns) == 10, "CoNLL-U row must contain ten columns")
        if columns[0].isdigit():
            words.append(columns)
        else:
            check("-" in columns[0] or "." in columns[0], "Invalid CoNLL-U ID")
    return sentences


def check_alignment(actual, expected, label):
    check(len(actual) == len(expected), f"{label}: sentence count differs")
    for index, (observed, gold) in enumerate(zip(actual, expected)):
        check([(row[0], row[1]) for row in observed] == [(row[0], row[1]) for row in gold],
              f"{label}: word ID/FORM or sentence boundary differs at sentence {index}")


def flatten(sentences):
    return [word for sentence in sentences for word in sentence]


def pair_scores(gold, predicted, inventory):
    gold_sets = [{f"{key}={value}" for key, value in row.items()} for row in gold]
    pred_sets = [{f"{key}={value}" for key, value in row.items()} for row in predicted]
    support = Counter(pair for row in gold_sets for pair in row)
    guessed = Counter(pair for row in pred_sets for pair in row)
    matched = Counter(pair for a, b in zip(gold_sets, pred_sets) for pair in a & b)
    rows = []
    for pair in sorted(inventory | set(support) | set(guessed)):
        tp, actual, guesses = matched[pair], support[pair], guessed[pair]
        rows.append({"feature": pair, "precision": tp / guesses if guesses else 0.,
                     "recall": tp / actual if actual else 0.,
                     "f1": 2 * tp / (actual + guesses) if actual + guesses else 0.,
                     "support": actual, "tp": tp, "fp": guesses - tp, "fn": actual - tp})
    return rows


def score(gold, predicted_upos, predicted_feats, labels, training_pairs, attributes):
    check(len(gold) == len(predicted_upos) == len(predicted_feats), "Prediction length differs")
    check(all(label in labels for label in predicted_upos), "Unknown predicted UPOS tag")
    gold_upos = [row[3] for row in gold]
    gold_feats = [parse_feats(row[5]) for row in gold]
    upos_ok = [a == b for a, b in zip(gold_upos, predicted_upos)]
    exact = [a == b for a, b in zip(gold_feats, predicted_feats)]
    upos_rows = pair_scores([{"UPOS": x} for x in gold_upos],
                            [{"UPOS": x} for x in predicted_upos],
                            {f"UPOS={x}" for x in labels})
    feature_rows = pair_scores(gold_feats, predicted_feats, training_pairs)
    all_attributes = sorted(attributes | {key for row in gold_feats + predicted_feats for key in row})
    correct_attributes = sum(a.get(key, "<NONE>") == b.get(key, "<NONE>")
                             for a, b in zip(gold_feats, predicted_feats) for key in all_attributes)
    words = len(gold)
    metrics = {
        "upos_accuracy": sum(upos_ok) / words,
        "upos_macro_f1_all": statistics.mean(row["f1"] for row in upos_rows),
        "upos_macro_f1_present": statistics.mean(row["f1"] for row in upos_rows if row["support"]),
        "feats_exact_accuracy": sum(exact) / words,
        "feats_attribute_accuracy": correct_attributes / (words * len(all_attributes)),
        "feats_macro_f1_all": statistics.mean(row["f1"] for row in feature_rows),
        "feats_macro_f1_present": statistics.mean(row["f1"] for row in feature_rows if row["support"]),
        "joint_accuracy": sum(a and b for a, b in zip(upos_ok, exact)) / words,
    }
    counts = {"words": words, "upos_errors": words - sum(upos_ok),
              "feats_exact_errors": words - sum(exact),
              "joint_errors": sum(not (a and b) for a, b in zip(upos_ok, exact)),
              "feature_pairs_all": len(feature_rows),
              "feature_pairs_present": sum(row["support"] > 0 for row in feature_rows),
              "attributes": len(all_attributes), "attribute_decisions": words * len(all_attributes)}
    return metrics, counts, feature_rows


def csv_text(rows, fields):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def main(root):
    PROVENANCE.clear()
    reference = root / "outputs/references/udpipe_2_15"
    experiment = root / "outputs/bert_upos_then_feats_v2"
    config = read_json(experiment / "experiment.json", root)
    info = read_json(reference / "reference_info.json", root)
    check(info["model"] == MODEL, "Unexpected UDPipe reference model")
    check(info["gold_annotations_sent"] is False, "Gold annotations were sent to the reference")
    check(info["parser_requested"] is False, "Unexpected reference parser setting")
    check(info["input"] == "conllu, gold words and sentences", "Unexpected reference input protocol")
    data = {}
    for split in ("train", "test"):
        path = root / f"data/UD_Italian-ISDT-r2.15/it_isdt-ud-{split}.conllu"
        data[split] = parse_conllu(read_text(path, root))
        check(PROVENANCE[path.relative_to(root).as_posix()] == config["data_hashes"][path.name],
              f"{split}: data hash differs from the experiment")
    check(info["test_sha256"] == config["data_hashes"]["it_isdt-ud-test.conllu"],
          "Reference and BERT test hashes differ")
    gold = flatten(data["test"])
    check(info["evaluated_words"] == len(gold), "Reference word count differs")
    training_feats = [parse_feats(row[5]) for row in flatten(data["train"])]
    training_pairs = {f"{key}={value}" for row in training_feats for key, value in row.items()}
    attributes = {key for row in training_feats for key in row}
    labels = set(config["head_labels"]["UPOS"])
    check(labels == {row[3] for row in flatten(data["train"])}, "Training UPOS inventory differs")
    check(training_pairs == {f"{key}={value}" for key, values in config["head_labels"].items()
                             if key != "UPOS" for value in values if value != "<NONE>"},
          "Training FEATS pair inventory differs")
    check(attributes == set(config["head_names"][1:]), "Training FEATS attributes differ")

    predicted, chunks, next_sentence = [], [], 0
    for request in info["requests"]:
        start, count = request["start_sentence"], request["sentences"]
        check(start == next_sentence and count > 0, "Reference chunks are not contiguous")
        expected = data["test"][start:start + count]
        check(len(expected) == count, "Reference chunk extends beyond the gold test")
        input_path = reference / request["input_file"]
        check(input_path.parent == reference, "Unexpected cached input path")
        input_text = read_text(input_path, root)
        input_sentences = parse_conllu(input_text)
        check_alignment(input_sentences, expected, f"Reference input chunk {start}")
        check(all(all(value == "_" for value in row[2:]) for row in flatten(input_sentences)),
              "Reference input contains linguistic annotations")
        response_path = reference / request["response_file"]
        check(response_path.parent == reference and response_path.suffix == ".json", "Unexpected response path")
        response = read_json(response_path, root)
        check(response["model"] == MODEL, f"Reference chunk {start}: model differs")
        conllu_path = response_path.with_suffix(".conllu")
        response_text = read_text(conllu_path, root)
        check(response["result"] == response_text, f"Reference chunk {start}: JSON/CoNLL-U differ")
        check(f"# udpipe_model = {MODEL}\n" in response_text, "Missing/wrong CoNLL-U model header")
        response_sentences = parse_conllu(response_text)
        check_alignment(response_sentences, expected, f"Reference response chunk {start}")
        predicted.extend(flatten(response_sentences))
        recorded_hash = request["request_sha256"]
        check(len(recorded_hash) == 64 and all(c in "0123456789abcdef" for c in recorded_hash),
              "Invalid recorded request digest")
        check(recorded_hash[:16] in input_path.name and recorded_hash[:16] in response_path.name,
              "Recorded request hash and cache filenames differ")
        # The original request-payload serialization is not present in the saved
        # notebook or metadata. Do not mislabel a digest of normalized file text
        # as verification of the original request digest.
        chunks.append({"start_sentence": start, "sentences": count,
                       "words": sum(map(len, expected)),
                       "input_file": input_path.relative_to(root).as_posix(),
                       "response_file": response_path.relative_to(root).as_posix(),
                       "input_normalized_text_sha256": sha256(input_text.encode("utf-8")),
                       "recorded_request_sha256": recorded_hash,
                       "recorded_request_hash_payload_verified": False,
                       "recorded_request_hash_filename_prefix_verified": True})
        next_sentence += count
    check(next_sentence == len(data["test"]), "Cached reference does not cover the complete test")
    udpipe_metrics, udpipe_counts, feature_rows = score(
        gold, [row[3] for row in predicted], [parse_feats(row[5]) for row in predicted],
        labels, training_pairs, attributes)

    runs, comparison = [], []
    gold_index = [(s, w, row) for s, sentence in enumerate(data["test"])
                  for w, row in enumerate(sentence)]
    systems = [(name, spec) for name, spec in config["systems"].items() if spec["stage"] == "upos_feats"]
    check(len(systems) == 3 and len(config["seeds"]) == 3, "Expected three architectures and three seeds")
    for system, spec in systems:
        system_runs = []
        for seed in config["seeds"]:
            run_dir = experiment / system / f"seed_{seed}"
            metadata = read_json(run_dir / "training_info.json", root)
            check(metadata["system"] == system and metadata["seed"] == seed
                  and metadata["experiment"] == config, f"{system}/{seed}: training metadata differs")
            rows = list(csv.DictReader(io.StringIO(read_text(run_dir / "predictions.csv", root))))
            check(len(rows) == len(gold_index), f"{system}/{seed}: word count differs")
            for observed, (s, w, gold_word) in zip(rows, gold_index):
                check(int(observed["seed"]) == seed
                      and int(observed["sentence_index"]) == s and int(observed["word_index"]) == w
                      and observed["word"] == gold_word[1] and observed["true_label"] == gold_word[3]
                      and parse_feats(observed["true_feats"]) == parse_feats(gold_word[5]),
                      f"{system}/{seed}: gold alignment differs at sentence {s}, word {w}")
            metrics, counts, _ = score(gold, [row["predicted_label"] for row in rows],
                                       [parse_feats(row["predicted_feats"]) for row in rows],
                                       labels, training_pairs, attributes)
            saved = read_json(run_dir / "test_results.json", root)
            check(saved["system"] == system and saved["seed"] == seed,
                  f"{system}/{seed}: saved evaluation identity differs")
            for metric in METRICS:
                check(math.isclose(metrics[metric], saved[metric], rel_tol=0., abs_tol=1e-12),
                      f"{system}/{seed}: recomputed {metric} differs from saved result")
            record = {"system": system, "seed": seed, "metrics": metrics, "counts": counts}
            runs.append(record)
            system_runs.append(record)
        summary = {"system": spec["name"], "architecture": spec["architecture"], "runs": len(system_runs)}
        for metric in METRICS:
            values = [run["metrics"][metric] * 100 for run in system_runs]
            summary[f"{metric}_mean_pct"] = statistics.mean(values)
            summary[f"{metric}_std_pct"] = statistics.stdev(values)
        comparison.append(summary)
    check(len(runs) == 9, "Expected nine joint-stage BERT runs")
    reference_summary = {"system": "UDPipe 2.15", "architecture": "udpipe", "runs": 1}
    for metric in METRICS:
        reference_summary[f"{metric}_mean_pct"] = udpipe_metrics[metric] * 100
        reference_summary[f"{metric}_std_pct"] = ""
    comparison.append(reference_summary)

    comparison_fields = ["system", "architecture", "runs"]
    for metric in METRICS:
        comparison_fields += [f"{metric}_mean_pct", f"{metric}_std_pct"]
    outputs = {
        "morphology_comparison.csv": csv_text(comparison, comparison_fields),
        "morphology_per_feature_metrics.csv": csv_text(
            feature_rows, ["feature", "precision", "recall", "f1", "support", "tp", "fp", "fn"]),
    }
    script_path = Path(__file__).resolve()
    payload = {
        "schema": "udpipe_morphology_evaluation_v1",
        "model": MODEL,
        "protocol": {"release": "UD Italian-ISDT 2.15", "input": "gold words and sentence boundaries",
                     "alignment": "sentence ordinal, integer word ID and FORM; original sent_id comments are absent from cached input",
                     "metrics_units": "fractions", "comparison_csv_units": "percentages",
                     "per_feature_csv_units": "fractions", "standard_deviation": "sample; blank for single UDPipe execution",
                     "feats_exact": "complete dictionaries, including Clitic and Typo and empty sets",
                     "feats_macro_all": "mean pair-presence F1 over training union gold union predicted pairs; excludes <NONE>",
                     "feats_macro_present": "mean pair-presence F1 over pairs with positive gold support",
                     "feats_attribute_accuracy": "all word-attribute decisions over training union gold union predicted attributes, including absence",
                     "joint_accuracy": "UPOS and complete FEATS both correct; excludes XPOS",
                     "official_score_caution": "Official UDPipe UFeats filters universal attributes; official AllTags additionally includes XPOS. Neither is substituted for these metrics."},
        "checks": {"data_hashes": True, "reference_model_identity": True,
                   "gold_annotations_removed_from_cached_input": True, "complete_sentence_word_alignment": True,
                   "cached_json_equals_conllu_after_universal_newline_normalization": True,
                   "nine_bert_run_gold_alignments_and_saved_metrics": True,
                   "original_request_payload_hashes": "not verified: original serialization is unavailable; recorded digests and filename consistency are retained, and raw cached-file hashes are provided independently"},
        "reference": {"metrics": udpipe_metrics, "counts": udpipe_counts, "chunks": chunks},
        "bert_runs": runs,
        "comparison_percentages": comparison,
        "provenance_sha256": dict(sorted(PROVENANCE.items())),
        "evaluator_sha256": sha256(script_path.read_bytes()),
        "generated_csv_sha256": {name: sha256(content.encode("utf-8")) for name, content in outputs.items()},
    }
    outputs["morphology_evaluation.json"] = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    # All validation completes before any output is written.
    for name, content in outputs.items():
        (reference / name).write_bytes(content.encode("utf-8"))
    print(json.dumps({"udpipe_metrics_percent": {key: value * 100 for key, value in udpipe_metrics.items()},
                      "udpipe_counts": udpipe_counts, "verified_bert_runs": len(runs),
                      "outputs": list(outputs)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    main(parser.parse_args().project_root.resolve())

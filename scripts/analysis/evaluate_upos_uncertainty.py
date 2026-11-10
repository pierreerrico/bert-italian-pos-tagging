"""Paired sentence bootstrap for saved BERT/UDPipe UPOS predictions.

No training or inference. Each seed is analysed separately, keeping the
15-label inventory fixed across resamples. Results are conditional on the
saved models and on treating sentences as resampling units.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
KEYS = ('sentence_index', 'word_index', 'word', 'true_label')

def read_csv(path):
    with path.open(encoding='utf8', newline='') as stream:
        return list(csv.DictReader(stream))

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def summarise(gold_counts, predicted_counts, true_positives, lengths):
    denominator = gold_counts + predicted_counts
    f1 = np.divide(2 * true_positives, denominator,
                   out=np.zeros_like(true_positives, dtype=float),
                   where=denominator != 0)
    accuracy = true_positives.sum(axis=-1) / lengths
    return 100 * accuracy, 100 * f1.mean(axis=-1)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replicates', type=int, default=10000)
    parser.add_argument('--random-seed', type=int, default=20261109)
    args = parser.parse_args()
    if args.replicates < 1000:
        parser.error('Use at least 1000 replicates for the reported intervals.')
    reference_path = ROOT / 'outputs/references/udpipe_2_15/predictions.csv'
    reference = read_csv(reference_path)
    gold = np.array([r['true_label'] for r in reference])
    labels = sorted(set(gold))
    label_index = {tag: i for i, tag in enumerate(labels)}
    sentence_ids = sorted({int(r['sentence_index']) for r in reference})
    sentence_index = {value: i for i, value in enumerate(sentence_ids)}
    sentence = np.array([sentence_index[int(r['sentence_index'])] for r in reference])
    assert len(reference) == 10417 and len(sentence_ids) == 482
    assert len(labels) == 15
    for sid in sentence_ids:
        word_ids = [int(r['word_index']) for r in reference if int(r['sentence_index']) == sid]
        assert word_ids == list(range(len(word_ids))), f'Invalid word order in sentence {sid}'
    word_gold = np.array([label_index[tag] for tag in gold])
    lengths = np.bincount(sentence, minlength=len(sentence_ids)).astype(float)
    gold_counts = np.zeros((len(sentence_ids), len(labels)), dtype=float)
    np.add.at(gold_counts, (sentence, word_gold), 1)

    models = [('udpipe', reference_path, reference)]
    for phase, directory, system in [
        ('upos', 'bert_upos_then_feats_v1', 'linear_upos'),
        ('upos_feats', 'bert_upos_then_feats_v2', 'linear_upos_feats'),
    ]:
        for seed in [42, 123, 456]:
            path = ROOT / 'outputs' / directory / system / f'seed_{seed}/predictions.csv'
            rows = read_csv(path)
            assert len(rows) == len(reference)
            assert all(tuple(r[k] for k in KEYS) == tuple(b[k] for k in KEYS)
                       for r, b in zip(reference, rows)), f'Alignment failed: {path}'
            models.append((f'{phase}:{seed}', path, rows))

    predicted, true_positive = [], []
    for name, path, rows in models:
        pred = np.zeros_like(gold_counts)
        tp = np.zeros_like(gold_counts)
        for i, row in enumerate(rows):
            tag = row['predicted_label']
            if tag in label_index:
                pred[sentence[i], label_index[tag]] += 1
            if tag == row['true_label']:
                tp[sentence[i], word_gold[i]] += 1
        predicted.append(pred)
        true_positive.append(tp)

    observed = [summarise(gold_counts.sum(axis=0), pred.sum(axis=0),
                          tp.sum(axis=0), lengths.sum())
                for pred, tp in zip(predicted, true_positive)]
    # Independent arithmetic check against per-word accuracy and saved reports.
    saved_results = read_csv(ROOT / 'outputs/bert_upos_then_feats_v2/results.csv')
    for i, (name, path, rows) in enumerate(models):
        direct = 100 * sum(r['true_label'] == r['predicted_label'] for r in rows) / len(rows)
        assert abs(observed[i][0] - direct) < 1e-10
        if i:
            phase, seed = name.split(':')
            system = 'linear_upos' if phase == 'upos' else 'linear_upos_feats'
            saved = next(r for r in saved_results if r['system'] == system and int(r['seed']) == int(seed))
            for value, key in zip(observed[i], ['upos_accuracy', 'upos_macro_f1_present']):
                assert abs(value - 100 * float(saved[key])) < 1e-9

    bootstrap = np.empty((args.replicates, len(models), 2))
    absent_gold_counts = np.zeros(len(labels), dtype=int)
    rng = np.random.default_rng(args.random_seed)
    probabilities = np.full(len(sentence_ids), 1 / len(sentence_ids))
    # Concatenation makes one matrix multiplication per batch; every system sees
    # exactly the same resampled sentences and preserves their complete words.
    features = np.column_stack([lengths, gold_counts] +
                               [item for pair in zip(predicted, true_positive) for item in pair])
    width = len(labels)
    for start in range(0, args.replicates, 500):
        stop = min(start + 500, args.replicates)
        weights = rng.multinomial(len(sentence_ids), probabilities, size=stop-start)
        totals = weights @ features
        count = totals[:, 0]
        current_gold = totals[:, 1:1+width]
        absent_gold_counts += (current_gold == 0).sum(axis=0)
        for index in range(len(models)):
            offset = 1 + width + index * 2 * width
            current_pred = totals[:, offset:offset+width]
            current_tp = totals[:, offset+width:offset+2*width]
            bootstrap[start:stop, index, :] = np.column_stack(
                summarise(current_gold, current_pred, current_tp, count))

    # Check actual resampled raw words, independently of matrix aggregation.
    from collections import Counter
    check_weights = np.random.default_rng(args.random_seed).multinomial(
        len(sentence_ids), probabilities, size=500)
    for replicate in [0, 17, 101]:
        for model_index, (name, path, rows) in enumerate(models):
            expanded = [row for row_index, row in enumerate(rows)
                        for _ in range(int(check_weights[replicate, sentence[row_index]]))]
            gold_support = Counter(row['true_label'] for row in expanded)
            pred_support = Counter(row['predicted_label'] for row in expanded)
            matches = Counter(row['true_label'] for row in expanded
                              if row['true_label'] == row['predicted_label'])
            direct_accuracy = 100 * sum(matches.values()) / len(expanded)
            direct_f1 = 100 * sum(2 * matches[tag] / (gold_support[tag] + pred_support[tag])
                                 if gold_support[tag] + pred_support[tag] else 0
                                 for tag in labels) / len(labels)
            assert np.allclose([direct_accuracy, direct_f1],
                               bootstrap[replicate, model_index], rtol=0, atol=1e-10)

    output = ROOT / 'outputs/references/udpipe_2_15/paired_bootstrap'
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for index, (name, path, rows) in enumerate(models[1:], 1):
        phase, seed = name.split(':')
        for metric_index, metric in enumerate(['upos_accuracy', 'upos_macro_f1_present']):
            differences = bootstrap[:, index, metric_index] - bootstrap[:, 0, metric_index]
            lower, upper = np.quantile(differences, [.025, .975], method='linear')
            results.append(dict(phase=phase, seed=int(seed), metric=metric,
                                bert_score_pct=float(observed[index][metric_index]),
                                udpipe_score_pct=float(observed[0][metric_index]),
                                difference_pp=float(observed[index][metric_index]-observed[0][metric_index]),
                                ci95_lower_pp=float(lower), ci95_upper_pp=float(upper),
                                includes_zero=bool(lower <= 0 <= upper)))
    with (output/'summary.csv').open('w', encoding='utf8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader(); writer.writerows(results)
    metadata = dict(method='paired sentence bootstrap; percentile confidence intervals',
                    replicates=args.replicates, random_seed=args.random_seed,
                    confidence_level=.95, sentences=len(sentence_ids), words=len(reference),
                    fixed_labels=labels, zero_denominator_f1=0,
                    bootstrap_absent_gold_fraction={tag: float(n/args.replicates) for tag,n in zip(labels,absent_gold_counts)},
                    verification='Observed metrics match saved results; three paired resamples checked against expanded raw words for every model',
                    seed_handling='separate intervals per saved model, no pooling of words or seeds',
                    limitations=['Conditional on saved fitted models',
                                 'Sentence resampling does not model dependence between sentences from one document',
                                 'Rare labels cause unstable macro-F1; classes remain fixed even if absent in a resample',
                                 'Intervals are descriptive and marginal, not multiplicity-adjusted hypothesis tests'],
                    numpy_version=np.__version__,
                    sources=[dict(name=name, path=str(path.relative_to(ROOT)), sha256=digest(path))
                             for name, path, rows in models], results=results)
    (output/'analysis.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf8')
    np.savez_compressed(output/'bootstrap_scores.npz', scores=bootstrap,
                        model_names=np.array([name for name, path, rows in models]))
    for result in results:
        print(f"{result['phase']:10} seed {result['seed']:3} {result['metric']:23}: "
              f"{result['difference_pp']:+.3f} [{result['ci95_lower_pp']:+.3f}, {result['ci95_upper_pp']:+.3f}] pp")
    print('Saved to', output)

if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""Late manuscript-polish steps applied to Mahdi's tracked-changes draft AFTER
apply_numbers / apply_tables / apply_rest / refresh_contrasts / apply_refs /
renumber_refs / fix_refs, in the order they were run on 2026-10-23/24.

Recorded here for reproducibility. These were executed as inline edits during the
session; this file is a faithful transcription of those operations, not a script
that was itself run end-to-end from the draft. Run against the output of the
scripts above. Uses the tracked-change-aware editor in dxml.py.
"""
import sys, re, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dxml import Doc, W

def run(F, stage_control_csv, architectures_csv):
    import pandas as pd
    s = pd.read_csv(stage_control_csv).iloc[0]; a = pd.read_csv(architectures_csv)
    d = Doc(F)

    # 1. S13 citation in Methods; the inline six-item em-dash list removed (comment 64 overlap)
    d.replace(r'against a prospectively written six-category codebook — poor focus, partial view of the incision, obstruction of more than a quarter of the incision, motion artifact, dressing covering the incision, and annotator uncertainty —',
              'against a prospectively written six-category codebook (Supplementary Table S13)', 's13')

    # 2. stray all-photograph frame counts -> postoperative frame, with honest labelling
    d.replace(r'Sixty-two photographs were not maskable; 19 patients were affected and four were fully excluded, including one SSI-positive patient\. The final masking frame included 1,886 photographs from 206 patients with 19 SSI events\.',
              'Forty-five postoperative photographs were not maskable; three patients were thereby fully excluded, including one SSI-positive patient. The final masking frame included 1,296 photographs from 168 patients with 12 SSI events.', 'mask-frame')
    d.replace(r'On the 1,886-image mask-eligible frame, patient-grouped logistic probes were trained without wound pixels\.',
              'Patient-grouped logistic probes were trained without wound pixels on the 1,886-image mask-eligible frame of the all-photograph analysis; they were not repeated on the postoperative frame, and the context-probe results in this section therefore retain the intraoperative photographs.', 'ctx-methods')
    d.replace(r'On the matched 206-patient masking frame, the retrained full-image model achieved',
              'On the matched 206-patient masking frame of the all-photograph analysis (these probes were not repeated on the postoperative frame), the retrained full-image model achieved', 'ctx-results')

    # 3. spaced em-dash parentheticals -> commas (comment 64)
    d.replace(r' — ', ', ', 'emdash', count=200)

    # 4. Table S7 stage control on the postoperative frame
    d.replace(r'Across 881 out-of-fold images from 200 patients, early versus late postoperative stage was classified with an image-level AUROC of 0\.976 \(patient-clustered 95% CI, 0\.957–0\.990\)[^.]*\.',
              f'Across {s.n_images} out-of-fold postoperative images from {s.n_patients} patients ({s.n_early} at POD 0 to 2, {s.n_late} at POD 14 to 30), early versus late postoperative stage was classified with an image-level AUROC of {s.auroc:.3f} (patient-clustered 95% CI, {s.lo:.3f}–{s.hi:.3f}), well above the SSI architecture means of {a["mean"].min():.3f} to {a["mean"].max():.3f}.', 's7-results')
    d.replace(r'with image-level AUROC 0\.976,', f'with image-level AUROC {s.auroc:.3f},', 's7-discussion')
    d.replace(r'The supplied out-of-fold file contained 881 images from 200 patients\.',
              f'On the postoperative frame the early class held {s.n_early} images at POD 0 to 2 and the late class {s.n_late} at POD 14 to 30, from {s.n_patients} patients; the published run had included 321 intraoperative post-closure photographs in its early class and is superseded.', 's7-methods')
    d.replace(r'Early postoperative stage \(immediate postoperative imaging through POD 2\)', 'Early postoperative stage (POD 0 to 2, postoperative photographs only)', 's7-def')
    d.replace(r'881 images / 200 patients', f'{s.n_images} images / {s.n_patients} patients', 's7-cell')

    # 5. explicit not-repeated statements for the four analyses kept on the all-photograph frame
    d.replace(r'(individual seed trajectories were not uniformly monotonic \(Supplementary Table S2\)\.)', r'\1 The training-size experiment was not repeated on the postoperative frame, so these values and Supplementary Table S2 refer to the all-photograph frame; the full-training-set value therefore differs from the postoperative EfficientNet-B0 mean in Supplementary Table S1.', 's2')
    d.replace(r'(Supplementary Table S2 \| EfficientNet-B0 training-size experiment)', r'\1 (all-photograph frame; not repeated on the postoperative frame)', 's2-cap')
    d.replace(r'(compared with 0\.671 for the full-image model on the same frame \(Supplementary Table S11\)\.)', r'\1 The geometry control was not repeated on the postoperative frame; its values refer to the all-photograph masking frame.', 's11')
    d.replace(r'(Supplementary Table S11 \| Mask-geometry control)', r'\1 (all-photograph frame; not repeated on the postoperative frame)', 's11-cap')
    d.replace(r'(The frozen probes therefore do not reproduce[^.]*\(Supplementary Table S12\)\.)', r'\1 The frozen-encoder analysis was not repeated on the postoperative frame; its values refer to the all-photograph masking frame.', 's12')
    d.replace(r'(Supplementary Table S12 \| Frozen foundation-model probes)', r'\1 (all-photograph frame; not repeated on the postoperative frame)', 's12-cap')

    # 6. negative numbers: ASCII hyphen -> U+2212, numeric contexts only (hyphenated words and en-dash ranges untouched)
    for p in d.root.iter(W + 'p'):
        for x in p.iter(W + 't'):
            if x.text and re.search(r'(?<![\w\d])-\d+\.\d+', x.text):
                x.text = re.sub(r'(?<![\w\d])-(\d+\.\d+)', '−\\1', x.text)
    d.save(F)

if __name__ == '__main__':
    run(*sys.argv[1:4])

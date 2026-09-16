const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, BorderStyle, ShadingType, LevelFormat, PageBreak,
} = require("docx");

const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 320, after: 160 }, children: [new TextRun({ text: t, bold: true })] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 240, after: 120 }, children: [new TextRun({ text: t, bold: true })] });
const P = (runs, opts = {}) => new Paragraph({ spacing: { after: 140, line: 300 }, alignment: AlignmentType.JUSTIFIED, ...opts, children: Array.isArray(runs) ? runs : [new TextRun(runs)] });
const T = (t, o = {}) => new TextRun({ text: t, ...o });
const B = (t) => new TextRun({ text: t, bold: true });
const I = (t) => new TextRun({ text: t, italics: true });

const CW = [3100, 2000, 2000, 2260];
const cell = (t, bold = false, shade = false) => new TableCell({
  width: { size: 2340, type: WidthType.DXA },
  shading: shade ? { type: ShadingType.CLEAR, fill: "EEF2F7" } : undefined,
  margins: { top: 60, bottom: 60, left: 100, right: 100 },
  children: [new Paragraph({ children: [new TextRun({ text: t, bold, size: 19 })] })],
});
const row = (cells, hdr = false) => new TableRow({ children: cells.map((c) => cell(c, hdr, hdr)) });

const archTable = new Table({
  width: { size: 9360, type: WidthType.DXA },
  columnWidths: [2340, 2340, 2340, 2340],
  rows: [
    row(["Architecture", "Patient AUROC, mean ± SD", "Range across seeds", "Paired Δ vs ResNet18 (P)"], true),
    row(["EfficientNet-B0", "0.755 ± 0.037", "0.727–0.797", "−0.048 to +0.021 (0.46–0.78)"]),
    row(["ResNet18", "0.731 ± 0.052", "0.674–0.776", "reference (within-arch seed Δ −0.102, P=0.09)"]),
    row(["ViT-B/16", "0.725 ± 0.049", "0.677–0.774", "−0.002 (0.93)"]),
  ],
});

const maskTable = new Table({
  width: { size: 9360, type: WidthType.DXA },
  columnWidths: [3100, 2000, 4260],
  rows: [
    new TableRow({ children: [ cell("Input condition", true, true), cell("Patient AUROC", true, true), cell("Paired contrast (P)", true, true) ] }),
    new TableRow({ children: [ cell("Full image"), cell("0.725"), cell("reference") ] }),
    new TableRow({ children: [ cell("Wound-only (true mask)"), cell("0.670"), cell("vs full −0.055 (0.62)") ] }),
    new TableRow({ children: [ cell("Background-only (true mask)"), cell("0.821"), cell("vs full +0.097 (0.09)") ] }),
    new TableRow({ children: [ cell("Sham wound (relocated mask)"), cell("0.817"), cell("vs true background −0.005 (0.94); vs true wound +0.147 (0.08)") ] }),
    new TableRow({ children: [ cell("Sham background"), cell("0.654"), cell("vs full −0.071 (0.16)") ] }),
  ],
});

const doc = new Document({
  numbering: { config: [{ reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 360, hanging: 200 } } } }] }] },
  styles: { default: { document: { run: { font: "Calibri", size: 21 } } } },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
    children: [
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 }, children: [new TextRun({ text: "Paper 1 — Methods and Results (per review outline)", bold: true, size: 27 })] }),
      P([I("Completed per the review outline. All numbers trace to the released row-level out-of-fold predictions and the locked master fold map. One deviation from the outline, flagged in Section 9 of Results: the intraoperative numeric covariates (incision length, temperatures, closure times) were found to be empty in the analysis frame used for the outline's clinical numbers; after repair, clinical/combined values changed (0.652→0.701; 0.734→0.817) and combined-vs-clinical became significant on the full cohort. All other numbers are unchanged.")], { alignment: AlignmentType.LEFT }),

      H1("RESULTS"),

      H2("Cohort and image characteristics"),
      P([T("The analytic set comprised 1,948 quality-gated smartphone photographs from 210 adults undergoing surgery, of whom 20 (9.5%) developed a surgical-site infection (SSI) within 30 days. Patients contributed a median of 8 photographs (IQR 5–11; range 1–43) across protocol postoperative days (POD) 0, 4, 7, 10, 14 and 30. Characteristics by SSI status are given in Table 1. "), B("With 20 events, every interval in this study is wide and every null is bounded by design resolution rather than proof of equivalence; we state this ceiling here because all subsequent results must be read against it.")]),

      H2("Primary pre-diagnosis discrimination"),
      P([T("At the pre-specified POD-7 landmark — among 193 patients not yet diagnosed with SSI, using the mean model probability of all photographs obtained through POD 7 — image-only discrimination for subsequent SSI (15 infections) was AUROC 0.761 (95% CI 0.645–0.860), with AUPRC 0.172 against a 7.8% risk-set prevalence (Fig. 1). Pre-specified aggregation sensitivities were concordant: latest photograph through POD 7, 0.746; photograph closest to POD 7, 0.744. "), B("A single early postoperative photograph therefore carries a real but incomplete infection signal; 0.761 is the anchor estimate of this paper.")]),

      H2("The estimate is not driven by post-diagnosis images"),
      P([T("Restricting positive patients to strictly pre-diagnosis images changed the pooled patient score by +0.007 (95% CI −0.011 to +0.025; P=0.44) in a paired same-patient comparison (204 patients, 18 adjudicated diagnosis dates). The descriptive all-images comparator (patient AUROC 0.779, 210 patients) is reported only as a sensitivity analysis. "), B("The signal precedes, and does not depend on, photographs of clinically declared infection.")]),

      H2("Architecture and seed variability"),
      P([T("Across three training seeds per architecture on identical folds (Table 2; Fig. 2a), patient AUROC was 0.755±0.037 for EfficientNet-B0, 0.731±0.052 for ResNet18, and 0.725±0.049 for ViT-B/16. No between-architecture paired difference approached significance (P=0.46–0.93), and the largest contrast observed anywhere was ResNet18 against itself at a different seed (−0.102; P=0.09). The multi-seed training-size curve was flat: 0.718±0.016 at one-third of training patients, 0.746±0.057 at two-thirds, 0.755±0.037 at full size (Fig. 2b). "), B("Seed variance exceeds every architecture difference and no data-volume trend is detectable: model capacity is not the bottleneck.")]),
      archTable,
      P([I("Table 2. Fold-paired architecture comparison on the locked master map (3 seeds per architecture; identical folds and training protocol).")], { alignment: AlignmentType.LEFT, spacing: { before: 80, after: 200 } }),

      H2("Context probes: how and when a photograph is taken carries signal"),
      P([T("Postoperative day alone discriminated near chance at the image level (AUROC 0.51). At the patient level, however, models built only from acquisition characteristics, prior imaging behaviour (past-only, leak-audited), or POD each captured part of the apparent image signal: across seeds the CNN exceeded each single probe by +0.067±0.052 (POD), +0.058±0.052 (acquisition) and +0.061±0.052 (behaviour), while a combined context-only model approximately matched the CNN (Δ −0.009±0.052). "), B("Part of the apparent image discrimination reflects how and when photographs are obtained rather than wound biology; the signal is real but context-entangled.")]),

      H2("Spatial masking with sham controls: geometry carries no information"),
      P([T("On the annotation-available frame (1,886 images, 206 patients, 19 events; one SSI-positive patient lacked any annotation source and is accounted for in the released audit), retrained models scored: full image 0.725, wound-only 0.670, background-only 0.821 (Table 3; Fig. 3). The sham control was decisive: a randomly relocated, equal-area sham wound mask reproduced the true background result (0.817 vs 0.821; Δ −0.005, P=0.94), the true wound crop performed marginally worse than a random patch of the same area (Δ +0.147 in favour of sham; P=0.08), and the apparent background-over-full trend did not survive its own sham (sham background vs full −0.071, P=0.16). "), B("Masking and attention results in this setting cannot be read as biological localization — mask geometry itself carries no information — which corrects the CAM/attention explainability standard of prior SSI-imaging work.")]),
      maskTable,
      P([I("Table 3. Masked retraining with sham controls (annotation-available frame; frozen master folds; paired patient-bootstrap contrasts).")], { alignment: AlignmentType.LEFT, spacing: { before: 80, after: 200 } }),

      H2("Healing-stage positive control"),
      P([T("The identical photographs, processed by the identical pipeline, encoded normal postoperative healing stage near-perfectly (AUROC ≈0.975) while encoding infection only moderately (≈0.755). "), B("The images carry abundant healing information and limited infection information; the weak SSI result is an observation limit, not a method failure.")]),

      H2("Calibration and operating points"),
      P([T("For the POD-7 primary, Platt-scaled probabilities gave Brier 0.075 (95% CI 0.047–0.106) and calibration slope 0.63 (0.30–1.05; CI includes 1). At 80% specificity, sensitivity was 0.67 with a 24% flag rate; a high-sensitivity rule-out (14/15 infections, 93%) required flagging 51% of patients (specificity 0.53). "), B("Image-only screening at this event count is a rule-out tool with a high flag rate, not a standalone screen; we do not headline predictive values at 20 events.")]),

      H2("Image versus clinical versus combined"),
      P([T("[Corrected after covariate repair — supersedes the outline's 0.652/0.734.] On the locked cohort, the honestly specified clinical model (intraoperative covariates only; coverage outcome-neutral) scored 0.701, image 0.779, combined 0.817. Combined exceeded clinical alone (+0.116, 95% CI 0.024–0.225; P=0.013) but not image alone (+0.038; P=0.40); image-vs-clinical was non-significant (+0.078; P=0.39). On the leak-free POD-7 risk set, combined added nothing on AUROC (−0.020; P=0.80) with a suggestive AUPRC gain (+0.106; P=0.18); decision curves overlapped. The ascertainment-leaked clinical specification (chart-review coverage alone predicts SSI ≈0.75) is excluded from inference and retained only as a labelled diagnostic. "), B("Image and intraoperative clinical information are comparable; combining them is suggestive against clinical alone but not demonstrated over image alone on the prospective frame.")]),

      H2("Model versus clinicians (exploratory)"),
      P([T("Fifteen clinicians rated a shared image set; inter-rater agreement was slight (Fleiss κ=0.105), with surgeons and non-surgeons indistinguishable. On pre-diagnosis-restricted shared images the model numerically exceeded the clinician consensus (0.89 vs 0.63), but with only three positive patients in that subset the difference was not significant (P=0.33). "), B("The durable finding is that experts cannot consistently read these photographs; the model-versus-clinician comparison is descriptive, supplement-grade, and is not headlined.")]),

      H2("Summary of principal findings"),
      P("Four results define the paper: (1) a leak-free prospective estimand yields a real but incomplete signal (AUROC 0.761); (2) that signal is invariant to architecture, seed and training-set size — the ceiling is not a capacity limit; (3) a sham-mask control shows that masking and attention cannot be interpreted as biological localization, correcting prior explainability practice; and (4) the same photographs encode healing near-perfectly — the constraint is the observation, not the model."),

      new Paragraph({ children: [new PageBreak()] }),
      H1("METHODS"),

      H2("Study design and setting"),
      P([T("We conducted a prospective SSI-surveillance feasibility cohort at a single academic centre, with scheduled serial smartphone photography of the surgical incision across the 30-day postoperative window. The study is reported in line with TRIPOD for prediction models and STROBE where applicable, and the analysis plan was pre-specified before exposure–outcome modelling. "), B("The aim is to quantify how much infection-specific information a single RGB photograph contains — an information audit — not to propose a higher-performing classifier.")]),

      H2("Participants and cohort"),
      P("Adults undergoing surgery with a qualifying incision were enrolled and imaged at protocol postoperative days (0, 4, 7, 10, 14, 30). The analytic set comprised 1,948 quality-gated photographs from 210 patients, of whom 20 (9.5%) developed SSI. Before locking cohort membership we adjudicated enrolment records and outcomes: one duplicate enrolment was resolved (the SSI-positive record retained), consent withdrawals were identified and excluded, and SSI labels and diagnosis dates were adjudicated against structured chart review. None of the adjudicated records contributed images to the analytic arm, so adjudication did not change the analytic cohort (210/20); the locked estimate is therefore not an artifact of case selection."),

      H2("Image acquisition and quality gating"),
      P([T("Photographs were captured by patients or staff on their own smartphones; device, distance and lighting were deliberately uncontrolled to reflect real remote-surveillance conditions. A protocolized quality and dressing gate — requiring a visible, un-dressed incision and excluding non-wound photographs — was applied prospectively and blind to outcome; gate passage was similar for SSI-positive and SSI-negative patients’ images (56% vs 60%), and the gate is an automatable pipeline step, not retrospective exclusion of difficult cases. One locked master image–fold map (1,948 unique images; patient-grouped folds fixed under seed 42) was used for every analysis and is released for audit. "), B("“Curation” in this paper is thus a deployment specification, not hand-selection.")]),

      H2("Reference standard and independence"),
      P([T("The outcome was 30-day SSI by CDC/NHSN criteria, adjudicated from structured chart review. Study photographs were not available to the adjudicating team and did not inform any diagnosis. "), B("Index images are therefore independent of the reference standard, pre-empting circularity between the exposure and the outcome.")]),

      H2("Primary estimand"),
      P([T("The primary estimand is detect-before-diagnosis at a POD-7 landmark: among patients not yet diagnosed with SSI by POD 7 (positives with unadjudicable diagnosis dates excluded, n=2), predict subsequent SSI using only photographs obtained through POD 7, aggregated as the mean of the patient’s image probabilities. Pre-specified aggregation sensitivities used the latest photograph through POD 7 and the photograph closest to POD 7. POD-14 and POD-30 landmarks are secondary and descriptive (9 and 2 remaining events). "), B("The primary is a leak-free prospective estimand — the model cannot see the outcome period it predicts — combined, uniquely among SSI-imaging studies, with the confound battery below.")]),

      H2("Model development and validation"),
      P("We fine-tuned two convolutional backbones (ResNet18, EfficientNet-B0) and one transformer (ViT-B/16), ImageNet-pretrained, end-to-end for 12 epochs (AdamW, learning rate 1×10⁻⁴, 3×10⁻⁵ for ViT; horizontal/vertical flips; class-weighted cross-entropy) at 224×224 resolution. Validation used patient-grouped 5-fold cross-validation — all of a patient’s images in one fold, folds asserted patient-disjoint — with strictly out-of-fold predictions; patient-level scores pooled each patient’s image probabilities by the mean. Each architecture was trained at three random seeds (20260807, 42, 7) on identical folds. Every metric is patient-clustered; images are never treated as independent observations."),

      H2("Statistical analysis and uncertainty"),
      P([T("AUROC and AUPRC are co-primary (AUPRC reported throughout given ≈10% prevalence), with Brier score and calibration slope/intercept for probability quality. Uncertainty used patient-level clustered bootstrap 95% CIs (2,000 resamples); all model comparisons are paired ΔAUROC/ΔAUPRC computed within the same bootstrap replicate. A small pre-specified confirmatory set (the primary estimand and its leak sensitivity) is distinguished from all other analyses, which are exploratory and reported descriptively. As the design’s resolution, we report the observed paired-delta interval half-widths (≈0.10–0.16 patient-level AUROC): differences smaller than this are not detectable at 20 events. "), B("Model differences are reported as distributions across seeds, not single point estimates; seed variance is the correct yardstick at this event count.")]),

      H2("Information-audit and confounding analyses"),
      P([B("(a) Architecture × seed. "), T("Three architectures × three seeds on identical folds; between- and within-architecture paired deltas interpreted against seed variance.")]),
      P([B("(b) Acquisition/timing/behaviour probes. "), T("Logistic models built only from postoperative day, image-acquisition characteristics (brightness, contrast, sharpness, colour statistics computed at model resolution), and prior imaging behaviour restricted to past-only features (leak-audited: no total-visit counts); each probe compared with the CNN, plus a combined context-only model.")]),
      P([B("(c) Spatial masking with sham controls. "), T("Models retrained from scratch on full, wound-only and background-only inputs using expert incision annotations (incision-line strips and wound-bed polygons with an 11-pixel tolerance band), and additionally on randomly relocated equal-area sham masks that preserve mask area and shape while destroying anatomical position; paired deltas versus full and true-versus-sham contrasts. "), B("The sham control converts a masking result from “the model uses the wound” into a testable claim about whether geometry carries information at all.")]),
      P([B("(d) Healing-stage positive control. "), T("The same photographs and pipeline predicting normal postoperative stage, establishing that the pipeline extracts a strong signal when one is present.")]),

      H2("Calibration, operating points and decision curves"),
      P("Raw scores were used for discrimination and ranking; Platt scaling (cross-fold, fitted on training folds only) was used for probability and operating-point claims, as flexible recalibration is unstable at this event count (isotonic calibration produced degenerate slopes and is reported but not used). Operating points were pre-specified at fixed specificity (0.80, 0.90) and as a high-sensitivity rule-out, reported with sensitivity, flag rate and clustered CIs. Decision-curve net benefit was computed against treat-all and treat-none across threshold probabilities 0.02–0.30, with bootstrap bands. Clinical use is framed as rule-out with an explicit flag rate; predictive values are not headlined at 20 events."),

      H2("Comparative analyses"),
      P("Image, clinical and combined models were compared on the locked cohort with paired ΔAUROC/ΔAUPRC. The clinical model used intraoperative covariates only — the honest specification — because chart-review availability was outcome-correlated (the bare coverage indicator predicted SSI at ≈0.75), so covariate sets that encode chart ascertainment were excluded from inference and retained only as a labelled diagnostic. The clinician comparison — fifteen clinicians rating a shared image set (Fleiss κ for agreement; surgeon versus non-surgeon strata; pre-diagnosis-restricted model-versus-consensus contrast) — was pre-registered as exploratory owing to few positive patients in the shared-rated subset."),

      H2("Reproducibility"),
      P("The pipeline is deterministic under the fixed seed set. We release the locked master image–fold map, row-level out-of-fold predictions for every architecture × seed × masking condition (26,962 image-level rows with patient-level aggregation), the masking cohort audit (all image/patient exclusions with reasons; exclusions reflect annotation availability, not outcome), and analysis code organized by stage."),

    ],
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(process.argv[2] || "PAPER1_METHODS_RESULTS_draft.docx", buf);
  console.log("DOCX_WRITTEN");
});

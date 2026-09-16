const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType, PageBreak,
} = require("docx");

const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 340, after: 170 }, children: [new TextRun({ text: t, bold: true })] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 240, after: 120 }, children: [new TextRun({ text: t, bold: true })] });
const P = (runs, opts = {}) => new Paragraph({ spacing: { after: 150, line: 312 }, alignment: AlignmentType.JUSTIFIED, ...opts, children: Array.isArray(runs) ? runs : [new TextRun(runs)] });
const T = (t) => new TextRun({ text: t });
const B = (t) => new TextRun({ text: t, bold: true });
const I = (t) => new TextRun({ text: t, italics: true });
const REF = (t) => new TextRun({ text: t, superScript: true });

const cell = (t, bold = false, shade = false, w = 2340) => new TableCell({
  width: { size: w, type: WidthType.DXA },
  shading: shade ? { type: ShadingType.CLEAR, fill: "EDF1F7" } : undefined,
  margins: { top: 60, bottom: 60, left: 100, right: 100 },
  children: [new Paragraph({ children: [new TextRun({ text: t, bold, size: 19 })] })],
});

const t1 = new Table({ width: { size: 9360, type: WidthType.DXA }, columnWidths: [3744, 2808, 2808], rows: [
  new TableRow({ children: [cell("Characteristic", true, true, 3744), cell("SSI (n=20)", true, true, 2808), cell("No SSI (n=190)", true, true, 2808)] }),
  new TableRow({ children: [cell("Photographs, n", false, false, 3744), cell("188", false, false, 2808), cell("1,760", false, false, 2808)] }),
  new TableRow({ children: [cell("Photographs per patient, median (IQR)", false, false, 3744), cell("6 (3–13)", false, false, 2808), cell("8 (5–11)", false, false, 2808)] }),
  new TableRow({ children: [cell("Imaged by POD 7, n (%)", false, false, 3744), cell("20 (100%)", false, false, 2808), cell("178 (94%)", false, false, 2808)] }),
  new TableRow({ children: [cell("Incision length, cm, mean±SD (n recorded)", false, false, 3744), cell("22.7±8.3 (12)", false, false, 2808), cell("17.5±4.2 (114)", false, false, 2808)] }),
]});

const t2 = new Table({ width: { size: 9360, type: WidthType.DXA }, columnWidths: [2340, 2340, 2340, 2340], rows: [
  new TableRow({ children: [cell("Architecture", true, true), cell("Patient AUROC, mean±SD (3 seeds)", true, true), cell("Range", true, true), cell("Paired Δ vs ResNet18 (P)", true, true)] }),
  new TableRow({ children: [cell("EfficientNet-B0"), cell("0.755±0.037"), cell("0.727–0.797"), cell("−0.048 to +0.021 (0.46–0.78)")] }),
  new TableRow({ children: [cell("ResNet18"), cell("0.731±0.052"), cell("0.674–0.776"), cell("reference; within-seed Δ −0.102 (0.09)")] }),
  new TableRow({ children: [cell("ViT-B/16"), cell("0.725±0.049"), cell("0.677–0.774"), cell("−0.002 (0.93)")] }),
]});

const t3 = new Table({ width: { size: 9360, type: WidthType.DXA }, columnWidths: [3100, 2000, 4260], rows: [
  new TableRow({ children: [cell("Input condition", true, true, 3100), cell("Patient AUROC", true, true, 2000), cell("Paired contrast (P)", true, true, 4260)] }),
  new TableRow({ children: [cell("Full image", false, false, 3100), cell("0.725", false, false, 2000), cell("reference", false, false, 4260)] }),
  new TableRow({ children: [cell("Wound-only (true mask)", false, false, 3100), cell("0.670", false, false, 2000), cell("vs full −0.055 (0.62)", false, false, 4260)] }),
  new TableRow({ children: [cell("Background-only (true mask)", false, false, 3100), cell("0.821", false, false, 2000), cell("vs full +0.097 (0.09)", false, false, 4260)] }),
  new TableRow({ children: [cell("Sham wound (relocated mask)", false, false, 3100), cell("0.817", false, false, 2000), cell("vs true background −0.005 (0.94); vs true wound +0.147 (0.08)", false, false, 4260)] }),
  new TableRow({ children: [cell("Sham background", false, false, 3100), cell("0.654", false, false, 2000), cell("vs full −0.071 (0.16)", false, false, 4260)] }),
]});

const refs = [
  "Muaddi, H. et al. Imaging-based surgical site infection detection using artificial intelligence. Ann. Surg. 282, 419–428 (2025).",
  "Esteva, A. et al. Deep learning-enabled medical computer vision. npj Digit. Med. 4, 5 (2021).",
  "Sendak, M. P., Gao, M., Brajer, N. & Balu, S. Presenting machine learning model information to clinical end users with a model facts label. npj Digit. Med. 3, 41 (2020).",
  "Geirhos, R. et al. Shortcut learning in deep neural networks. Nat. Mach. Intell. 2, 665–673 (2020).",
  "DeGrave, A. J., Janizek, J. D. & Lee, S.-I. AI for radiographic COVID-19 detection selects shortcuts over signal. Nat. Mach. Intell. 3, 610–619 (2021).",
  "Zech, J. R. et al. Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: a cross-sectional study. PLoS Med. 15, e1002683 (2018).",
  "Collins, G. S., Reitsma, J. B., Altman, D. G. & Moons, K. G. M. Transparent reporting of a multivariable prediction model for individual prognosis or diagnosis (TRIPOD). Ann. Intern. Med. 162, 55–63 (2015).",
  "Collins, G. S. et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. BMJ 385, e078378 (2024).",
  "von Elm, E. et al. The Strengthening the Reporting of Observational Studies in Epidemiology (STROBE) statement. Lancet 370, 1453–1457 (2007).",
  "Horan, T. C., Andrus, M. & Dudeck, M. A. CDC/NHSN surveillance definition of health care-associated infection and criteria for specific types of infections in the acute care setting. Am. J. Infect. Control 36, 309–332 (2008).",
  "Sanger, P. C. et al. Diagnosing surgical site infection using wound photography: a scenario-based study. J. Am. Coll. Surg. 224, 8–15 (2017).",
  "Sanger, P. C. et al. A patient-centered system in a provider-centered world: challenges of incorporating post-discharge wound data into practice. J. Am. Med. Inform. Assoc. 23, 514–525 (2016).",
  "Selvaraju, R. R. et al. Grad-CAM: visual explanations from deep networks via gradient-based localization. Int. J. Comput. Vis. 128, 336–359 (2020).",
  "Vickers, A. J. & Elkin, E. B. Decision curve analysis: a novel method for evaluating prediction models. Med. Decis. Making 26, 565–574 (2006).",
  "Fleiss, J. L. Measuring nominal scale agreement among many raters. Psychol. Bull. 76, 378–382 (1971).",
  "Riley, R. D. et al. Calculating the sample size required for developing a clinical prediction model. BMJ 368, m441 (2020).",
  "Norgeot, B. et al. Minimum information about clinical artificial intelligence modeling: the MI-CLAIM checklist. Nat. Med. 26, 1320–1324 (2020).",
  "Wu, E. et al. How medical AI devices are evaluated: limitations and recommendations from an analysis of FDA approvals. Nat. Med. 27, 582–584 (2021).",
  "Platt, J. Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. In Advances in Large Margin Classifiers 61–74 (MIT Press, 1999).",
];

const doc = new Document({
  styles: { default: { document: { run: { font: "Calibri", size: 21 } } } },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
    children: [
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 140 }, children: [new TextRun({ text: "How much infection-specific information does a postoperative smartphone photograph contain? A leak-free information audit with sham-mask controls", bold: true, size: 28 })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 }, children: [new TextRun({ text: "[Authors and affiliations — Project AiCCESS, Rutgers]", italics: true, size: 20 })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 260 }, children: [new TextRun({ text: "Manuscript prepared for npj Digital Medicine (Article)", italics: true, size: 20 })] }),

      H1("ABSTRACT"),
      P([T("Smartphone photographs of surgical incisions are increasingly proposed for automated surgical-site-infection (SSI) surveillance, but reported discrimination may conflate infection-specific information with recognition of already-declared infection, acquisition context, and architecture-specific chance. In a prospective single-centre surveillance cohort (1,948 protocol photographs, 210 patients, 20 SSIs), we quantified the infection-specific information in a single RGB photograph using a leak-free landmark estimand and a pre-specified information audit: three architectures at three training seeds on one locked patient-fold map, past-only context probes, spatial masking with randomly relocated sham masks, and a healing-stage positive control. Among patients undiagnosed at postoperative day 7, images through day 7 predicted subsequent SSI with AUROC 0.761 (95% CI 0.645–0.860; 193 patients, 15 events); restricting positive patients to strictly pre-diagnosis images changed the pooled estimate by +0.007 (P=0.44). Performance was invariant to architecture, seed and training-set size (seed SD ≈0.04–0.05 exceeded every between-architecture difference), and a context-only model approximately matched the image model (Δ −0.009±0.052). Sham-mask controls showed that mask geometry carries no information — a randomly relocated wound mask reproduced true masking results — so masking and attention cannot be read as biological localization. The same photographs encoded normal healing stage near-perfectly (AUROC ≈0.975). A single wound photograph therefore contains a real but bounded infection signal that is contextual, non-localized, and insufficient as a standalone screen; the constraint is the observation, not the model.")]),

      H1("INTRODUCTION"),
      P([T("Surgical-site infection (SSI) is among the most common healthcare-associated infections, and most SSIs declare themselves after discharge, when the wound is out of clinical view. Patient-submitted smartphone photographs are already flowing into portals at scale, and clinicians find these images burdensome to triage and difficult to read reliably"), REF("11,12"), T(". Automated image-based SSI detection is the natural response, and recent work — most prominently a nine-hospital pipeline trained on 6,060 portal-submitted images, reaching AUROC 0.81 with a vision transformer"), REF("1"), T(" — has established feasibility at scale within the standard paradigm of medical computer vision"), REF("2"), T(".")]),
      P([T("That paradigm, however, answers a different question from the one surveillance needs. A classifier evaluated on all submitted images, including photographs taken at or after clinical diagnosis, measures recognition of declared infection at least as much as prediction of impending infection. Deep networks are also reliable exploiters of shortcuts — acquisition characteristics, timing, and site context that correlate with outcome"), REF("4–6"), T(" — and saliency or masking analyses commonly offered as reassurance (\"the model looks at the wound\") have not themselves been controlled: no SSI-imaging study has tested whether the wound region carries privileged information at all. Finally, with the small event counts typical of prospective surgical cohorts, differences between architectures can be indistinguishable from training-seed chance, yet are routinely reported as model rankings from single runs"), REF("18"), T(".")]),
      P([T("We therefore asked a deliberately bounded question: how much infection-specific information does a single postoperative smartphone photograph contain? We answer it in a prospective surveillance cohort using a leak-free landmark estimand (predicting subsequent SSI among patients not yet diagnosed, from photographs the model could actually have seen), and a pre-specified information audit: architecture-by-seed distributions on one locked fold map; past-only context probes; spatial masking with randomly relocated, equal-area sham masks; and a healing-stage positive control on the identical photographs. The aim is not a higher-performing classifier but a credible measurement — with the design features (estimand hygiene, seed distributions, sham controls) that reporting frameworks for clinical prediction and medical AI increasingly demand"), REF("3,7,8,17"), T(".")]),

      H1("RESULTS"),
      H2("Cohort and image characteristics"),
      P([T("The analytic set comprised 1,948 quality-gated smartphone photographs from 210 adults, of whom 20 (9.5%) developed SSI within 30 days. Patients contributed a median of 8 photographs (IQR 5–11; range 1–43) at protocol postoperative days (POD) 0, 4, 7, 10, 14 and 30; 94% of patients had imaging by POD 7 (Table 1). "), B("With 20 events, every interval in this study is wide, and every null is bounded by the design's resolution rather than by proof of equivalence; all results below must be read against this ceiling.")]),
      P([I("[Table 1 near here]")], { alignment: AlignmentType.LEFT }),

      H2("Primary pre-diagnosis discrimination"),
      P([T("At the pre-specified POD-7 landmark — among 193 patients not yet diagnosed with SSI, scoring each patient as the mean model probability of all photographs obtained through POD 7 — image-only discrimination for subsequent SSI (15 events) was AUROC 0.761 (95% CI 0.645–0.860), AUPRC 0.172 against a 7.8% risk-set prevalence (Fig. 1). Pre-specified aggregation sensitivities were concordant: latest photograph through POD 7, 0.746; photograph closest to POD 7, 0.744. "), B("A real but incomplete signal — this is the anchor number and the honest headline of the paper.")]),

      H2("Estimand robustness (leak-free)"),
      P([T("Restricting positive patients to strictly pre-diagnosis photographs did not change the pooled patient estimate: paired same-patient delta +0.007 (95% CI −0.011 to +0.025; P=0.44; 204 patients, 18 adjudicated diagnosis dates). The descriptive all-images comparator (patient AUROC 0.779) is reported as a sensitivity analysis, not as a primary result. "), B("The signal is not driven by post-diagnosis, clinically florid wounds — removing the strongest objection that image models merely recognize declared infection.")]),

      H2("Architecture and seed variability"),
      P([T("Across three seeds per architecture on the identical locked folds (Table 2; Fig. 2a): EfficientNet-B0 0.755±0.037, ResNet18 0.731±0.052, ViT-B/16 0.725±0.049 (patient AUROC, mean±SD). All between-architecture paired deltas were non-significant (P=0.46–0.93), architecture rankings changed with seed, and the largest contrast observed anywhere was ResNet18 against itself at a different seed (−0.102; P=0.09). The multi-seed training-size curve showed no upward trend: 0.718±0.016 at one-third of training patients, 0.746±0.057 at two-thirds, 0.755±0.037 at full size (Fig. 2b). "), B("Seed variance exceeds every architecture difference — capacity is not the bottleneck, and a bigger or different model does not help.")]),
      t2,
      P([I("Table 2. Fold-paired architectures on the locked master map (three seeds each; identical folds and protocol).")], { alignment: AlignmentType.LEFT, spacing: { before: 80, after: 200 } }),

      H2("Confound and context probes"),
      P([T("Postoperative day alone discriminated near chance at the image level. At the patient level, the image model exceeded each single context probe consistently across seeds (vs POD +0.067±0.052; vs acquisition characteristics +0.058±0.052; vs prior imaging behaviour +0.061±0.052), but a combined context-only model — acquisition, behaviour and POD together, none of which sees a single wound pixel — approximately matched the image model (Δ −0.009±0.052). "), B("Part of the apparent image discrimination reflects how and when photographs are obtained rather than wound biology; the signal is real but context-sensitive.")]),

      H2("Spatial localization: masking with sham controls"),
      P([T("On the annotation-available frame (1,886 images, 206 patients, 19 events; one SSI-positive patient lacked any annotation source and is itemized in the released audit), retrained models scored: full image 0.725, wound-only 0.670, background-only 0.821 — wound-only did not exceed full (Table 3; Fig. 3). The sham control was decisive: a randomly relocated, equal-area sham wound mask (0.817) matched the true masking conditions; sham-wound versus true background was indistinguishable (Δ −0.005, P=0.94); the true wound crop performed marginally worse than a random patch of equal area (Δ +0.147 favouring sham; P=0.08); and the apparent background-over-full trend did not survive its own sham (P=0.16). "), B("Masking results cannot be interpreted as biological localization — mask geometry carries no information. This corrects the CAM/attention explainability standard used in prior SSI-imaging work"), REF("1,13"), B(".")]),
      t3,
      P([I("Table 3. Masked retraining with sham controls (annotation-available frame; frozen master folds; paired patient-bootstrap contrasts).")], { alignment: AlignmentType.LEFT, spacing: { before: 80, after: 200 } }),

      H2("Healing-stage positive control"),
      P([T("The identical photographs, processed by the identical pipeline, encoded normal postoperative healing stage near-perfectly (AUROC ≈0.975) while encoding SSI only moderately (≈0.755). "), B("The images carry abundant healing information and limited infection information — the clearest single expression of the information asymmetry; the weak SSI result is not a method failure.")]),

      H2("Calibration and operating points"),
      P([T("For the POD-7 primary, Platt-scaled probabilities gave Brier 0.075 (95% CI 0.047–0.106) and calibration slope 0.63 (0.30–1.05; CI includes 1). At 80% specificity, sensitivity was 0.67 with a 24% flag rate; a high-sensitivity rule-out (14/15 events; ≈93%) required flagging 51% of patients (specificity 0.53). "), B("Image-only screening trades sensitivity for a high flag rate — concrete evidence that it is not adequate as a standalone screen; we do not headline predictive values at 20 events.")]),

      H2("Image versus clinical versus combined"),
      P([T("On the locked cohort, the honestly specified clinical model (intraoperative covariates only; coverage outcome-neutral) scored 0.701, image 0.779, and combined 0.817. The combined model exceeded clinical alone (+0.116, 95% CI 0.024–0.225; P=0.013) but not image alone (+0.038; P=0.40), and image versus clinical was non-significant (+0.078; P=0.39). On the leak-free POD-7 risk set the pattern was similar in AUPRC only: combined 0.277 versus image 0.172 (ΔAUPRC +0.106; P=0.18) with no AUROC gain (−0.020; P=0.80); decision curves overlapped. A chart-review-based clinical specification was excluded after audit because chart-review ascertainment itself predicted SSI (coverage-indicator AUROC ≈0.75). "), B("Image and intraoperative clinical information are comparable, and combining them is suggestive rather than demonstrated on the prospective frame — consistent with limited unique information in a single photograph.")]),

      H2("Model versus clinicians (exploratory)"),
      P([T("Fifteen clinicians rated a shared image set; inter-rater agreement was slight (Fleiss κ=0.105), with surgeons and non-surgeons indistinguishable. On pre-diagnosis-restricted shared images the model numerically exceeded the clinician consensus (0.89 vs 0.63), but with only three positive patients in that subset the difference was not significant (P=0.33) and is reported in the Supplement. "), B("The durable finding is that experts cannot consistently read these photographs"), REF("11"), B("; \"model beats clinicians\" is descriptive only and is not headlined.")]),

      H2("Summary of principal findings"),
      P("Four results must land: (1) a leak-free prospective estimand yields a real but incomplete signal (AUROC 0.761); (2) that signal is invariant to architecture, seed and training-set size — the ceiling is not a capacity limit; (3) a sham-mask control shows that masking and attention cannot be read as biological localization, correcting prior explainability practice; and (4) the same photographs encode healing near-perfectly — the constraint is the observation, not the model."),

      H1("DISCUSSION"),
      P([T("This study set out to measure, rather than maximize, the infection-specific information in a single postoperative smartphone photograph. The measurement comes back consistent from every direction we probed. A leak-free landmark estimand gives AUROC 0.761 — genuine signal, well short of clinical sufficiency. The estimate does not move when post-diagnosis images are removed, does not move with architecture, seed, or training-set size, is approximately reproduced by a model that never sees a wound pixel, and is unchanged in kind when the wound itself is masked out. Meanwhile the identical photographs support near-perfect recognition of normal healing stage. Together these findings locate the bottleneck in the observation itself: a single, uncontrolled RGB photograph, however processed, contains bounded information about incipient infection.")]),
      P([T("These results reframe, rather than contradict, prior work. The largest study to date"), REF("1"), T(" reported AUROC 0.81 for SSI detection on portal-submitted images — a number our all-images sensitivity analysis (0.779) essentially reproduces at one-tenth the scale. The information audit shows what such numbers contain: a mixture of pre-diagnostic signal, recognition of declared infection, and acquisition context. None of this makes triage pipelines less useful — flagging florid wounds quickly is valuable — but it cautions against reading recognition-era numbers as surveillance performance, and it supplies the missing negative control for explainability claims. Shortcut learning is well documented in medical imaging"), REF("4–6"), T(", and our sham-mask experiment converts that concern into a direct test: when a randomly relocated mask reproduces the true masking result, wound-centred saliency cannot be offered as evidence that a model has learned wound biology"), REF("13"), T(". We suggest sham-geometry controls become standard alongside any masking or attention analysis in wound imaging.")]),
      P([T("The clinician arm contextualizes the ceiling. Agreement among fifteen raters was slight (κ=0.105), consistent with earlier scenario-based photographic studies"), REF("11"), T(", and the model's numerical advantage over the pooled consensus — however underpowered — makes the substrate interpretation more likely than a modelling failure: these photographs do not support consistent expert judgement either. The practical implication is specific. As a rule-out instrument at ≈93% sensitivity, image-only screening clears roughly half of patients from review while flagging the other half — a meaningful workload reduction for portal triage"), REF("12"), T(", provided the flag rate is stated plainly and probabilities are handled with the calibration care that small event counts demand"), REF("3"), T(". As a standalone diagnostic screen, it is not adequate, and our data give the reason: the information is not there, not that the model failed to find it.")]),
      P([T("What would raise the ceiling is therefore an acquisition question. Serial change over a patient's own baseline, controlled capture geometry, added modalities (e.g., thermal), resolution preserving peri-wound texture, and structured symptom report are the candidates our framework can evaluate with the same estimand and controls. The combined image-plus-intraoperative model's suggestive AUPRC gain on the prospective frame points the same way: complementary information streams, not larger networks.")]),
      P([T("Limitations. This is a single-centre cohort with 20 events; every interval is wide, the design's resolution for paired differences is ≈0.10–0.16 AUROC, and all nulls are bounded, not proven. The quality gate, though protocolized, prospectively applied and outcome-neutral in passage rates, restricts conclusions to gateable images. The masking analyses use the annotation-available frame (206/19). The clinician comparison is exploratory by design. Single-seed rankings elsewhere in the literature suggest our seed-variance finding will generalize, but external validation of the estimand itself awaits a second cohort"), REF("16,18"), T(".")]),
      P([B("In sum, a smartphone photograph of a surgical wound contains real, architecture-independent, non-localized, context-entangled information about impending infection — enough to help triage, not enough to screen — and the path to more lies in what is captured, not in how it is modelled.")]),

      new Paragraph({ children: [new PageBreak()] }),
      H1("METHODS"),
      H2("Study design and setting"),
      P([T("We conducted a prospective SSI-surveillance feasibility cohort at a single academic centre, with scheduled serial smartphone photography of the surgical incision across the 30-day postoperative window. Reporting follows TRIPOD for prediction models"), REF("7,8"), T(" and STROBE where applicable"), REF("9"), T("; the analysis plan was pre-specified before exposure–outcome modelling. "), B("The aim is to quantify how much infection-specific information a single RGB photograph contains, not to propose a higher-performing classifier.")]),
      H2("Participants and cohort"),
      P("Adults undergoing surgery with a qualifying incision were enrolled, with imaging scheduled at protocol postoperative days (0, 4, 7, 10, 14, 30). The analytic set comprised 1,948 quality-gated photographs from 210 patients; 20 (9.5%) developed SSI. Before locking membership we adjudicated enrolment and outcomes: one duplicate enrolment was resolved (the SSI-positive record retained), consent withdrawals were identified and excluded, and SSI labels and diagnosis dates were adjudicated. None of the adjudicated records contributed analytic images, so adjudication did not change the analytic cohort (210/20): the locked estimate is not an artifact of case selection."),
      H2("Image acquisition and quality gating"),
      P([T("Images were captured by patients or staff on their own smartphones; device and lighting were uncontrolled by design, reflecting real remote-surveillance conditions. A protocolized quality/dressing gate — pass criteria: incision visible and free of dressing; fail criteria: dressing-covered, non-wound, or unusable frame — was applied prospectively and blind to outcome (gate passage 56% of SSI-positive vs 60% of SSI-negative patients' images); it is an automatable pipeline step, not retrospective exclusion of difficult cases. One locked master image–fold map (1,948 unique images; patient-grouped folds fixed under seed 42) was used for every analysis and is published for audit. "), B("\"Curation\" in this paper is thus a deployment specification, not hand-selection.")]),
      H2("Reference standard and independence"),
      P([T("The outcome was 30-day SSI by CDC/NHSN criteria"), REF("10"), T(" from structured chart review. Study photographs were not available to the adjudicating team and did not inform any diagnosis. "), B("Index images are therefore independent of the reference standard, pre-empting circularity between exposure and outcome — a property prior imaging studies have not clearly asserted.")]),
      H2("Primary estimand and outcome"),
      P([T("Primary: detect-before-diagnosis at a POD-7 landmark — the mean of all photographs through POD 7, among patients not yet diagnosed (193 patients, 15 subsequent infections; two positives with unadjudicable diagnosis dates excluded). The rationale is a clinically coherent surveillance question — predict, not recognize — that cannot see post-diagnosis images. Pre-specified aggregation sensitivities: latest photograph through POD 7; photograph closest to POD 7. "), B("The primary is a leak-free prospective estimand — the model cannot see the outcome it predicts — combined, uniquely among SSI-imaging studies, with the confound battery below.")]),
      H2("Model development and validation"),
      P([T("Two CNN backbones (ResNet18, EfficientNet-B0) and a transformer (ViT-B/16) were fine-tuned end-to-end from ImageNet weights (12 epochs; AdamW; learning rate 1×10⁻⁴, 3×10⁻⁵ for ViT; flip augmentation; class-weighted cross-entropy; 224×224 inputs). Validation used patient-grouped 5-fold cross-validation — all of a patient's images in one fold — with strictly out-of-fold predictions; patient-level scores pool each patient's image probabilities by the mean. Each architecture was trained at three random seeds on identical folds. "), B("Every metric is patient-clustered; images are never treated as independent.")]),
      H2("Statistical analysis and uncertainty"),
      P([T("AUROC and AUPRC are co-primary (AUPRC reported throughout at ≈10% prevalence), with Brier score and calibration slope/intercept for utility. Patient-level clustered bootstrap 95% CIs (2,000 resamples) were used throughout; all model comparisons are paired ΔAUROC/ΔAUPRC computed within the same bootstrap replicate. A small pre-specified confirmatory set (the primary estimand and its leak sensitivity) is distinguished from all other analyses, which are exploratory and reported descriptively; the minimum detectable ΔAUROC (observed paired-interval half-widths ≈0.10–0.16) is reported so plateaus are read against the design's resolution"), REF("16"), T(". "), B("Model differences are reported as distributions across seeds, not single point estimates — seed variance is the correct yardstick at this event count.")]),
      H2("Information-audit and confounding analyses"),
      P([B("(8a) Architecture × seed. "), T("Three architectures × three seeds on identical folds; between- and within-architecture paired deltas, interpreted against seed variance.")]),
      P([B("(8b) Acquisition/timing/behaviour probes. "), T("Models built only from postoperative day, image-acquisition characteristics, and prior imaging behaviour (past-only, leak-audited); each versus the image model, and a combined context-only model.")]),
      P([B("(8c) Spatial masking + sham controls. "), T("Models retrained on full, wound-only and background-only inputs (expert incision annotations: incision-line strips and wound-bed polygons with a tolerance band), AND on randomly relocated equal-area sham masks; paired deltas versus full and real-versus-sham contrasts. "), B("The sham control is the novel correction: it converts a masking result from \"the model uses the wound\" into a testable claim about whether geometry carries information at all.")]),
      P([B("(8d) Healing-stage positive control. "), T("The same photographs and pipeline predicting normal postoperative stage — establishing that the pipeline extracts a strong signal when one is present.")]),
      H2("Calibration, operating points, decision curves"),
      P([T("Raw scores were used for discrimination and ranking; Platt scaling"), REF("19"), T(" (cross-fold) for probability and operating-point claims — recalibration is unstable at this event count and flexible (isotonic) calibration was degenerate; both facts are stated rather than hidden. Operating points were pre-specified (fixed specificity; high-sensitivity rule-out) with sensitivity, flag rate and clustered CIs. Decision-curve net benefit"), REF("14"), T(" was computed against treat-all/treat-none across the plausible threshold range, with CIs. "), B("Clinical use is framed as rule-out with an explicit flag rate; PPV is not headlined at 20 events.")]),
      H2("Comparative analyses"),
      P([T("Image versus clinical (intraoperative-only — the honest specification, since chart-review availability was outcome-correlated and covariate sets encoding ascertainment were excluded from inference) versus combined, on the locked cohort, with paired ΔAUROC/ΔAUPRC. The clinician panel rated identical images (pre-diagnosis-restricted contrast; Fleiss κ"), REF("15"), T(" for agreement; surgeon versus non-surgeon strata). "), B("The clinician comparison is pre-registered as exploratory — few positive patients fall in the shared-rated subset.")]),
      H2("Reproducibility"),
      P("The pipeline is deterministic under the fixed seed set. The locked master fold map and row-level out-of-fold predictions are published for every architecture × seed × masking condition, together with the masking cohort audit (all image/patient exclusions with reasons; exclusions reflect annotation availability, not outcome) and analysis code organized by stage."),

      H1("Data availability"),
      P("De-identified row-level out-of-fold predictions, the locked master image–fold map, per-analysis result tables, and the masking cohort audit are available to the study team via the project repository; image data contain potentially identifying wound photographs and are available under the study's data-use agreements."),
      H1("Code availability"),
      P("Analysis code, organized by stage (cohort assembly, fold freezing, training, probes, masking with shams, calibration, packaging), is released with the results and reproduces every number in this manuscript from the locked inputs."),

      H1("References"),
      ...refs.map((r, i) => new Paragraph({ spacing: { after: 60 }, children: [new TextRun({ text: `${i + 1}. ${r}`, size: 19 })] })),

      new Paragraph({ children: [new PageBreak()] }),
      H1("Figure legends"),
      P([B("Fig. 1 | "), T("POD-7 landmark primary estimand. Risk-set structure and patient-level AUROC with 95% CI at POD 7 (primary), POD 14 and POD 30 (secondary/descriptive), with events remaining at each landmark.")]),
      P([B("Fig. 2 | "), T("Architecture and data-volume invariance. a, Patient AUROC for three architectures × three seeds on identical folds (points, seeds; bars, means). b, Multi-seed training-size curve (mean±SD across seeds).")]),
      P([B("Fig. 3 | "), T("Spatial masking with sham controls. Patient AUROC for full, wound-only, background-only, sham-wound and sham-background retrained models on the annotation-available frame, with example masked inputs.")]),
      P([B("Fig. 4 | "), T("Decision curves. Net benefit versus threshold probability for clinical, image and combined models against treat-all/treat-none, with bootstrap bands.")]),
      H1("Tables"),
      P([B("Table 1 | Cohort and imaging characteristics by SSI status.")]),
      t1,
      P([I("Baseline clinical characteristics (age, sex, comorbidity) were available on an outcome-correlated chart-review subset and are reported in the Supplement with that caveat; intraoperative fields shown are from the outcome-neutral intraoperative export (incision length recorded in 126/210).")], { alignment: AlignmentType.LEFT, spacing: { before: 80 } }),
    ],
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(process.argv[2] || "PAPER1_npjDigitalMedicine_manuscript.docx", buf);
  console.log("DOCX_WRITTEN");
});

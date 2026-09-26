#!/usr/bin/env python
"""Recover true postoperative days for photographs lacking a day number.

Evidence precedence, highest first:
  1. explicit day number in the filename (e.g. "D7 Post-Op", "Day_5.xxxx")
  2. the photograph's own EXIF capture date minus the patient's surgery date
  3. the paired thermal (FLIR) capture date at the same timepoint
  4. intraoperative naming convention ("(Pre-Standard n)" / "(Post-Standard n)") -> day 0

Surgery date per patient = median(EXIF capture date - labelled day) over that
patient's day-labelled photographs; cross-checked against chart review
(diagnosis date - POD). EXIF is read from the original images in the OneDrive
archive, because the materialised modelling copies are re-encoded and stripped.

PRIVACY: capture datetimes and surgery dates are calendar dates tied to patients
(PHI). They stay in the working directory and are never written to the published
outputs; only relative postoperative days are published.
"""
import io, re, zipfile
import numpy as np, pandas as pd
from PIL import Image, ExifTags, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
try:
    import pillow_heif; pillow_heif.register_heif_opener()
except Exception:
    pass

ZIP = "/Volumes/Backup Plus/OneDrive_2026-07-22.zip"
CHART = "/Volumes/Backup Plus/SSI chart review.xlsx"

def norm(s):
    return re.sub(r"[^a-z0-9]", "", re.sub(r"_noIF", "", str(s), flags=re.I).lower())

def scan_exif(zip_path=ZIP):
    rows = []
    with zipfile.ZipFile(zip_path) as zf:
        for n in [x for x in zf.namelist() if re.search(r"\.(jpe?g|heic|png)$", x, re.I)]:
            head = zf.open(n).read(262144)          # EXIF lives in the first bytes
            dto = None
            try:
                ex = Image.open(io.BytesIO(head)).getexif()
                tm = {ExifTags.TAGS.get(k, k): v for k, v in ex.items()}
                dto = tm.get("DateTimeOriginal") or tm.get("DateTime")
                if not dto and hasattr(ex, "get_ifd"):
                    ifd = ex.get_ifd(0x8769) or {}
                    dto = {ExifTags.TAGS.get(k, k): v for k, v in ifd.items()}.get("DateTimeOriginal")
            except Exception:
                pass
            base = n.split("/")[-1]
            pid = re.search(r"RU-?A\s?(\d{4})", base) or re.search(r"RU-?A\s?(\d{4})", n)
            day = re.search(r"\bD(\d{1,2})\b", base, re.I) or re.search(r"Day[_ ]?(\d{1,2})", base, re.I)
            rows.append(dict(filename=base, key=norm(base),
                             pid="RU-A" + pid.group(1) if pid else None,
                             day_in_name=int(day.group(1)) if day else None,
                             is_intraop_name=bool(re.search(r"(Pre|Post)[- ]?(Standard|Thermal)", base, re.I)),
                             modality="thermal" if re.search(r"thermal", base, re.I) else "standard",
                             exif_datetime=str(dto) if dto else None))
    ex = pd.DataFrame(rows)
    ex["dt"] = pd.to_datetime(ex.exif_datetime, format="%Y:%m:%d %H:%M:%S", errors="coerce")
    return ex

def surgery_dates(ex):
    dl = ex[ex.dt.notna() & ex.pid.notna() & ex.day_in_name.notna()].copy()
    dl["implied"] = dl.dt.dt.normalize() - pd.to_timedelta(dl.day_in_name, unit="D")
    g = dl.groupby("pid").implied
    return pd.DataFrame({"surgery_date_exif": g.median(), "n_day_imgs": g.size(),
                         "spread_days": (g.max() - g.min()).dt.days})

def chart_surgery_dates(path=CHART):
    ch = pd.read_excel(path)
    ch.columns = [c.strip().lower().replace(" ", "_") for c in ch.columns]
    ch["pid"] = ch.patient_id.astype(str).str.strip()
    sd = pd.to_datetime(ch.ssi_diagnosis_date, errors="coerce") - pd.to_timedelta(
        pd.to_numeric(ch.pod, errors="coerce"), unit="D")
    ch["sd"] = sd
    ch = ch[(ch.sd.dt.year >= 2024) & (ch.sd.dt.year <= 2026)]     # drops year typos
    return ch.dropna(subset=["sd"]).drop_duplicates("pid").set_index("pid").sd

def recover(frame, ex, surg, chart_sd):
    """frame needs: pid, Image_Filename, phase_norm, pod."""
    f = frame.copy()
    f["key"] = f.Image_Filename.map(norm)
    f["fn"] = f.Image_Filename.astype(str)
    sd = f.pid.map(surg.surgery_date_exif).fillna(f.pid.map(chart_sd))
    own = f.key.map(ex.dropna(subset=["dt"]).drop_duplicates("key").set_index("key").dt)
    th = ex[(ex.modality == "thermal") & ex.is_intraop_name & ex.dt.notna()].copy()
    th["tp"] = np.where(th.filename.str.contains("pre", case=False), "pre", "post")
    tmap = th.groupby(["pid", "tp"]).dt.median()
    paired = pd.Series([tmap.get((p, t), pd.NaT) for p, t in zip(f.pid, f.phase_norm)], index=f.index)
    dayname = f.fn.str.extract(r"(?:\bD|Day[_ ]?)(\d{1,2})\b", flags=re.I)[0].astype(float)
    intraop = f.fn.str.contains(r"(?:Pre|Post)[- ]?\s?Standard", case=False, regex=True)
    rec, src = [], []
    for i in f.index:
        if pd.notna(dayname[i]):
            rec.append(float(dayname[i])); src.append("filename_day_number")
        elif pd.notna(own[i]) and pd.notna(sd[i]):
            rec.append(float((own[i].normalize() - sd[i].normalize()).days)); src.append("own_EXIF_vs_surgery_date")
        elif pd.notna(paired[i]) and pd.notna(sd[i]):
            rec.append(float((paired[i].normalize() - sd[i].normalize()).days)); src.append("paired_thermal_EXIF")
        elif intraop[i]:
            rec.append(0.0); src.append("intraoperative_by_naming_convention")
        else:
            rec.append(np.nan); src.append("unresolved")
    f["recovered_pod"], f["evidence"] = rec, src
    return f

# src/data_utils.py
import os, glob
from typing import List, Dict
import pandas as pd

def load_patient_files(root_dir: str, patient_set: set[str]) -> List[Dict]:
    """
    Scan root_dir/<PatientID>/*/registered/ for T1_corrected and T2_FLAIR_corrected.
    Returns a list of dicts with keys 't1' and 't2' for each patient.
    """
    records = []
    for patient in sorted(os.listdir(root_dir)):
        if patient not in patient_set: 
            continue
        pdir = os.path.join(root_dir, patient)
        if not os.path.isdir(pdir): 
            continue

        # find first year folder starting with 20XX
        years = [y for y in sorted(os.listdir(pdir)) if y.startswith("20")]
        for year in years:
            reg = os.path.join(pdir, year, "registered")
            if not os.path.isdir(reg):
                continue

            t1 = glob.glob(os.path.join(
                reg, "T2_FLAIR_corrected_canonical.nii_toMag.nii.gz"))
            t2 = glob.glob(os.path.join(
                reg, "QSM_canonical.nii.gz"))
            
            if t1 and t2:
                records.append({"t1": t1[0], "t2": t2[0]})
                break
    return records

def load_all_patient_files(baseline_dir: str,
                           followup_dir: str,
                           excel_path: str) -> List[Dict]:
    """
    Reads an Excel listing PatientIDs for baseline and follow_up,
    then scans both dirs and merges the results.
    """
    
    df = pd.read_excel(excel_path, dtype=str).rename(columns=str.strip)
    base_set   = set(df.loc[df.FolderType=="baseline","PatientID"])
    follow_set = set(df.loc[df.FolderType=="follow_up","PatientID"])

    recs = []
    recs += load_patient_files(baseline_dir, base_set)
    recs += load_patient_files(followup_dir, follow_set)
    return recs
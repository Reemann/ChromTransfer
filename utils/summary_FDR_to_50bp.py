import os
import glob
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import pybedtools


# ---------- Helpers ----------

def _process_one_chrom(chrom, chrom_df):
    """
    Process a single chromosome in an isolated temp directory.
    Returns a DataFrame with columns: bin_chrom, bin_start, bin_end, score
    """
    # Each worker uses its own tempdir to avoid clashes
    tmpdir = tempfile.mkdtemp(prefix=f"pybedtools_{chrom}_")
    try:
        pybedtools.set_tempdir(tmpdir)

        # Build 50-bp bins across this chromosome
        chrom_end = int(chrom_df["end"].max())
        # Range is [0, chrom_end) in 50-bp steps
        bins = [(chrom, start, start + 50) for start in range(0, chrom_end, 50)]
        bins_df = pd.DataFrame(bins, columns=["chrom", "start", "end"])

        # Prepare BedTools inputs
        # windows: chrom, start, end, score
        windows_df = chrom_df[["chr", "start", "end", '-10lg(FDR)']].rename(
            columns={"chr": "chrom", '-10lg(FDR)': "score"}
        )

        # Ensure integer positions (bedtools expects ints)
        bins_df["start"] = bins_df["start"].astype(int)
        bins_df["end"] = bins_df["end"].astype(int)
        windows_df["start"] = windows_df["start"].astype(int)
        windows_df["end"] = windows_df["end"].astype(int)

        # Convert to BedTool
        fiftybp = pybedtools.BedTool.from_dataframe(bins_df)
        windows = pybedtools.BedTool.from_dataframe(windows_df)

        # Intersect: bin (3 cols) + window (4 cols)
        inter = fiftybp.intersect(windows, wa=True, wb=True)
        inter_df = inter.to_dataframe(disable_auto_names=True, header=None)

        if inter_df.empty:
            # No overlaps on this chromosome; return empty result with correct columns
            return pd.DataFrame(columns=["bin_chrom", "bin_start", "bin_end", "score"])

        inter_df.columns = [
            "bin_chrom", "bin_start", "bin_end",
            "win_chrom", "win_start", "win_end", "score"
        ]

        # Aggregate: mean score per bin
        summarized = (
            inter_df.groupby(["bin_chrom", "bin_start", "bin_end"])["score"]
            .mean()
            .reset_index()
        )
        return summarized, inter_df

    finally:
        # Clean tempdir for this worker
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass


def process_prediction_file(bed_fi, bed_df_total, config):
    """
    Process one predictions file by splitting work across chromosomes.
    """

    out_fi = os.path.join(config.output_dir, f"FDR_{config.target_species}_50bp_bins.bed")
    out_fi_total = os.path.join(config.output_dir, f"FDR_{config.target_species}_50bp_bins_total.bed")

    if os.path.exists(out_fi) and os.path.exists(out_fi_total):
        return f"Skip (exists): {out_fi}, {out_fi_total}"

    print(f"Processing {bed_fi}")

    # Load prediction CSV, index by region_num, and join genomic coords
    bed_df = pd.read_csv(bed_fi, sep=",", header=0)
    bed_df.set_index('region_num', inplace=True)
    # bed_df = pd.concat([bed_df_total[['chr', 'start', 'end']], bed_df[['prediction']]], axis=1)
    bed_df = pd.concat([bed_df_total[['chr', 'start', 'end']], bed_df[['-10lg(FDR)']]], axis=1)

    # Parallel per chromosome
    chroms = bed_df['chr'].dropna().unique().tolist()

    results = []
    retults_total = []
    with ProcessPoolExecutor(max_workers=config.predict_num_workers) as ex:
        futs = {ex.submit(_process_one_chrom, chrom, bed_df[bed_df['chr'] == chrom].copy()): chrom
                for chrom in chroms}
        for fut in as_completed(futs):
            chrom = futs[fut]
            try:
                df_part, df_part_total = fut.result()
                if df_part is not None and not df_part.empty:
                    results.append(df_part)
                if df_part_total is not None and not df_part_total.empty:
                    retults_total.append(df_part_total)
            except Exception as e:
                print(f"[WARN] Chromosome {chrom} failed: {e}")

    if not results:
        # No overlaps at all; write empty file with no header
        open(out_fi, "w").close()
        return f"Wrote empty: {out_fi}"

    summarized_all = pd.concat(results, axis=0, ignore_index=True)
    total_all = pd.concat(retults_total, axis=0, ignore_index=True)

    # Sort for tidy output (optional)
    summarized_all.sort_values(["bin_chrom", "bin_start", "bin_end"], inplace=True)
    total_all.sort_values(["bin_chrom", "bin_start", "bin_end"], inplace=True)

    # Write as BED (chrom, start, end, score)
    summarized_all.to_csv(out_fi, sep="\t", header=False, index=False)
    total_all.to_csv(out_fi_total, sep="\t", header=False, index=False)
    
    return f"Done: {out_fi}, {out_fi_total}"



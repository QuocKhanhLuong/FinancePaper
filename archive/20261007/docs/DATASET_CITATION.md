# Manuscript citation/access package — Freddie SFLLD

**Planned dataset, not yet used.** No real files were downloaded in this stage.
Replace future tense with past tense only after the local receipt and hashes exist.

## Dataset description (ready for protocol/manuscript methods plan)

We will use the Freddie Mac Single-Family Loan-Level Dataset, Standard annual
samples, Release47 (2026-07-29), with performance through2026-03-31. Our frozen
cohort comprises15 annual samples (2000–2008,2011,2014,2017,2020–2022), nominally
750,000 unique source loans. Origination attributes and monthly servicing records
will support a retrospective six-month observation window and an ensuing12-month
observed90+DPD endpoint before termination. Exact eligible/labelled loan and monthly
row counts will be supplied after intake. This is mortgage serious-delinquency
prediction, distinct from credit-card default and corporate bankruptcy.

## License/use statement

Freddie Mac's [SFLLD terms, effective November2025](https://capitalmarkets.freddiemac.com/crt/docs/pdfs/fre_terms_conditions_sflld.pdf)
expressly permit noncommercial academic/research results and related derived
products to be made public subject to non-reconstruction and non-identification
conditions. We will distribute original code and reviewed aggregate scientific
outputs only. Individual loan records, transformed rows, completion references,
record-level explanations and model artifacts are not supplied by this repository.
This is not a Creative Commons/open-data license; access also remains subject to
the provider's website and applicable registration terms. See the complete
[permission matrix](DATASET_PROVENANCE_AUDIT.md).

## Citation

No dataset DOI or provider-prescribed citation string was verified. The following
is our bibliographic citation of the provider's named, dated release and guide,
not a claim that Freddie supplied an official citation format.

**APA (dataset):** Freddie Mac. (2026). *Single-Family Loan-Level Dataset: Standard
annual samples (Release 47, July 29, 2026)* [Data set].
https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset

**Supporting documentation:** Freddie Mac. (2026). *Single-Family Loan-Level
Dataset: General User Guide (Release 47, July 2026).*
https://www.freddiemac.com/fmac-resources/research/pdf/general_user_guide_july_2026.pdf

**Plain manuscript:** Freddie Mac, *Single-Family Loan-Level Dataset*, Standard
annual samples, Release47, released29 July2026; General User Guide, July2026;
performance cutoff31 March2026. Official landing page:
https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset.

```bibtex
@misc{freddiemac_sflld_data_r47_2026,
  author = {{Freddie Mac}},
  title = {Single-Family Loan-Level Dataset: Standard Annual Samples},
  year = {2026},
  month = jul,
  howpublished = {Data set, Release 47},
  url = {https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset},
  note = {Released 2026-07-29; performance cutoff 2026-03-31;
          planned vintages 2000--2008, 2011, 2014, 2017, 2020--2022}
}

@manual{freddiemac_sflld_r47_2026,
  author = {{Freddie Mac}},
  title = {Single-Family Loan-Level Dataset: General User Guide},
  year = {2026},
  month = jul,
  edition = {Release 47},
  url = {https://www.freddiemac.com/fmac-resources/research/pdf/general_user_guide_july_2026.pdf},
  note = {Standard annual samples; release date 2026-07-29;
          performance cutoff 2026-03-31. Documentation accessed 2026-10-03}
}
```

Also retain the [release notes](https://www.freddiemac.com/fmac-resources/research/pdf/release_notes.pdf)
and cite the landing page; the notes URL is mutable, hence the explicit release.

## Version and access statement

Planned ZIPs: `sample_2000.zip` through `sample_2008.zip`, `sample_2011.zip`,
`sample_2014.zip`, `sample_2017.zip`, `sample_2020.zip`, `sample_2021.zip`,
`sample_2022.zip`. Each contains the same year's `sample_orig_YYYY.txt` and
`sample_perf_YYYY.txt` in the July2026 layout. Raw SHA256 and download date:
**PENDING OFFICIAL ACCESS**. Do not fabricate these from documentation access dates.

Reproducers register/sign in to [Clarity through the official dataset page](https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset),
accept the provider's current terms themselves, obtain the exact official release,
and run our offline preparer. We do not distribute files or bypass the login.
If Release47 is no longer obtainable, exact reproduction is unavailable until an
authorized copy is supplied; do not silently substitute a later revised release.

## Data Availability / governance paragraph

The underlying mortgage data are owned by Freddie Mac and are available through
its registered-access Single-Family Loan-Level Dataset service, subject to its
terms. We provide preprocessing code, a frozen cohort specification and, after
authorized intake, checksum metadata to support reproduction. We do not redistribute
raw or processed loan records. Analysis is confined to mortgage credit-performance
research; no linkage to identify borrowers is performed. Natural missing values
remain unknown and only artificially hidden observed fields have verification truth.
Only aggregate, non-identifying, non-reconstructive research outputs are eligible
for publication. Institutional ethics/data-governance review, if required, must be
reported as actually obtained; no approval or exemption is claimed here.

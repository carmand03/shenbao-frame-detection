Metadata association outputs
1. association_coverage.csv, metadata_conflicts.csv, prediction_quality.csv, matched_segment_metadata.csv: audit eligibility and denominators.
2. group_frame_prevalence.csv: counts, prevalence, cluster-bootstrap intervals, lift, group-vs-rest effects and descriptive odds ratios.
3. clustered_unadjusted_tests.csv: omnibus predictor associations with document clustering and BH-FDR.
4. clustered_adjusted_tests.csv and adjusted_odds_ratios.csv: joint adjustment for relevance, gender and decade.
5. paired_model_differences.csv and clustered_model_interaction_tests.csv: paired model prevalence and association differences.
6. conditional_polarity_by_group.csv: polarity distributions among segments containing each base frame.
7. iid_association_diagnostics.csv and presence_residuals.csv: contingency diagnostics, not document-clustered inference.
8. all_eligible_sensitivity.csv: own-model eligible populations; denominators may differ.
9. PNGs: saved visualizations; association_provenance.json: reproducibility settings.
Intervals are pointwise. Skipped tests are not evidence of no association. All analyses are associational, not causal.
Matched segments: 4,599; documents: 4,381; metadata reference: Gemma 3 27B.

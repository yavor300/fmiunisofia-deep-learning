# Experiment Configs

Keep the existing files. They serve different purposes:

- `000_*`: baseline row for the model report.
- `001_*`, `011_*`, `012_*`, `013_*`: short smoke/preprocessing experiments.
- `008_*`, `009_*`, `010_*`: earlier loss-function experiments.
- `020_*` to `025_*`: final architecture comparison.
- `026_*` to `027_*`: final scheduler comparison using U-Net + ResNet34.
- `028_*` to `031_*`: final loss comparison using U-Net + ResNet34.
- `032_*` to `033_*`: final preprocessing comparison using U-Net + ResNet34.

The final report does not need every historical file to be rerun. Prefer the `020_*` to
`033_*` configs for the polished comparison, and use the earlier configs when you need
quick smoke runs or phase-specific evidence.

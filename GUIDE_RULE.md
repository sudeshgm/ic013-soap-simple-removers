# How SOAP picks the "Max. Sal. PC response" guide (IC013 finding, SOAP @06ede71, arXiv 2604.00172)

**Code.** `main_salient.py --pca_guide_dim d [--flip_guide]`:
- The guide is `m = feat @ v_d + b` with `b = -mean @ v_d` (welford.get_truncated_weights_and_biases_at_indices).
  `d` is a raw index into the ascending eigh, so -1 is the top PC and -2 the 2nd.
- `--flip_guide` negates m.
- ncut.select_foreground('pca') takes the seed as argmax(m), and the bipartition side containing the seed becomes the foreground.
- No code anywhere chooses `d` or the flip. They are hard-coded constants in notebook.ipynb
  (`pca_guide_dim = -2 # -2 for DINOv2, -1 for CAPI`, `flip_guide = True`). Nothing is given for MAE or any other model.

**Paper.**
- App. E.1 / Table E.2: select the foreground "based on the patch with the highest response <v_d, z>", using "the strongest
  salient principal component ... for each model".
- Salient PCs are described qualitatively: components with centre-biased activations on real images that are absent on
  synthetic inputs (Fig. B.2).
- "I-JEPA did not have a good salient principal component" shows the choice was made by inspection.
- The flip (eigenvector sign) is never mentioned.

**Conclusion.** The MAE guide PC and flip cannot be set by a rule that SOAP's code or paper defines.

**Tried, not adopted.** scripts/salient_pc_rule.py is our own operationalisation:
- It scores the top-8 ranked PCs by centre bias on the real-image maps minus the absolute centre bias on the synthetic maps,
  and flips when the real centre bias is negative.
- On the Block-1 ImageNet fits it does NOT recover the known choices: DINOv2 gives -5 flipped (known: -2 flipped);
  CAPI gives -7 flipped (known: -1).
- So it is not a faithful reproduction of SOAP's rule, and it must not be used for the gate.

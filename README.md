# LaGSplat

**LaGSplat: Inferring Physics-Governed Interactive Simulation from Monocular
Video Using Latent Lagrangian Gaussian Splatting**
Louen Pottier. Preprint: **[arXiv:2608.16324](https://arxiv.org/abs/2608.16324)**.

LaGSplat infers interactive, physics-governed dynamics from one or a few
monocular videos. A low-dimensional latent state `q` plays two roles at once: it
is the generalised coordinate of a learned dissipative Lagrangian, and the
conditioning variable of a Gaussian Splatting decoder. Because the primitives of
that decoder are explicit points `mu_i(q)` that move with the object, a force `f`
applied anywhere in the image pulls back into a latent generalised force
`J(q)^T f` and enters the equations of motion, which pixel-space or neural-field
decoders cannot do. At inference the filmed object answers a push that was never
measured, annotated or seen during training.

This repository reproduces **one result of the paper**: the multi-step
prediction on the **two-segment** soft continuum robot of Krauss et al. 2026,
evaluated on their data and their split. The other experiments of the paper are
not part of it.

## Results

Two segments, four chambers. MSE in pixels of `[0, 1]`, native 32 x 32 rendering.

| Model | d | AE floor (1e-5) | MSE 0.5 s (1e-3) | freq. error |
|---|---:|---:|---:|---:|
| Osc. + deconv | 10 | 5.43 | 22.7 | 76.8 % |
| Osc. + ABCD (VON) | 10 | 14.5 | 6.56 | 31.4 % |
| Koopman + deconv | 10 | 3.95 | 5.66 | 18.3 % |
| Koopman + ABCD | 10 | 5.17 | 0.984 | 8.3 % |
| **LaGSplat** | **4** | 6.31 | **0.696** | **3.9 %** |

Only the LaGSplat row is produced here. The baseline MSE values are the ones
published by the authors; their AE floors and frequency errors are our own
measurements on their released checkpoints. The frequency error compares a free
rollout to the video over the depressurised tail of the clip, with the same
spectral estimator on both: 1.5621 Hz observed against 1.5016 Hz held. Section 4
of the preprint defines the metrics.

## Operating point

| | value |
|---|---|
| Data | `scr_dataset_raw_2seg_32x32_59fps.npz` (Zenodo, `processed_data.zip`) |
| Images | 32 x 32 RGB, 59.94 fps (`DT = 1001/60000`) |
| Pressures | `p1..p4` from the NPZ, aligned by the authors, normalised by 101 325 Pa |
| Split | first 80 % train, last 20 % validation, contiguous (43 872 / 10 968 frames) |
| Latent | `d = 4`, frozen PCA whitening after the autoencoder |
| Decoder | Gaussian Splatting 2D+t (gsplat), 2048 Gaussians |
| LNN | learned mass `M(q)`, full dissipation `C(q)`, invex deformation potential, invex pressure forcing |
| LNN training | `lr 1e-3`, `c0 = 1`, `sigma 1`, 500 epochs, seed 0, from scratch, frozen AE |

## Quick start

```bash
py -m pip install -r requirements.txt        # read the file: torch and gsplat first

py scripts/fetch_data.py --seg 2seg          # Zenodo, SHA256 check, 80/20 split

py scripts/run_2seg_npz.py                   # the whole protocol
py scripts/run_2seg_npz.py --dry-run         # print the commands without running them
py scripts/run_2seg_npz.py --only 5          # the MSE evaluation alone
```

The protocol is six steps: autoencoder, frozen PCA whitening of the latent,
visibility metric of the decoder, LNN, then the two evaluations. Steps 1 and 4
take several GPU hours. `fetch_data.py --zip <path>` avoids re-downloading the
4.5 GB Zenodo archive if it is already on the machine.

`gsplat` needs CUDA and a compilation step. Without it the code falls back on an
in-house pure-torch decoder, about 100 times slower and **not** weight
compatible with gsplat checkpoints. It is there to read and run the code, not to
reproduce the figures: set `DEC2PT_BACKEND = 'gsplat'` in the case config to
turn any silent fallback into an `ImportError`.

## Layout

```
code/            the LaGSplat implementation, and code/config.py, the base config
cases/krauss2026_2seg_npz/
                 config.py of the case, overriding only what it changes;
                 data/ and checkpoints/ are produced locally
scripts/         fetch_data (Zenodo, checksum, split), run_2seg_npz (the protocol)
```

`code/` is the research implementation reduced to what this experiment needs.
A few optional code paths are still described in the comments but are not part
of it, and not part of the published protocol either: the curved decoder metric
(`LNN_METRIC_FROM_DECODER`), the MLP decoder, and the flow-based trimming of the
startup transient.

## Third-party data

The dataset comes from Zenodo, DOI `10.5281/zenodo.17812071`, archive
`processed_data.zip` (about 4.5 GB), under CC BY-ND 4.0, and `fetch_data.py`
checks its SHA256.

The reference implementation of Krauss et al. is at
[github.com/UThenrik/visual_oscillators_for_SCR](https://github.com/UThenrik/visual_oscillators_for_SCR)

## License

The code in this repository is released under the LaGSplat Research License
(`LICENSE`): free use, reproduction and modification for **non-commercial
academic research**, including verifying and reproducing the published results.
Commercial use, integration into a product or a service, and redistribution to
third parties require a separate written licence from the holder (CEA). No
patent licence is granted. Third-party data keeps its own terms, see above.

## Cite this work

```bibtex
@article{pottier2026lagsplat,
  title   = {LaGSplat: Inferring Physics-Governed Interactive Simulation from
             Monocular Video Using Latent Lagrangian Gaussian Splatting},
  author  = {Pottier, Louen},
  journal = {arXiv preprint arXiv:2608.16324},
  year    = {2026},
  url     = {https://arxiv.org/abs/2608.16324}
}
```

If you use the data, cite the dataset and the paper it comes with:

```bibtex
@dataset{krauss2026scrdataset,
  title     = {Soft Continuum Robot Dataset (SCR)},
  author    = {Krauss, Henrik and Licher, Johann and Takeishi, Naoya
               and Raatz, Annika and Yairi, Takehisa},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.17812071},
  url       = {https://doi.org/10.5281/zenodo.17812071},
  note      = {CC BY-ND 4.0}
}

@article{krauss2026von,
  title   = {Learning Visually Interpretable Oscillator Networks for
             Soft Continuum Robots from Video},
  author  = {Krauss, Henrik and Licher, Johann and Takeishi, Naoya
             and Raatz, Annika and Yairi, Takehisa},
  journal = {IEEE Robotics and Automation Letters},
  year    = {2026},
  doi     = {10.1109/LRA.2026.3703241},
  note    = {arXiv:2511.18322}
}
```

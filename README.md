# Challenge 2.2 — forecasting a pitching flat-plate wake

Entry for Challenge 2.2 of the Data-Driven Reduced-Complexity Modeling of
Fluid Flows community challenge (arXiv:2601.06183).

Task: given 70 snapshots of velocity over a flat plate pitching about its
midchord (2D DNS, Re = 100), forecast the next 130 snapshots, for both
in-sample and out-of-sample pitching parameters. The prescribed kinematics
alpha(t) and alpha_dot(t) are supplied over the whole forecast window.

## Layout

    driver.py          top-level entry point (challenge requirement)
    utilities/         all reusable code
    experiments/       standalone studies, each self-contained
    notes/             handover documents, working notes
    results/           submission files
    data/              not tracked; see notes/DATA.md

## Status

Pre-Gate-0. No challenge data downloaded yet.

## Pre-registration

PREREGISTRATION.md records predictions made before the data was seen.
It is not edited after the data arrives; corrections go in a dated
appendix so the original text stays readable.

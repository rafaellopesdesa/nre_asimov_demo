# Finite Asimov construction with NRE

Train $r_{s,\boldsymbol\psi}$, construct $\mathcal A_M$, and compare $t_{\mu,A}$ and $q_0$.

```bash
python -m pip install -r requirements.txt
python -m jupyterlab NRE_Asimov.ipynb
```

Run the cells in order. Set `DEVICE` and `TRAIN` in the first cell; `TRAIN = False`
loads the saved selection and NRE weights from `workspace/`.

## RooFit same-sample normalizer

The physics example uses [`RooSameSampleNormalizer`](roofit/RooSameSampleNormalizer.h)
to normalize each process shape as its nuisance parameters change. Its value
is $Z(\\boldsymbol\\alpha)=\\sum_i m_i V_i(\\boldsymbol\\alpha)$, where the fixed
measure $m_i=u_i\\widehat r_i$ is the normalized reference weight times the
nominal process ratio. Divide $\\widehat r_i V_i$ by this scalar; keep the
physical yield outside the division. The measure is defined on the selected
reference sample, not by the generated Asimov weights.

The [implementation](roofit/RooSameSampleNormalizer.cxx) uses the multiplicative
HistFactory code-5 interpolation of the analysis (ROOT 6.34.10): nominal value
one, boundaries at $\\pm1$, and positive down/up anchors. For $N$ events and $K$
ordered nuisance parameters, each anchor array has $KN$ elements; entry
`j*N+i` is the anchor for nuisance `j` and event `i`.

With ROOT, RooFit and a C++ compiler available, run this PyROOT example from
the repository root. ACLiC builds the class and its dictionary; load it before
opening a workspace that uses the class.

```python
import math
import ROOT

assert ROOT.gSystem.CompileMacro("roofit/RooSameSampleNormalizer.cxx", "kO")
alpha = ROOT.RooRealVar("alpha", "shape nuisance", 0., -5., 5.)
z = ROOT.RooSameSampleNormalizer("z", "shape integral", ROOT.RooArgList(alpha),
                                 [0.25, 0.75], [0.8, 0.9], [1.1, 1.2])
alpha.setVal(1.)
assert math.isclose(z.getVal(), 0.25 * 1.1 + 0.75 * 1.2)
alpha.setVal(0.4)
assert math.isclose(z.getVal(), z.directValue())  # independent ROOT interpolation
```

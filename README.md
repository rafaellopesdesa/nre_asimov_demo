# Finite Asimov construction with NRE

Train $r_{s,\boldsymbol\psi}$, construct $\mathcal A_M$, and compare $t_{\mu,A}$ and $q_0$.

```bash
python -m pip install -r requirements.txt
python -m jupyterlab NRE_Asimov.ipynb
```

Run the cells in order. Set `DEVICE` and `TRAIN` in the first cell; `TRAIN = False`
loads the saved selection and NRE weights from `workspace/`.

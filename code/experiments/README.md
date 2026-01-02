# Experimentos reproducibles (Python)

Este directorio contiene código para **demostrar empíricamente** el marco del paper (distinción \(E_f/E_i/E_p\), estabilidad en \(\Pi\), y contraste con copia nula/probe).

## Instalación

En la raíz del repo:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Ejecutar (genera figuras en `figures/`)

```bash
python -m experiments.run_all --out figures --trials 200 --seed 0
```

Genera gráficos `.png` listos para incluir en LaTeX.

## Qué produce

- **Escenario espurio (finito-muestra)**: contrasta una variable con señal vs una de ruido, mostrando estabilidad \(\Pr_{E\sim\Pi}[\Delta(E)>\varepsilon]\) vs \(n\), y la diferencia entre drop-one \(\Delta\) y swap/probe \(\widetilde{\Delta}\).
  - `figures/spurious_signal_vs_noise_stability_curve.png`
  - `figures/spurious_signal_vs_noise_delta_hist.png`
- **Escenario leakage/protocolo**: una variable (ID de grupo) es altamente predictiva bajo split aleatorio, pero no bajo holdout por grupos, mostrando explícitamente dependencia del protocolo \(\Pi\).
  - `figures/leakage_protocol_comparison.png`
- **Escenario XOR**: señal puramente condicional (marginalmente nula) y no explotable por un modelo lineal, ilustrando “informativa pero no predictiva” (dependencia de la familia de modelos).
  - `figures/xor_not_exploitable_logreg.png`

## Incluir en LaTeX (ejemplo)

```tex
\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{figures/leakage_protocol_comparison.png}
\caption{Dependencia del protocolo \(\Pi\): una variable de ID de grupo parece “relevante” bajo split aleatorio, pero pierde estabilidad bajo holdout por grupos.}
\label{fig:leakage_protocol}
\end{figure}
```



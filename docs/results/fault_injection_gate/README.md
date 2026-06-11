# Fault-Injection GO/NO-GO Gate

Single-fault MAP localization accuracy under noisy observations:
classical undirected metric-dimension placement vs directed/noisy
JS-divergence placement. Decision rule: GO if the mean directed
advantage is at least 5% absolute.

**Decision: NO-GO** — mean classical 24.7%, mean directed 25.0%, advantage +0.3%.

| Family | Noise | Budget | Classical | Directed | Δ |
| --- | ---: | ---: | ---: | ---: | ---: |
| layered | 0.02 | 2 | 19.2% | 20.9% | +1.7% |
| layered | 0.02 | 3 | 25.2% | 26.7% | +1.5% |
| layered | 0.02 | 4 | 30.3% | 35.5% | +5.1% |
| layered | 0.05 | 2 | 18.2% | 18.8% | +0.6% |
| layered | 0.05 | 3 | 23.8% | 24.6% | +0.8% |
| layered | 0.05 | 4 | 28.0% | 32.4% | +4.3% |
| layered | 0.10 | 2 | 16.6% | 16.9% | +0.3% |
| layered | 0.10 | 3 | 19.9% | 20.9% | +1.0% |
| layered | 0.10 | 4 | 23.8% | 28.1% | +4.3% |
| layered | 0.20 | 2 | 13.8% | 13.2% | -0.6% |
| layered | 0.20 | 3 | 15.3% | 16.8% | +1.5% |
| layered | 0.20 | 4 | 18.1% | 19.1% | +1.0% |
| chain | 0.02 | 2 | 22.9% | 22.6% | -0.3% |
| chain | 0.02 | 3 | 32.5% | 31.3% | -1.2% |
| chain | 0.02 | 4 | 40.9% | 39.0% | -2.0% |
| chain | 0.05 | 2 | 21.8% | 22.9% | +1.0% |
| chain | 0.05 | 3 | 30.4% | 30.1% | -0.3% |
| chain | 0.05 | 4 | 37.8% | 35.4% | -2.4% |
| chain | 0.10 | 2 | 19.6% | 21.2% | +1.6% |
| chain | 0.10 | 3 | 27.3% | 26.5% | -0.8% |
| chain | 0.10 | 4 | 33.1% | 31.5% | -1.7% |
| chain | 0.20 | 2 | 16.0% | 17.2% | +1.3% |
| chain | 0.20 | 3 | 21.1% | 20.8% | -0.3% |
| chain | 0.20 | 4 | 25.4% | 23.4% | -1.9% |
| spider | 0.02 | 2 | 22.2% | 20.6% | -1.6% |
| spider | 0.02 | 3 | 32.0% | 29.5% | -2.5% |
| spider | 0.02 | 4 | 38.0% | 36.9% | -1.2% |
| spider | 0.05 | 2 | 20.8% | 21.6% | +0.8% |
| spider | 0.05 | 3 | 29.6% | 29.2% | -0.4% |
| spider | 0.05 | 4 | 34.2% | 33.3% | -0.9% |
| spider | 0.10 | 2 | 19.1% | 21.1% | +1.9% |
| spider | 0.10 | 3 | 25.8% | 25.4% | -0.4% |
| spider | 0.10 | 4 | 30.0% | 30.2% | +0.2% |
| spider | 0.20 | 2 | 15.7% | 16.1% | +0.4% |
| spider | 0.20 | 3 | 20.2% | 19.5% | -0.7% |
| spider | 0.20 | 4 | 23.2% | 21.3% | -1.9% |
| arborescence | 0.02 | 2 | 21.2% | 21.1% | -0.1% |
| arborescence | 0.02 | 3 | 28.7% | 29.7% | +1.0% |
| arborescence | 0.02 | 4 | 37.1% | 39.1% | +2.0% |
| arborescence | 0.05 | 2 | 20.2% | 20.8% | +0.6% |
| arborescence | 0.05 | 3 | 26.2% | 27.9% | +1.7% |
| arborescence | 0.05 | 4 | 33.5% | 35.9% | +2.4% |
| arborescence | 0.10 | 2 | 18.6% | 18.2% | -0.4% |
| arborescence | 0.10 | 3 | 23.8% | 24.8% | +1.0% |
| arborescence | 0.10 | 4 | 28.9% | 28.1% | -0.8% |
| arborescence | 0.20 | 2 | 15.3% | 15.0% | -0.3% |
| arborescence | 0.20 | 3 | 17.3% | 17.5% | +0.2% |
| arborescence | 0.20 | 4 | 21.2% | 20.4% | -0.7% |

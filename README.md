# VolGuard ⌱

> Mitigate  high volatility in DeFi markets using AI-powered market
makers.

## Overview

Defi is infamous for having high volatility, in large part due to the
small and sporadic number of transactions in varying sizes. Traditional 
Financial markets have a continuous stream of market orders, largely 
facilitated by entities such as High Frequency Trading firms. VolGuard 
aims to reduce the volatile and sproadic nature of the DeFi markets by
using ML-driven market maker contracts, which attempt to predict future 
volatility and brace against it before it occurs.

VolGuard ingests DeFi data from protocols such as Uniswap, and
generates an off-chain AI prediction of future volatility. Based on this 
prediction, a smart contract system is invoked to deploy a corresponding
strategy that best capitalizes on the anticipated market conditions.
<br/>

## Architecture
<br/>
<img src="/arch_diagram.png" width="40%"/>

## Setup
1. Create new folder and clone repository into folder using:

```
git clone https://github.com/AI-and-Blockchain/S23_VolGuard.git
```

2. Install dependdencies via the following pip command:
```
pip install scikit-learn torch pandas numpy parquet pyarrow plotly Flask matplotlib
```
 
3.  Within that folder create a new folder called historical_files
4.  Run either historicalData.js or hourlyhistoricalData.js with node to fill historical_files with csv data
5. Run any of the pytorch AI algorithms in the Ai folder to get your prediction

## Stack

- [Amberdata](https://amberdata.io/) - A flexible digital asset API with DeFi data out of the box.
- [PyTorch](https://pytorch.org/) - A Python library for building fast, end to end ML models.
- [Python](https://python.org/) - High level general purpose programming language, popular for ML workloads.
- [Solidity](https://soliditylang.org/) - Programming language designed for creating smart contracts on Ethereum.


## License

`VolGuard` is free and open source software under the Apache 2.0 License.

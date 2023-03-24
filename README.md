# VolGuard ⌱

> Mitigate the high volatility of DeFi markets using AI-powered market
makers.

## Overview

Defi is infamous for having high volatility, in large part due to the
small and sporadic number of transactions in varying sizes. Traditional 
Financial markets have a continuous stream of market orders, largely 
facilitated by entities such as High Frequency Trading firms. VolGuard 
aims to reduce the volatile and sproadic nature of the DeFi markets by
using ML-driven market maker contracts, which attempt to predict future 
volatility and brace against it before it occurs.

VolGuard takes in DeFi data from protocols such as Uniswap, and
generates an off-chain AI prediction of future volatility. Based on that
prediction, a smart contract system is called to deploy a corresponding 
strategy to best take advantage of the future possibilities.
<br/>

## Architecture
<br/>
<img src="/arch_diagram.png" width="40%"/>

## Stack

- [Amberdata](https://amberdata.io/) - A flexible digital asset API with DeFi data out of the box.
- [Tensorflow](https://tensorflow.org/) - A Python library for building fast, end to end ML models.
- [Python](https://python.org/) - High level general purpose programming language, popular for ML workloads.
- [Solidity](https://soliditylang.org/) - Programming language designed for creating smart contracts on Ethereum.


## License

`VolGuard` is free and open source software under the Apache 2.0 License.

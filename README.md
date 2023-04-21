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

## System Vision

The vision behind VolGuard is to create an easy-to-use predictive market
making system that users can deploy and self-host in a completely decentralized
mannner, allowing many different users to participate, modify, or create 
new methods for reducing volatility in DeFi markets.

## Architecture
<br/>
<img src="/architecture.png"/>

## Getting started
### Metamask

Follow this [link](https://support.metamask.io/hc/en-us/articles/360015489531-Getting-started-with-MetaMask) and download metamask that works with your browser. 

Go to https://ai-and-blockchain.github.io/S23_VolGuard/

While logged into the Metamask extension, press connect wallet in the top right corner

<img src="/connectwallet.png"/>

Then select your datasource, Ai model to use, and strategy

<img src="/selectmodels.png"/>

Then hit deploy, and check your latest activity on Etherscan.io to find your contract

<img src="/gotoetherscan.png"/>

You can now fund the contract with whatever amount of ETH you would like, and the contract will automatically update your position based on your selected model's predictions
## Self hosting
To host the backend locally first run

git clone https://github.com/AI-and-Blockchain/S23_VolGuard.git

Then within the cloned repository run 

pip install -r /backend/requirements.txt

Then once all the pip dependencies are downloaded you can run the flask application with the command

python3 /backend/app.py



## User stories
[VolGuard](https://ai-and-blockchain.github.io/S23_VolGuard/)

## Stack

- [Amberdata](https://amberdata.io/) - A flexible digital asset API with DeFi data out of the box.
- [PyTorch](https://pytorch.org/) - A Python library for building fast, end to end ML models.
- [Python](https://python.org/) - High level general purpose programming language, popular for ML workloads.
- [Solidity](https://soliditylang.org/) - Programming language designed for creating smart contracts on Ethereum.


## License

`VolGuard` is free and open source software under the Apache 2.0 License.

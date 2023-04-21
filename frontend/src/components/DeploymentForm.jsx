import Dropdown from "./Dropdown";
import axios from "axios";
import React, { useEffect, useState, useContext } from "react";
import WalletContext from "./WalletContext";
import Web3 from "web3";
import contractAbi from "../contract_files/abi.json";
import contractBytecode from "../contract_files/bytecode.json";

const DataSourceOptions = [{ id: 1, name: "amber-data" }];
const OracleAddressOptions = [
  {
    id: 1,
    name: "3 hours",
    address: "0x0436411B895786d063f5E1F432cba2B5991c0D1C",
  },
  {
    id: 2,
    name: "6 hours (reccommended)",
    address: "0x87517E0C222541aC195e321041eE91239CaDe5ec",
  },
  {
    id: 3,
    name: "12 hours",
    address: "0xB790452e13FeD72411efc7Dbd5dF305aE950a5eE",
  },
  {
    id: 4,
    name: "24 hours",
    address: "0xa888AcaB87f9D775cB09eDE5fDdF5Fb8148b5Ab8",
  },
];
const StrategyOptions = [{ id: 1, name: "Gammaswap" }];

const uniswapRouterAddress = "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D";
const ethAddress = "0x0000000000000000000000000000000000000000";
const usdcAddress = "0x07865c6e87b9f70255377e024ace6630c1eaa37f";
const gasEstimate = 12000000;

function DeploymentForm() {
  const [selectedDataSource, setSelectedDataSource] = useState(
    DataSourceOptions[0]
  );
  const [selectedAIModel, setSelectedAIModel] = useState(
    OracleAddressOptions[0]
  );
  const [selectedStrategy, setSelectedStrategy] = useState(StrategyOptions[0]);
  const [web3, setWeb3] = useState(null);
  const [initialBalance, setInitialBalance] = useState(0);

  useEffect(() => {
    if (window.ethereum) {
      setWeb3(new Web3(window.ethereum));
    }
  }, []);

  async function deployContract(
    accountAddress,
    selectedOracleAddress,
    uniswapRouterAddress,
    ethAddress,
    usdcAddress,
    initialBalanceWei,
    gasEstimate
  ) {
    try {
      if (!web3) {
        alert("Web3 not detected. Please install MetaMask.");
        return;
      }
      // Get the contract instance
      const GammaTrader = new web3.eth.Contract(contractAbi);

      // Deploy the contract with the required constructor arguments
      const deployedContract = await GammaTrader.deploy({
        data: contractBytecode,
        arguments: [
          selectedOracleAddress,
          uniswapRouterAddress,
          ethAddress,
          usdcAddress,
        ],
        value: initialBalanceWei,
      }).send({
        from: accountAddress,
        gas: gasEstimate,
      });
    } catch (error) {
      console.error("Error deploying contract:", error);
      console.log("Selected Oracle Address:", selectedOracleAddress);
      console.log("Uniswap Router Address:", uniswapRouterAddress);
      console.log("ETH Address:", ethAddress);
      console.log("USDC Address:", usdcAddress);
      console.log("Initial Balance (Wei):", initialBalanceWei);
      console.log("Account Address:", accountAddress);
      console.log("Gas Estimate:", gasEstimate);
      alert("Contract deployment failed.");
    }
  }

  const { connected, account } = useContext(WalletContext);
  const handleDeploy = async () => {
    if (!account) {
      alert("Please connect your wallet first.");
      return;
    }
    const balance = await web3.eth.getBalance(account);
    const initialBalanceWei = web3.utils.toWei(
      initialBalance.toString(),
      "ether"
    );
    if (web3.utils.toBN(initialBalanceWei).gt(web3.utils.toBN(balance))) {
      alert("Insufficient balance.");
      return;
    }
    deployContract(
      account,
      selectedAIModel.address,
      uniswapRouterAddress,
      ethAddress,
      usdcAddress,
      initialBalanceWei,
      gasEstimate
    );
  };

  return (
    <div className="container mx-auto max-w-4xl px-4">
      <div className="bg-white rounded-lg  shadow-lg p-6 mt-8">
        <h2 className="text-xl font-semibold mb-4">Deploy</h2>
        <div className="grid grid-cols-1 gap-10">
          <Dropdown
            label="Data-source"
            options={DataSourceOptions}
            selectedOption={selectedDataSource}
            setSelectedOption={setSelectedDataSource}
          />
          <Dropdown
            label="AI-model"
            options={OracleAddressOptions}
            selectedOption={selectedAIModel}
            setSelectedOption={setSelectedAIModel}
          />
          <Dropdown
            label="Strategy"
            options={StrategyOptions}
            selectedOption={selectedStrategy}
            setSelectedOption={setSelectedStrategy}
          />
          <div>
            <label
              htmlFor="initial-balance"
              className="block text-sm font-medium"
            >
              Initial balance (ETH)
            </label>
            <input
              type="number"
              step="any"
              placeholder="Initial ETH balance"
              value={initialBalance}
              className="mt-1 block w-full py-2 px-3 border border-gray-300 bg-white rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
              onChange={(e) => setInitialBalance(e.target.value)}
            />
          </div>
        </div>
        <button
          type="button"
          className="w-full sm:w-auto mt-6 py-2 px-4 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-md"
          onClick={handleDeploy}
        >
          Deploy
        </button>
      </div>
    </div>
  );
}

export default DeploymentForm;

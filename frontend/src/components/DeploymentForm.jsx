import Dropdown from "./Dropdown";
import axios from "axios";
import React, { useEffect, useState, useContext } from "react";
import WalletContext from "./WalletContext";
import Web3 from "web3";
import contractAbi from "../contract_files/abi.json";
import contractBytecode from "../contract_files/bytecode.json";

const DataSourceOptions = [{ id: 1, name: "amber-data" }];
const OracleAddressOptions = [
  { id: 1, name: "3 hours", address: "0x2088c6c71c7e2609a98bFaf89AC5Ed618518Da74" },
  { id: 2, name: "6 hours (reccommended)", address: "0xSimpleOracleAddress2" },
  { id: 3, name: "12 hours", address: "0xSimpleOracleAddress3" },
  { id: 4, name: "24 hours", address: "0xSimpleOracleAddress4" },
];
const StrategyOptions = [{ id: 1, name: "Gammaswap" }];

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

  async function deployContract(accountAddress, selectedOracleAddress) {
    try {
      if (!web3) {
        alert("Web3 not detected. Please install MetaMask.");
        return;
      }

      // Estimate the gas required to deploy the contract
      const gasEstimate = await web3.eth.estimateGas({
        data: contractBytecode,
      });

      // Get the contract instance
      const GammaTrader = new web3.eth.Contract(contractAbi);

      // Deploy the contract with the required constructor arguments
      const deployedContract = await GammaTrader.deploy({
        data: contractBytecode,
        arguments: [selectedOracleAddress],
        value: initialBalanceWei,
      }).send({
        from: accountAddress,
        gas: gasEstimate,
      });
    } catch (error) {
      console.error("Error deploying contract:", error);
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
    deployContract(account, selectedAIModel.url);
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

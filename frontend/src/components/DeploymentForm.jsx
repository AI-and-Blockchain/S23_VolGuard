import Dropdown from "./Dropdown";
import axios from 'axios';
import React, { useState, useContext } from "react";
import WalletContext from "./WalletContext";
const DataSourceOptions = [{ id: 1, name: "amber-data" }];

const AIModelOptions = [
  { id: 1, name: "3 hours" },
  { id: 2, name: "6 hours (reccommended)" },
  { id: 3, name: "12 hours " },
  { id: 4, name: "24 hours " },
];

const StrategyOptions = [{ id: 1, name: "Gammaswap" }];

async function deployContract(accountAddress, selectedModel) {
  try {
    const response = await axios.post('https://johnbartleydev.pythonanywhere.com/deploy', {
      accountaddr: accountAddress,
      model: selectedModel,
    });
    console.log(response.data);
  } catch (error) {
    console.error(error);
  }
}


function DeploymentForm() {
  const [selectedDataSource, setSelectedDataSource] = React.useState(
    DataSourceOptions[0]
  );
  const [selectedAIModel, setSelectedAIModel] = React.useState(
    AIModelOptions[0]
  );
  const [selectedStrategy, setSelectedStrategy] = React.useState(
    StrategyOptions[0]
  );

  const { connected, account } = useContext(WalletContext);
  const handleDeploy = () => {
    if (!account) {
      alert("Please connect your wallet first.");
      return;
    }
    deployContract(account, selectedAIModel.id);
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
            options={AIModelOptions}
            selectedOption={selectedAIModel}
            setSelectedOption={setSelectedAIModel}
          />
          <Dropdown
            label="Strategy"
            options={StrategyOptions}
            selectedOption={selectedStrategy}
            setSelectedOption={setSelectedStrategy}
          />
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

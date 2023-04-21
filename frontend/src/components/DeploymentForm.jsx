import React from "react";
import Dropdown from "./Dropdown";
import axios from 'axios';

const DataSourceOptions = [{ id: 1, name: "amber-data" }];

const AIModelOptions = [
  { id: 1, name: "3 hours" },
  { id: 2, name: "6 hours (reccommended)" },
  { id: 3, name: "12 hours " },
  { id: 4, name: "24 hours " },
];

const StrategyOptions = [{ id: 1, name: "Gammaswap" }];

async function deployContract(accountAddress, privateKey, selectedModel) {
  try {
    const response = await axios.post('https://johnbartleydev.pythonanywhere.com/deploy', {
      accountaddr: accountAddress,
      privatekey: privateKey,
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
  const [accountAddress, setAccountAddress] = React.useState("");
  const [privateKey, setPrivateKey] = React.useState("");

  const handleDeploy = () => {
    deployContract(accountAddress, privateKey, selectedAIModel.id);
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
          <input
            className="input border border-gray-400 p-2 mt-4 w-full"
            type="text"
            placeholder="Account Address"
            value={accountAddress}
            onChange={(e) => setAccountAddress(e.target.value)}
          />
          <input
            className="input border border-gray-400 p-2 mt-4 w-full"
            type="password"
            placeholder="Private Key"
            value={privateKey}
            onChange={(e) => setPrivateKey(e.target.value)}
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

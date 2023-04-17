import React from 'react';
import Dropdown from './Dropdown';

const DataSourceOptions = [
  { id: 1, name: 'amber-data' },
];

const AIModelOptions = [
  { id: 1, name: 'hourly' },
  { id: 2, name: 'daily' },
];

const StrategyOptions = [
  { id: 1, name: 'Gammaswap' },
];

function DeploymentForm() {
  const [selectedDataSource, setSelectedDataSource] = React.useState(DataSourceOptions[0]);
  const [selectedAIModel, setSelectedAIModel] = React.useState(AIModelOptions[0]);
  const [selectedStrategy, setSelectedStrategy] = React.useState(StrategyOptions[0]);

  return (
    <div className="container mx-auto px-4">
      <div className="bg-white rounded-lg shadow-lg p-6 mt-8">
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
        >
          Deploy
        </button>
      </div>
    </div>
  );
}

export default DeploymentForm;

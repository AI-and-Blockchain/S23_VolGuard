import React from 'react';
import { Tab } from '@headlessui/react';
import DeploymentForm from './DeploymentForm';

function classNames(...classes) {
  return classes.filter(Boolean).join(' ');
}

export default function Deploy() {
  return (
      <Tab.Group>
        <Tab.List className="flex py-2 space-x-1 bg-gray-300">
          {['Deploy', 'Dashboard'].map((tabName) => (
            <Tab
              key={tabName}
              className={({ selected }) =>
                classNames(
                  'w-full py-2.5 px-4 text-sm font-medium leading-5',
                  selected ? 'bg-gray-100 bg-opacity-80 shadow-sm rounded-lg mx-2' : ''
                )
              }
            >
              {tabName}
            </Tab>
          ))}
        </Tab.List>
        <Tab.Panels className="mt-4">
          <Tab.Panel>
            <DeploymentForm />
          </Tab.Panel>
          <Tab.Panel>
            {/* Dashboard content will be added here */}
          </Tab.Panel>
        </Tab.Panels>
      </Tab.Group>
  );
}

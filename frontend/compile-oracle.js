import path from "path";
import fs from "fs";
import solc from "solc";

const contractPath = path.resolve(
  new URL("simple-oracle.sol", import.meta.url).pathname
);
const contractSource = fs.readFileSync(contractPath, "utf8");
const input = {
  language: "Solidity",
  sources: {
    "simple-oracle.sol": {
      content: contractSource,
    },
  },
  settings: {
    outputSelection: {
      "*": {
        "*": ["abi", "evm.bytecode"],
      },
    },
  },
};

const output = JSON.parse(solc.compile(JSON.stringify(input)));
const abi = output.contracts["simple-oracle.sol"]["SimpleOracle"].abi;
const bytecode =
  output.contracts["simple-oracle.sol"]["SimpleOracle"].evm.bytecode.object;

fs.writeFileSync("./bytecode.json", JSON.stringify(bytecode));
fs.writeFileSync("./abi.json", JSON.stringify(abi));

console.log(
  "Contract compiled successfully. Bytecode and ABI saved to the project folder."
);

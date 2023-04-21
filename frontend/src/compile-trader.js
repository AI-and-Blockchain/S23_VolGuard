import path from "path";
import fs from "fs";
import solc from "solc";

const __dirname = path.dirname(new URL(import.meta.url).pathname);

const contractPath = path.resolve(
  path.join(__dirname, "contract_files", "gamma-trader.sol")
);
const contractSource = fs.readFileSync(contractPath, "utf8");

const input = {
  language: "Solidity",
  sources: {
    "gamma-trader.sol": {
      content: contractSource,
    },
    "simple-oracle.sol": {
      content: fs.readFileSync(
        path.join(__dirname, "contract_files", "simple-oracle.sol"),
        "utf8"
      ),
    },
    "IUniswapV2Router02.sol": {
      content: fs.readFileSync(
        path.join(__dirname, "contract_files", "IUniswapV2Router02.sol"),
        "utf8"
      ),
    },
    "IUniswapV2Router01.sol": {
      content: fs.readFileSync(
        path.join(__dirname, "contract_files", "IUniswapV2Router01.sol"),
        "utf8"
      ),
    },
    "IERC20.sol": {
      content: fs.readFileSync(
        path.join(__dirname, "contract_files", "IERC20.sol"),
        "utf8"
      ),
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
const abi = output.contracts["gamma-trader.sol"]["GammaTrader"].abi;
const bytecode =
  output.contracts["gamma-trader.sol"]["GammaTrader"].evm.bytecode.object;

fs.writeFileSync(
  path.join(__dirname, "contract_files", "bytecode.json"),
  JSON.stringify(bytecode)
);
fs.writeFileSync(
  path.join(__dirname, "contract_files", "abi.json"),
  JSON.stringify(abi)
);

console.log(
  "Contract compiled successfully. Bytecode and ABI saved to the build folder."
);

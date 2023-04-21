// SPDX-License-Identifier: Apache 2.0
pragma solidity ^0.8.0;

import "./simple-oracle.sol"; // Import the SimpleOracle contract
import "./IUniswapV2Router02.sol"; // Import the Uniswap interface
import "./IUniswapV2Router01.sol"; // Import the Uniswap interface

import "./IERC20.sol"; // Import the ERC20 interface

contract GammaTrader {
    SimpleOracle public oracle;
    IUniswapV2Router02 public uniswapRouter;
    address public ethAddress;
    address public usdcAddress;

    constructor(
        address _oracleAddress,
        address _uniswapRouterAddress,
        address _ethAddress,
        address _usdcAddress
    ) payable {
        oracle = SimpleOracle(_oracleAddress);
        uniswapRouter = IUniswapV2Router02(_uniswapRouterAddress);
        ethAddress = _ethAddress;
        usdcAddress = _usdcAddress;
    }

    function trade() public payable {
        uint256 storedNumber = oracle.getStoredNumber();

        if (storedNumber > 1) {
            // Buy ETH
            uint256 deadline = block.timestamp + 15; // 15 seconds from now
            uniswapRouter.swapExactTokensForETH(
                msg.value,
                0,
                getPathForTokenToETH(),
                address(this),
                deadline
            );
        } else {
            // Swap for ETH-USDC LP
            uint256 deadline = block.timestamp + 15; // 15 seconds from now
            uniswapRouter.addLiquidityETH{value: msg.value}(
                usdcAddress,
                msg.value,
                0,
                0,
                address(this),
                deadline
            );
        }
    }

    function getPathForTokenToETH() private view returns (address[] memory) {
        address[] memory path = new address[](2);
        path[0] = usdcAddress;
        path[1] = ethAddress;
        return path;
    }

    // Function to withdraw accidentally sent tokens
    function withdrawTokens(address _tokenAddress) external {
        uint256 balance = IERC20(_tokenAddress).balanceOf(address(this));
        IERC20(_tokenAddress).transfer(msg.sender, balance);
    }

    // Function to withdraw accidentally sent ETH
    function withdrawETH() external {
        uint256 balance = address(this).balance;
        payable(msg.sender).transfer(balance);
    }
}

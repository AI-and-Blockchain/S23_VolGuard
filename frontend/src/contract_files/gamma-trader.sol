// SPDX-License-Identifier: Apache 2.0
pragma solidity ^0.8.0;

import "./simple-oracle.sol";
import "./IUniswapV2Router02.sol";
import "./IUniswapV2Router01.sol";
import "./IERC20.sol";

contract GammaTrader {
    SimpleOracle public oracle;
    IUniswapV2Router02 public uniswapRouter;
    address public ethAddress;
    address public usdcAddress;
    address public owner;

    event TradeExecuted(address indexed trader, uint256 amount, bool boughtETH);
    event Withdrawal(address indexed to, address token, uint256 amount);

    modifier onlyOwner() {
        require(msg.sender == owner, "Caller is not the owner");
        _;
    }

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
        owner = msg.sender;
    }

    function trade(uint256 minAmountOut) public payable {
        uint256 storedNumber = oracle.getStoredNumber();
        uint256 deadline = block.timestamp + 15;

        if (storedNumber > 1) {
            uint[] memory amounts = uniswapRouter.swapExactTokensForETH(
                msg.value,
                minAmountOut,
                getPathForTokenToETH(),
                address(this),
                deadline
            );
            emit TradeExecuted(msg.sender, amounts[0], true);
        } else {
            (uint amountToken, uint amountETH, ) = uniswapRouter.addLiquidityETH{value: msg.value}(
                usdcAddress,
                msg.value,
                0,
                0,
                address(this),
                deadline
            );
            emit TradeExecuted(msg.sender, amountETH, false);
        }
    }

    function getPathForTokenToETH() private view returns (address[] memory) {
        address[] memory path = new address[](2);
        path[0] = usdcAddress;
        path[1] = ethAddress;
        return path;
    }

    // Function to withdraw accidentally sent tokens
    function withdrawTokens(address _tokenAddress) external onlyOwner {
        uint256 balance = IERC20(_tokenAddress).balanceOf(address(this));
        IERC20(_tokenAddress).transfer(msg.sender, balance);
        emit Withdrawal(msg.sender, _tokenAddress, balance);
    }

    // Function to withdraw accidentally sent ETH
    function withdrawETH() external onlyOwner {
        uint256 balance = address(this).balance;
        payable(msg.sender).transfer(balance);
        emit Withdrawal(msg.sender, ethAddress, balance);
    }
}

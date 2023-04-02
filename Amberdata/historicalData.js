const request = require('request');
const prompt = require('prompt-sync')({sigint: true});
const fs = require("fs");

let userrequest = false;
while (!userrequest) {
  
  let apikey = prompt('Enter API key\n');
  
  

  let request = prompt('What historical pair data is needed\n');
  
  if (request == 'ETH_USDC') {
    ETH_USDC();
    userrequest = true;
  } 
  else if(request == 'KISHU_WETH') {
    KISHU_WETH()
    userrequest = true;
  }
  else if(request == 'WETH_TRU') {
    WETH_TRU()
    userrequest = true;
  }
  else if(request == 'WISE_ETH') {
    WISE_ETH()
    userrequest = true;
  }
  else if(request == 'ELON_ETH') {
    ELON_WETH()
    userrequest = true;
  }
  else if(request == 'DAI_USDC') {
    DAI_USDC()
    userrequest = true;
  }
  else{
    console.log('Incorrect Pair, defaulting to ETH-USDC\n');
    ETH_USDC();
    userrequest = true;
    console.log('Incorrect Pair');
  }
}
function ETH_USDC() {
  const options = {
    method: 'GET',
    url: 'https://web3api.io/api/v2/market/defi/ohlcv/0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640/historical/?exchange=uniswapv3&startDate=2023-01-01&endDate=2023-03-21&timeInterval=days&format=csv&fields=timestamp%2Copen%2Chigh%2Clow%2Cvolume%2Cclose&timeFormat=human_readable',
    headers: {accept: 'application/json', 'x-api-key': apikey}
  };
  
  request(options, function (error, response, body) {
    if (error) throw new Error(error);
  
    console.log(body);
    fs.writeFile("../../historical_files/ETHUSDCdata.csv", body, "utf-8", (err) => {
      if (err) console.log(err);
      else console.log("Data saved");
    });
  });
}
// THESE still rely on amber data api which does not currently work
function  DAI_WETH() {
  sdk.defiOhlcvHistorical({
    exchange: 'uniswapv3',
    startDate: '2023-01-01',
    endDate: '2023-03-21',
    timeInterval: 'days',
    format: 'json',
    pool: 'DAI_WETH'
  })
    .then(({ data }) => console.log(data))
    .catch(err => console.error(err));
}

function  KISHU_WETH() {
    sdk.defiOhlcvHistorical({
        exchange: 'uniswapv3',
        startDate: '2023-01-01',
        endDate: '2023-03-21',
        format: 'csv',
        pool: 'KISHU_WETH'
    })
        .then(({ data }) => console.log(data))
        .catch(err => console.error(err));
}

function  WETH_TRU() {
    sdk.defiOhlcvHistorical({
        exchange: 'uniswapv3',
        startDate: '2023-01-01',
        endDate: '2023-03-21',
        format: 'csv',
        pool: 'WETH_TRU'
    })
        .then(({ data }) => console.log(data))
        .catch(err => console.error(err));
}
function  WISE_ETH() {
    sdk.defiOhlcvHistorical({
        exchange: 'uniswapv3',
        startDate: '2023-01-01',
        endDate: '2023-03-21',
        format: 'csv',
        pool: 'WISE_ETH'
    })
        .then(({ data }) => console.log(data))
        .catch(err => console.error(err));
}

function  ELON_WETH() {
    sdk.defiOhlcvHistorical({
        exchange: 'uniswapv3',
        startDate: '2023-01-01',
        endDate: '2023-03-21',
        format: 'csv',
        pool: 'ELON_WETH'
    })
        .then(({ data }) => console.log(data))
        .catch(err => console.error(err));
}

function  DAI_USDC() {
    sdk.defiOhlcvHistorical({
        exchange: 'uniswapv3',
        startDate: '2023-01-01',
        endDate: '2023-03-21',
        format: 'csv',
        pool: 'USD_ETH'
    })
        .then(({ data }) => console.log(data))
        .catch(err => console.error(err));
}
const sdk = require('api')('@amberdata-api/v2#fyrizmskyulb1zj');
const prompt = require('prompt-sync')({sigint: true});

let userrequest = false;
while (!userrequest) {
  
  let apikey = prompt('Enter API key');
  
  sdk.auth(apikey)

  let request = prompt('What historical pair data is needed');
  
  if (request == 'DAI_WETH') {
    DAI_WETH();
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
    console.log('Incorrect Pair');
  }
}
function  DAI_WETH() {
    sdk.defiOhlcvHistorical({
        exchange: 'uniswapv3',
        startDate: '2023-01-01',
        endDate: '2023-03-21',
        format: 'csv',
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
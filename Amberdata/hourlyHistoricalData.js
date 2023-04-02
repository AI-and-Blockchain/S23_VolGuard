const request = require('request');
const fs = require("fs");
apikey = ''
const options = {
  method: 'GET',
  url: 'https://web3api.io/api/v2/market/defi/ohlcv/0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640/historical/?exchange=uniswapv3&startDate=2023-03-01&endDate=2023-04-01&timeInterval=hours&format=csv&fields=timestamp%2Copen%2Chigh%2Clow%2Cvolume%2Cclose&timeFormat=human_readable',
  headers: {accept: 'application/json', 'x-api-key': apikey}
};

request(options, function (error, response, body) {
  if (error) throw new Error(error);

  console.log(body);
  fs.writeFile("../../historical_files/hourlyETHUSDCdata.csv", body, "utf-8", (err) => {
    if (err) console.log(err);
    else console.log("Data saved");
  });
});
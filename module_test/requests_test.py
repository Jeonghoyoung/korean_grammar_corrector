import requests
import json


if __name__ == '__main__':
    url = 'http://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList?serviceKey=RV5qrbJRiVSlbKPN32nnD3kFJt1g%2BTluv17SHOslrqUvunmsT0BuEKZluh0doSFS7SAYLR4U4O6Sn14uhN4wdA%3D%3D&numOfRows=10&pageNo=1&dataCd=ASOS&dateCd=DAY&startDt=20000101&endDt=20211231&stnIds=108'

    contents = requests.get(url).text
    print(contents)
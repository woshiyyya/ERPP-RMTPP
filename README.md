# A pytorch implementation of ERPP and RMTPP

## Data
```
data/train_day.csv
data/test_day.csv
```

## Requirements

```
pytorch = 0.4.1
numpy = 1.14.2
tqdm = 4.28.1
pandas = 0.23.4
abc
```

## Run ERPP (Event Recurrent Point Process)
With default setting:
```
python main.py --model=erpp
```

## Run ERPP (Recurrent Marked Temporal Point Process)
With default setting:
```
python main.py --model=rmtpp
```
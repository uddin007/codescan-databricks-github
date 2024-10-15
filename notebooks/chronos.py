#!/usr/bin/env python
# coding: utf-8

# ### **Chronos Model:Learning the Language of Time Series**

# ### Models Overview:
# Chronos is a family of pretrained time series forecasting models based on language model architectures. A time series is transformed into a sequence of tokens via scaling and quantization, and a language model is trained on these tokens using the cross-entropy loss. Once trained, probabilistic forecasts are obtained by sampling multiple future trajectories given the historical context. Chronos models have been trained on a large corpus of publicly available time series data, as well as synthetic data generated using Gaussian processes.
# [Original paper publised on 2-May, 2024](https://arxiv.org/abs/2403.07815)

# <img src="https://github.com/amazon-science/chronos-forecasting/blob/main/figures/main-figure.png?raw=true" style="margin-left: 10px"  width="1100px;">

# Figure: High level depictions of Chronos Models. (Left) The input time series is scaled and quantized to obtain a sequence of tokens. (Center) The tokens are fed into a language model which may either be an encoder-decoder or a decoder-only model. The model is trained using the cross-entropy loss. (Right) During inference, we autoregressively sample tokens from the model and map them back to numerical values. Multiple trajectories are sampled to obtain a predictive distribution.

# ### In summary:
#  Chronos tokenizes time series into discrete bins through simple scaling and quantization of real values. In that way, we can train off-the-shelf language models on this “language of time series,” with no changes to the model architecture (above Figure). Remarkably, this straightforward approach proves to be effective and efficient, underscoring the potential for language model architectures to address a broad range of time series problems with minimal modifications.

# ### Zero-Shot Results:
# 
# The following figures showcases the remarkable zero-shot performance of Chronos models on 27 datasets against local models, task-specific models and other pretrained models. For details on the evaluation setup and other results, please refer to the paper.

# <img src="https://github.com/amazon-science/chronos-forecasting/blob/main/figures/zero_shot-agg_scaled_score.png?raw=true" style="margin-left: 10px"  width="1100px;">

# Figure: Performance of different model on Benchmark II, comprising 27 datasets not seen by Chronos models during training. This benchmark provides insights into the zero-shot performance of Chronos models against local statistical models, which fit parameters individually for each time series, task-specific models trained on each task, and pretrained models trained on a large corpus of time series. Pretrained Models (Other) indicates that some (or all) of the datasets in Benchmark II may have been in the training corpus of these models. The probabilistic (WQL: The Weighted Quantile Loss (wQL) error metric measures the accuracy of a model’s forecast at a specified quantile. It is particularly useful when there are different costs for underpredicting and overpredicting) and point (MASE: mean absolute scaled error) forecasting metrics were normalized using the scores of the Seasonal Naive baseline and aggregated through a geometric mean to obtain the Agg. Relative WQL and MASE, respectively.

# ### Architecture:
# The models in this repository are based on the T5 architecture (Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer). The only difference is in the vocabulary size: Chronos-T5 models use 4096 different tokens, compared to 32128 of the original T5 models, resulting in fewer parameters.

# In[ ]:


## %pip install git+https://github.com/amazon-science/chronos-forecasting.git


# In[ ]:


import pandas as pd
sdf = spark.read.csv("/FileStore/tables/service_revenue.csv", inferSchema=True, header=True)
df = sdf.toPandas()
df.tail()


# In[ ]:


df["ds"] = pd.to_datetime(df["ds"]) # convert 'ds' column to datetime
df1 = df[df["ds"] < "2022-09-01"] # filter rows where 'ds' is less than "2022-09-01"
df1.tail()


# In[ ]:


import torch
from chronos import ChronosPipeline

pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map="cpu",  # use "cpu" for CPU inference and "mps" for Apple Silicon
    torch_dtype=torch.bfloat16,
)

# df = pd.read_csv("https://raw.githubusercontent.com/AileenNielsen/TimeSeriesAnalysisWithPython/master/data/AirPassengers.csv")

# context must be either a 1D tensor, a list of 1D tensors,
# or a left-padded 2D tensor with batch as the first dimension
# forecast shape: [num_series, num_samples, prediction_length]
forecast = pipeline.predict(
    context=torch.tensor(df1["y"]),
    prediction_length=21,
    num_samples=20,
)


# In[ ]:


# print(ChronosPipeline.predict.__doc__)


# In[ ]:


df2 = df[(df["ds"] >= "2022-09-01") & (df["ds"] < "2024-06-01")] # filter forecast range
df2.tail()


# In[ ]:


import matplotlib.pyplot as plt  # requires: pip install matplotlib
import numpy as np

forecast_index = range(len(df1), len(df1) + 21)
low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)

plt.figure(figsize=(8, 4))
plt.plot(df1["ds"],df1["y"],color="royalblue", label="historical data")
plt.plot(df2["ds"],df2["y"],color="orange",label="actual data")
plt.plot(df2["ds"], median, color="tomato", label="median forecast")
plt.fill_between(df2["ds"], low, high, color="tomato", alpha=0.3, label="80% prediction interval")
plt.legend()
plt.grid()
plt.show()


# In[ ]:


mape = np.mean(np.abs((df2["y"] - median) / df2["y"])) * 100.00
print(f"MAPE: {mape:.2f}%")


# ### Second scenerio: Future Acquistion (as known event + outcome)

# In[ ]:


df3 = df[df["ds"] < "2021-09-01"]
df3.tail()


# In[ ]:


forecast = pipeline.predict(
    context=torch.tensor(df3["y"]),
    prediction_length=21,
    num_samples=20,
)


# In[ ]:


# print(ChronosPipeline.predict.__doc__)


# In[ ]:


df4 = df[(df["ds"] >= "2021-09-01") & (df["ds"] < "2023-06-01")] # filter forecast range


# In[ ]:


import matplotlib.pyplot as plt  # requires: pip install matplotlib
import numpy as np

low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)

plt.figure(figsize=(8, 4))
plt.plot(df3["ds"],df3["y"],color="royalblue", label="historical data")
plt.plot(df4["ds"],df4["y"],color="orange",label="actual data")
plt.plot(df4["ds"], median, color="tomato", label="median forecast")
plt.fill_between(df4["ds"], low, high, color="tomato", alpha=0.3, label="80% prediction interval")
plt.legend()
plt.grid()
plt.show()


# In[ ]:


periods_aq = 15
median_aq = 45000000
low_aq = 40000000
high_aq = 50000000


# In[ ]:


import matplotlib.pyplot as plt  # requires: pip install matplotlib
import numpy as np

low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)

low[-periods_aq:] = low[-periods_aq:] + low_aq
median[-periods_aq:] = median[-periods_aq:] + median_aq
high[-periods_aq:] = high[-periods_aq:] + high_aq

plt.figure(figsize=(8, 4))
plt.plot(df3["ds"],df3["y"],color="royalblue", label="historical data")
plt.plot(df4["ds"],df4["y"],color="orange",label="actual data")
plt.plot(df4["ds"], median, color="tomato", label="median forecast")
plt.fill_between(df4["ds"], low, high, color="tomato", alpha=0.3, label="80% prediction interval")
plt.legend()
plt.grid()
plt.show()


# In[ ]:


mape = np.mean(np.abs((df4["y"] - median) / df4["y"])) * 100
print(f"MAPE: {mape:.2f}%")


# ### Third scenerio: Not enough adjustment points to predict

# In[ ]:


df5 = df[df["ds"] < "2022-05-01"]
df5.tail()


# In[ ]:


forecast = pipeline.predict(
    context=torch.tensor(df5["y"]),
    prediction_length=21,
    num_samples=20,
)


# In[ ]:


# print(ChronosPipeline.predict.__doc__)


# In[ ]:


df6 = df[(df["ds"] >= "2022-05-01") & (df["ds"] < "2024-02-01")] # Filter forecast range


# In[ ]:


import matplotlib.pyplot as plt  # requires: pip install matplotlib
import numpy as np

low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)

plt.figure(figsize=(8, 4))
plt.plot(df5["ds"],df5["y"],color="royalblue", label="historical data")
plt.plot(df6["ds"],df6["y"],color="orange",label="actual data")
plt.plot(df6["ds"], median, color="tomato", label="median forecast")
plt.fill_between(df6["ds"], low, high, color="tomato", alpha=0.3, label="80% prediction interval")
plt.legend()
plt.grid()
plt.show()


# In[ ]:


periods_aq = 21
median_aq = 45000000
low_aq = 40000000
high_aq = 50000000


# In[ ]:


import matplotlib.pyplot as plt  # requires: pip install matplotlib
import numpy as np

low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)

low[-periods_aq:] = low[-periods_aq:] + low_aq
median[-periods_aq:] = median[-periods_aq:] + median_aq
high[-periods_aq:] = high[-periods_aq:] + high_aq

plt.figure(figsize=(8, 4))
plt.plot(df5["ds"],df5["y"],color="royalblue", label="historical data")
plt.plot(df6["ds"],df6["y"],color="orange",label="actual data")
plt.plot(df6["ds"], median, color="tomato", label="median forecast")
plt.fill_between(df6["ds"], low, high, color="tomato", alpha=0.3, label="80% prediction interval")
plt.legend()
plt.grid()
plt.show()


# In[ ]:


mape = np.mean(np.abs((df6["y"] - median) / df6["y"])) * 100
print(f"MAPE: {mape:.2f}%")

